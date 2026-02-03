"""Talker model adapted for nano-vllm style continuous batching.

This module adapts Qwen3TTSTalkerForConditionalGeneration to work with nano-vllm's
architecture by:
1. Removing the nested code_predictor.generate() call
2. Simplifying forward() to only generate codebook 0
3. Adding methods compatible with nano-vllm's model runner
"""

import torch
from torch import nn
import torch.distributed as dist
from nano_qwen3tts_vllm.engine.model_runner.base import get_context


from nano_qwen3tts_vllm.layers.layernorm import Qwen3TTSRMSNorm
from nano_qwen3tts_vllm.models.qwen3_tts_share import Qwen3TTSDecoderLayer

class Qwen3TTSCodePredictorModel(nn.Module):
    def __init__(self, config, talker_config=None):
        super().__init__()
        self.vocab_size = config.vocab_size
        
        codec_embedding_dim = talker_config.hidden_size if talker_config else config.hidden_size
         
        self.layers = nn.ModuleList([Qwen3TTSDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.codec_embedding = nn.ModuleList(
            [nn.Embedding(config.vocab_size, codec_embedding_dim) for _ in range(config.num_code_groups - 1)]
        )

    def forward(
        self,
        input_embeds: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = input_embeds
        
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
            
            
        hidden_states = self.norm(hidden_states)
        
        
        return hidden_states


class Qwen3TTSCodePredictorForCausalLM(nn.Module):
    def __init__(self, config, talker_config):
        super().__init__()
        self.model = Qwen3TTSCodePredictorModel(config, talker_config)
        self.vocab_size = config.vocab_size
        
        print(config.hidden_size, talker_config.hidden_size)
        self.lm_head = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_code_groups - 1)]
        )
        
        if config.hidden_size != talker_config.hidden_size:
            self.small_to_mtp_projection = torch.nn.Linear(talker_config.hidden_size, config.hidden_size, bias=True)
        else:
            self.small_to_mtp_projection = torch.nn.Identity()
            
    def convert_state_dict(self, state_dict):
        """
        Convert state dict from original format to model format.

        Handles:
        1. Extracting code_predictor keys (remove "talker.code_predictor." prefix)
        2. No fusion: attention uses separate q_proj, k_proj, v_proj; MLP uses separate gate_proj, up_proj
        (Same approach as Qwen3TTSTalkerForCausalLM.convert_state_dict.)
        """
        transformed = {}

        for key, value in state_dict.items():
            if not key.startswith("talker.code_predictor."):
                continue
            key_without_prefix = key.replace("talker.code_predictor.", "", 1)
            transformed[key_without_prefix] = value

        return transformed
    
    def load_state_dict(self, state_dict, strict=True):
        state_dict = self.convert_state_dict(state_dict)
        
        super().load_state_dict(state_dict, strict=strict)
        
    def get_input_embeddings(self, input_ids, input_embeds, generation_steps):
        input_embeds_final = []
        if input_embeds is not None and input_embeds.shape[1] > 1:
            generation_steps = input_embeds.shape[1] - 2  # hidden & layer 0
            input_embeds_final = [input_embeds]
        # Generation stage
        else:
            for i, ids in enumerate(input_ids):
                input_embeds_final.append(self.model.codec_embedding[generation_steps[i]-1](ids))
                
                
        input_embeds_final = torch.stack(input_embeds_final)
        return input_embeds_final
            
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds = self.small_to_mtp_projection(inputs_embeds)
        hidden_states = self.model(inputs_embeds, positions)
        return hidden_states
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        generation_steps: list[int],
    ) -> torch.Tensor:
        # hidden_states: (total_tokens, hidden_size) from model forward
        # We need one logit vector per sequence (last position) for the sampler: (num_seqs, vocab_size)
        final_logits = []
        hidden_states = hidden_states.view(len(generation_steps), -1, hidden_states.shape[-1])
        for idx, generation_step in enumerate(generation_steps):
            logits = self.lm_head[generation_step](hidden_states[idx])
            final_logits.append(logits.unsqueeze(0))
        final_logits = torch.cat(final_logits, dim=0)
        # Take last token logits per sequence so sampler gets (num_seqs, vocab_size)
        return final_logits[:, -1, :]

