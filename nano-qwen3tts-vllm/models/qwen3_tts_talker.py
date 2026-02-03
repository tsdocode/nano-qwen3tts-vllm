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


from nano_qwen3tts_vllm.layers.embed_head import ParallelLMHead
from nano_qwen3tts_vllm.layers.layernorm import Qwen3TTSRMSNorm
from nano_qwen3tts_vllm.models.qwen3_tts_share import Qwen3TTSDecoderLayer




class Qwen3TTSTalkerResizeMLP(nn.Module):
    def __init__(self, input_size: int, intermediate_size: int, output_size: int, act: str, bias=False):
        super().__init__()
        self.linear_fc1 = nn.Linear(input_size, intermediate_size, bias=bias)
        self.linear_fc2 = nn.Linear(intermediate_size, output_size, bias=bias)
        self.act_fn = torch.nn.functional.silu

    @torch.compile
    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3TTSTalkerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.layers = nn.ModuleList([Qwen3TTSDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Text embedding for processing text input (vocab_size=151936, dim=2048)
        self.text_embedding = nn.Embedding(config.text_vocab_size, config.text_hidden_size)

    def get_input_embeddings(self):
        """Get codec embedding layer."""
        return self.codec_embedding
    
    def get_text_embeddings(self):
        """Get text embedding layer."""
        return self.text_embedding

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


class Qwen3TTSTalkerForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3TTSTalkerModel(config)
        self.codec_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        
        # Text projection: projects text embeddings (2048-dim) to talker hidden size (1024-dim)
        self.text_projection = Qwen3TTSTalkerResizeMLP(
            config.text_hidden_size,    # 2048 input
            config.text_hidden_size,    # 2048 intermediate
            config.hidden_size,         # 1024 output
            config.hidden_act,          # "silu"
            bias=True
        )
    
    def convert_state_dict(self, state_dict):
        """Transform state dict weights to match model architecture.

        Handles:
        1. Removing "talker." prefix from all keys
        2. Skipping code_predictor and speaker_encoder keys (not part of talker model)
        3. MLP uses separate gate_proj, up_proj, down_proj (no fusion)
        4. Attention uses separate q_proj, k_proj, v_proj (no fusion)
        """
        transformed = {}

        for key, value in state_dict.items():
            # Skip code_predictor keys (used by predictor model, not talker)
            if key.startswith("talker.code_predictor."):
                continue
            
            # Skip speaker_encoder keys (not part of talker model architecture)
            if key.startswith("speaker_encoder."):
                continue

            # Remove "talker." prefix
            if key.startswith("talker."):
                key_without_prefix = key[7:]  # Remove "talker."
            else:
                key_without_prefix = key

            transformed[key_without_prefix] = value

        return transformed

    def load_state_dict(self, state_dict, strict=True):
        state_dict = self.convert_state_dict(state_dict)
        super().load_state_dict(state_dict, strict=strict)

    def get_input_embeddings(self):
        """Get codec embedding layer."""
        return self.model.get_input_embeddings()
    
    def get_text_embeddings(self):
        """Get text embedding layer."""
        return self.model.get_text_embeddings()

    def forward(
        self,
        input_embeds: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:        
        hidden_states = self.model(input_embeds, positions)
        return hidden_states
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.codec_head(hidden_states)
        return logits