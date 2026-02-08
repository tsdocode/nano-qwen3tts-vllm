import os
import json

import torch
from typing import Optional
import time
from tqdm import tqdm
from safetensors.torch import load_file

from nano_qwen3tts_vllm.engine.model_runner.base import ModelRunner
from nano_qwen3tts_vllm.config import Qwen3TTSConfig
from nano_qwen3tts_vllm.models.qwen3_tts_talker import Qwen3TTSTalkerForCausalLM
from nano_qwen3tts_vllm.engine.sequence import Sequence, SequenceStatus
from nano_qwen3tts_vllm.sampling_params import SamplingParams

from nano_qwen3tts_vllm.utils.context import set_context, get_context, reset_context
from nano_qwen3tts_vllm.config import Config
from multiprocessing.synchronize import Event

from logging import getLogger

logger = getLogger(__name__)


class TalkerModeModelRunner(ModelRunner):
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        super().__init__(config, rank, event)
        self.model = self.load_model(config)
        self.post_init(rank)
        
        # if not config.enforce_eager:
        #     self.capture_cudagraph_prefill()

    def load_model(self, config: Config):
        with open(os.path.join(config.model, "config.json"), "r") as f:
            model_config = json.load(f)
            model_config = Qwen3TTSConfig(**model_config)
        
        self.full_config = model_config
            
        model = Qwen3TTSTalkerForCausalLM(model_config.talker_config)
        
        self.model_config = model_config.talker_config
        
        state_dict = load_file(
            os.path.join(config.model, "model.safetensors")
        )
        model.load_state_dict(state_dict)   
        return model

    
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool, input_embeds: Optional[torch.Tensor] = None):
        start = time.time()
        model_input = input_embeds if input_embeds is not None else input_ids
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            hidden_states = self.model(model_input, positions)
        else:
            if is_prefill:
                logger.info(f"[talker mode model runner] Running prefill graph")
                num_tokens = model_input.size(0)
                context = get_context()
                num_seqs = context.cu_seqlens_q.size(0) - 1
                # Prefill graph only supports single sequence, no prefix cache,
                # and token count within captured range
                if num_seqs == 1 and num_tokens < 256 and context.block_tables is None:
                    graph = self.graphs_prefill[num_tokens]
                    graph_vars = self.graph_vars_prefill
                    graph_vars["input_embeds"][:num_tokens] = model_input
                    graph_vars["positions"][:num_tokens] = positions
                    graph_vars["cu_seqlens_q"][0] = 0
                    graph_vars["cu_seqlens_q"][1] = num_tokens
                    graph_vars["cu_seqlens_k"][0] = 0
                    graph_vars["cu_seqlens_k"][1] = num_tokens
                    graph_vars["slot_mapping"].fill_(-1)
                    graph_vars["slot_mapping"][:num_tokens] = context.slot_mapping
                    start_time = time.time()
                    graph.replay()
                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000
                    logger.info(f"[talker mode model runner] Prefill graph latency: {latency_ms:.2f}ms")
                    hidden_states = graph_vars["outputs"][:num_tokens]
                else:
                    hidden_states = self.model(model_input, positions)
            else:
                bs = input_embeds.size(0)
                context = get_context()
                graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
                graph_vars = self.graph_vars
                graph_vars["input_embeds"][:bs] = model_input
                graph_vars["positions"][:bs] = positions
                graph_vars["slot_mapping"].fill_(-1)
                graph_vars["slot_mapping"][:bs] = context.slot_mapping
                graph_vars["context_lens"].zero_()
                graph_vars["context_lens"][:bs] = context.context_lens
                graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
                graph.replay()
                
                hidden_states = graph_vars["outputs"][:bs]
            
        logits = self.model.compute_logits(hidden_states)
        
        if is_prefill:
            context = get_context()
            last_indices = context.cu_seqlens_q[1:] - 1
            hidden_states = hidden_states[last_indices].contiguous()
        
        logger.info(f"[talker mode model runner] Model run latency: {time.time() - start}")

        return logits, hidden_states

    def prepare_decode_talker(self, seqs: list[Sequence]):
        positions = []
        slot_mapping = []
        context_lens = []
        input_embeds_list = []
        for seq in seqs:
            emb = seq.decode_input_embeds
            if emb is None:
                raise ValueError(f"Sequence {seq.seq_id} has no decode_input_embeds set")
            input_embeds_list.append(emb)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        input_embeds = torch.cat([e.reshape(-1, e.shape[-1]) if e.dim() > 1 else e.unsqueeze(0) for e in input_embeds_list], dim=0).to(dtype=torch.bfloat16)
        if input_embeds.device.type != "cuda":
            input_embeds = input_embeds.pin_memory().cuda(non_blocking=True)
        else:
            input_embeds = input_embeds.cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        input_ids = torch.zeros(len(seqs), dtype=torch.int64, device="cuda")
        return input_ids, input_embeds, positions

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_embeds = None
        if is_prefill:
            input_ids, input_embeds, positions = self.prepare_prefill(seqs)
        else:
            input_ids, input_embeds, positions = self.prepare_decode_talker(seqs)
            
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits, hidden_states = self.run_model(input_ids, positions, is_prefill, input_embeds)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids, hidden_states
    
    
    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = self.model_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        input_embeds = torch.zeros(max_bs, hf_config.hidden_size)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_embeds[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_embeds[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_embeds=input_embeds,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
        
    @torch.inference_mode()
    def capture_cudagraph_prefill(self):
        config = self.config
        hf_config = self.model_config
        max_num_tokens = 256
        input_embeds = torch.zeros(max_num_tokens, hf_config.hidden_size, device="cuda")
        positions = torch.zeros(max_num_tokens, dtype=torch.int64, device="cuda")
        slot_mapping = torch.zeros(max_num_tokens, dtype=torch.int32, device="cuda")
        cu_seqlens_q = torch.zeros(2, dtype=torch.int32, device="cuda")
        cu_seqlens_k = torch.zeros(2, dtype=torch.int32, device="cuda")
        outputs = torch.zeros(max_num_tokens, hf_config.hidden_size, device="cuda")
        self.graph_bs_prefill = range(1, max_num_tokens)
        self.graphs_prefill = {}
        graph_pool = self.graph_pool

        for bs in tqdm(reversed(self.graph_bs_prefill)):
            graph = torch.cuda.CUDAGraph()
            cu_seqlens_q[0] = 0
            cu_seqlens_q[1] = bs
            cu_seqlens_k[0] = 0
            cu_seqlens_k[1] = bs
            set_context(True, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=bs, max_seqlen_k=bs, slot_mapping=slot_mapping[:bs])
            outputs[:bs] = self.model(input_embeds[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, graph_pool):
                outputs[:bs] = self.model(input_embeds[:bs], positions[:bs])    # capture
            if graph_pool is None:
                graph_pool = graph.pool()
            self.graphs_prefill[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars_prefill = dict(
            input_embeds=input_embeds,
            positions=positions,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            slot_mapping=slot_mapping,
            outputs=outputs,
        )
