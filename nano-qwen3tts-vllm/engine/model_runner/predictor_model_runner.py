import os
import json
import time
import torch
from typing import Optional
from tqdm import tqdm

from safetensors.torch import load_file

from nano_qwen3tts_vllm.engine.model_runner.base import ModelRunner
from nano_qwen3tts_vllm.config import Qwen3TTSConfig
from nano_qwen3tts_vllm.models.qwen3_tts_predictor import Qwen3TTSCodePredictorForCausalLM
from nano_qwen3tts_vllm.engine.sequence import Sequence
from nano_qwen3tts_vllm.sampling_params import SamplingParams

from nano_qwen3tts_vllm.utils.context import set_context, get_context, reset_context
from nano_qwen3tts_vllm.config import Config
from multiprocessing.synchronize import Event


from logging import getLogger
logger = getLogger(__name__)


class PredictorSequence(Sequence):
    def __init__(self, token_ids: Optional[list[int]], sampling_params = SamplingParams(), input_embeds: Optional[torch.Tensor] = None, generation_steps: int = 0, request_id: Optional[str] = None):
        super().__init__(token_ids, sampling_params, input_embeds, request_id=request_id)
        self.generation_steps = generation_steps


class PredictorModelRunner(ModelRunner):
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        super().__init__(config, rank, event)
        self.model = self.load_model(config)
        self.graphs_prefill = {}  # set before post_init; warmup_model() may call run_model()
        self.post_init(rank)
        if not config.enforce_eager:
            self.capture_cudagraph_prefill()

    def load_model(self, config: Config):
        with open(os.path.join(config.model, "config.json"), "r") as f:
            model_config = json.load(f)
            model_config = Qwen3TTSConfig(**model_config)
        
        model_config.talker_config.code_predictor_config.talker_hidden_size = model_config.talker_config.hidden_size
        model = Qwen3TTSCodePredictorForCausalLM(model_config.talker_config.code_predictor_config, model_config.talker_config)
        
        self.model_config = model_config.talker_config.code_predictor_config
        
        state_dict = load_file(
            os.path.join(config.model, "model.safetensors")
        )
        model.load_state_dict(state_dict)   
        return model
    
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([], input_embeds=torch.zeros(1, 8, self.model_config.talker_hidden_size)) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    
    @torch.inference_mode()
    def run_model(
        self,
        positions: torch.Tensor,
        input_embeds: Optional[torch.Tensor] = None,
        is_prefill: bool = False,
        generation_steps: list[int] = [],
    ) -> torch.Tensor:
        start = time.time()        
        if self.enforce_eager or input_embeds.size(0) > 512:
            hidden_states = self.model(input_embeds, positions)
        elif is_prefill:
            # get_input_embeddings returns 3D [1, total_tokens, hidden] for prefill
            # via torch.stack; flatten to 2D [total_tokens, hidden] for flash attention
            if input_embeds.dim() == 3:
                input_embeds = input_embeds.view(-1, input_embeds.size(-1))
                
                
            num_tokens = input_embeds.size(0)
            context = get_context()
            num_seqs = context.cu_seqlens_q.size(0) - 1
            # Prefill graph: use when we have a captured (num_seqs, num_tokens) and no prefix cache
            key = (num_seqs, num_tokens)
            if key in self.graphs_prefill and context.block_tables is None:
                logger.info(f"[predictor model runner] Running prefill graph with num_seqs={num_seqs} num_tokens={num_tokens}")
                graph = self.graphs_prefill[key]
                graph_vars = self.graph_vars_prefill
                graph_vars["input_embeds"][:num_tokens] = input_embeds
                graph_vars["positions"][:num_tokens] = positions
                cu_len = num_seqs + 1
                graph_vars["cu_seqlens_q"][:cu_len] = context.cu_seqlens_q
                graph_vars["cu_seqlens_k"][:cu_len] = context.cu_seqlens_k
                graph_vars["slot_mapping"].fill_(-1)
                graph_vars["slot_mapping"][:num_tokens] = context.slot_mapping
                # Model reads from get_context(); point context at graph buffers for replay
                set_context(
                    True,
                    cu_seqlens_q=graph_vars["cu_seqlens_q"][:cu_len],
                    cu_seqlens_k=graph_vars["cu_seqlens_k"][:cu_len],
                    max_seqlen_q=context.max_seqlen_q,
                    max_seqlen_k=context.max_seqlen_k,
                    slot_mapping=graph_vars["slot_mapping"],
                )
                graph.replay()
                hidden_states = graph_vars["outputs"][:num_tokens]
            else:
                hidden_states = self.model(input_embeds, positions)
        else:
            logger.info(f"[predictor model runner] Running decode graph")
            bs = input_embeds.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_embeds"][:bs] = input_embeds
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            # Use outputs from the graph; do NOT run self.model() again (that would double the work).
            hidden_states = graph_vars["outputs"][:bs]
            
        logits = self.model.compute_logits(hidden_states, generation_steps)        
        return logits
        
    
    def run(self, seqs: list[PredictorSequence], is_prefill: bool) -> list[int]:
        input_embeds = None
        if is_prefill:
            input_ids, input_embeds, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)
            
        generation_steps = [seq.generation_steps for seq in seqs]
            
        input_embeds = self.model.get_input_embeddings(input_ids, input_embeds, generation_steps)

        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        
        try:
            logits = self.run_model(positions, input_embeds, is_prefill, generation_steps)
        except Exception as e:
            logger.error(f"[predictor model runner] Error running model: {e}")
            import traceback
            traceback.print_exc()
            raise e
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        
        reset_context()
        return token_ids
    
    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        input_embeds = torch.zeros(max_bs, self.model_config.talker_hidden_size)
        generation_steps = torch.zeros(max_bs, dtype=torch.int32)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, self.model_config.hidden_size)
        self.graph_bs = list(range(1,16)) + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            input_embeds[:bs].copy_(
                self.model.get_input_embeddings(input_ids[:bs], None, generation_steps[:bs])
            )
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
        max_num_tokens = 36
        # Support multi-sequence prefill: capture for num_seqs in [1,2,4,8]
        max_num_seqs_prefill = min(16, getattr(config, "max_num_seqs", 512))
        prefill_num_seqs_list = [n for n in range(1,8) if n <= max_num_seqs_prefill]
        max_cu_len = max(prefill_num_seqs_list) + 1
        input_embeds = torch.zeros(max_num_tokens, hf_config.talker_hidden_size, device="cuda")
        positions = torch.zeros(max_num_tokens, dtype=torch.int64, device="cuda")
        slot_mapping = torch.zeros(max_num_tokens, dtype=torch.int32, device="cuda")
        cu_seqlens_q = torch.zeros(max_cu_len, dtype=torch.int32, device="cuda")
        cu_seqlens_k = torch.zeros(max_cu_len, dtype=torch.int32, device="cuda")
        outputs = torch.zeros(max_num_tokens, hf_config.hidden_size, device="cuda")
        self.graphs_prefill = {}  # (num_seqs, num_tokens) -> graph
        graph_pool = self.graph_pool

        for num_seqs in prefill_num_seqs_list:
            cu_len = num_seqs + 1
            for bs in tqdm(
                reversed(range(1, max_num_tokens)),
                desc=f"prefill graphs num_seqs={num_seqs}",
                leave=False,
            ):
                if bs < num_seqs:
                    continue
                graph = torch.cuda.CUDAGraph()
                # Equal split of tokens across sequences
                cu_seqlens_q[:cu_len].zero_()
                cu_seqlens_k[:cu_len].zero_()
                per_seq = bs // num_seqs
                for i in range(1, cu_len):
                    cu_seqlens_q[i] = i * per_seq
                    cu_seqlens_k[i] = i * per_seq
                cu_seqlens_q[cu_len - 1] = bs
                cu_seqlens_k[cu_len - 1] = bs
                max_seqlen_q = (bs + num_seqs - 1) // num_seqs
                max_seqlen_k = max_seqlen_q
                set_context(
                    True,
                    cu_seqlens_q=cu_seqlens_q[:cu_len],
                    cu_seqlens_k=cu_seqlens_k[:cu_len],
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    slot_mapping=slot_mapping[:bs],
                )
                outputs[:bs] = self.model(input_embeds[:bs], positions[:bs])  # warmup
                with torch.cuda.graph(graph, graph_pool):
                    outputs[:bs] = self.model(input_embeds[:bs], positions[:bs])  # capture
                if graph_pool is None:
                    graph_pool = graph.pool()
                self.graphs_prefill[(num_seqs, bs)] = graph
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

