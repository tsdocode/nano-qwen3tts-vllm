import torch
from nano_qwen3tts_vllm.engine.llm_engine.base import LLMEngine
from nano_qwen3tts_vllm.engine.sequence import Sequence, SequenceStatus
from nano_qwen3tts_vllm.engine.scheduler import Scheduler
from nano_qwen3tts_vllm.sampling_params import SamplingParams
from nano_qwen3tts_vllm.engine.model_runner.talker_mode_runner import TalkerModeModelRunner


from nano_qwen3tts_vllm.config import Config
import logging

logger = logging.getLogger(__name__)


class TalkerScheduler(Scheduler):
    def __init__(self, config: Config):
        super().__init__(config)
        self.request_id_to_seq: dict[str, Sequence] = {}

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill: same as base
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            logger.info(f"[talker scheduler] Scheduled {len(scheduled_seqs)} sequences for prefill")
            return scheduled_seqs, True

        # decode: only schedule seqs that have decode_input_embeds set (interface has fed next input)
        # Iterate at most once over running to avoid infinite loop when all seqs wait for decode input
        run_count = len(self.running)
        for _ in range(run_count):
            if not self.running or num_seqs >= self.max_num_seqs:
                break
            seq = self.running.popleft()
            if len(seq) > 0 and seq.decode_input_embeds is None:
                self.running.append(seq)
                continue
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        if not scheduled_seqs:
            return [], False
        self.running.extendleft(reversed(scheduled_seqs))
        logger.info(f"[talker scheduler] Scheduled {len(scheduled_seqs)} sequences for decode")
        return scheduled_seqs, False

    def clear_request(self, request_id: str):
        if request_id in self.request_id_to_seq:
            seq = self.request_id_to_seq.pop(request_id)
            self.block_manager.deallocate(seq)
            if seq in self.running:


                self.running.remove(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], hidden_states: list[torch.Tensor]):
        idx = 0
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id, hidden_states[idx])
            seq.decode_input_embeds = None
            idx += 1
            if seq.request_id is not None:
                finish = not seq.ignore_eos and token_id == self.eos
            else:
                finish = (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens >= seq.max_tokens
            if finish:
                seq.status = SequenceStatus.FINISHED
                if seq.request_id is not None:
                    self.request_id_to_seq.pop(seq.request_id, None)
                self.block_manager.deallocate(seq)
                self.running.remove(seq)



class TalkerLLMEngine(LLMEngine):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.model_runner = TalkerModeModelRunner(self.config, 0, self.events)
        self.scheduler = TalkerScheduler(self.config)

    def add_request(
        self,
        inputs_embeds: list[torch.Tensor],
        sampling_params: SamplingParams | list[SamplingParams],
        request_id: str | None = None,
    ):
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(inputs_embeds)
        for inp_embeds, sp in zip(inputs_embeds, sampling_params):
            if request_id is not None and request_id in self.scheduler.request_id_to_seq:
                seq = self.scheduler.request_id_to_seq[request_id]
                seq.decode_input_embeds = inp_embeds
                return
            seq = Sequence([], input_embeds=inp_embeds, sampling_params=sp, request_id=request_id)
            if request_id is not None:
                self.scheduler.request_id_to_seq[request_id] = seq
            self.scheduler.add(seq)

    def clear_request(self, request_id: str):
        self.scheduler.clear_request(request_id)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return [], 0
        token_ids, hidden_states = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, hidden_states)
        outputs = [(seq.request_id, seq.seq_id, seq.completion_token_ids, seq.last_hidden_state) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def step_with_outputs(self):
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return [], 0, []
        
        token_ids, hidden_states = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, hidden_states)
        outputs = [(seq.request_id, seq.seq_id, seq.completion_token_ids, seq.last_hidden_state) for seq in seqs if seq.is_finished]
        outputs_all = [(seq.request_id, seq.seq_id, seq.completion_token_ids, seq.last_hidden_state, seq.is_finished) for seq in seqs]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens, outputs_all
            