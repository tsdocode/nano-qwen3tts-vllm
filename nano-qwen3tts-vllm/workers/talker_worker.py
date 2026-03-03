"""
Talker worker process: sync loop that receives commands over ZMQ and runs TalkerLLM.
Runs in a separate process so the main event loop is never blocked by step_with_outputs().
"""

import logging
import os
import sys

import torch

try:
    import zmq
except ImportError:
    zmq = None

from nano_qwen3tts_vllm.workers.protocol import (
    deserialize_command,
    serialize_talker_result,
    CMD_ADD_REQUEST,
    CMD_RUN_STEP,
    CMD_CLEAR_REQUEST,
    CMD_SHUTDOWN,
)
from nano_qwen3tts_vllm.llm import TalkerLLM
from nano_qwen3tts_vllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


def _sampling_params_from_dict(d: dict) -> SamplingParams:
    return SamplingParams(
        temperature=d.get("temperature", 1.0),
        max_tokens=d.get("max_tokens", 64),
        ignore_eos=d.get("ignore_eos", False),
        do_sample=d.get("do_sample", True),
        top_k=d.get("top_k", 50),
        top_p=d.get("top_p", 1.0),
    )


def run_talker_worker(
    command_connect_addr: str,
    result_connect_addr: str,
    model_path: str,
    *,
    gpu_memory_utilization: float = 0.9,
    enforce_eager: bool = False,
    tensor_parallel_size: int = 1,
) -> None:
    """
    Entrypoint for the talker worker process. Sync loop: recv command → execute → send result.

    - command_connect_addr: ZMQ address where main PUSHes commands (worker PULL connects here).
    - result_bind_addr: ZMQ address where worker PUSHes results (main PULL connects here).
    """
    if zmq is None:
        raise ImportError("pyzmq is required for talker worker. pip install pyzmq")

    # Memory split: worker only loads talker, use same split logic to get talker_util and proc_frac
    from nano_qwen3tts_vllm.interface import _compute_memory_split
    mem_cfg = _compute_memory_split(model_path, gpu_memory_utilization)
    talker_util = mem_cfg["talker_util"]
    proc_frac = mem_cfg["process_gpu_memory_fraction"]
    # Leave headroom so KV cache fits (non-PyTorch overhead + fragmentation).
    talker_util = min(talker_util, 0.85)

    # Cap this process's GPU memory before loading the model (so e.g. 0.3 = at most 30% of GPU).
    if torch.cuda.is_available():
        try:
            set_frac = getattr(torch.cuda, "set_per_process_memory_fraction", None) or getattr(
                getattr(torch.cuda, "memory", None), "set_per_process_memory_fraction", None
            )
            if set_frac is not None:
                set_frac(proc_frac, 0)
                logger.info(f"[talker_worker] set_per_process_memory_fraction({proc_frac})")
        except Exception as e:
            logger.warning(f"[talker_worker] set_per_process_memory_fraction failed: {e}")

    logger.info(
        f"[talker_worker] loading model from {model_path} "
        f"(gpu_util={talker_util}, proc_frac={proc_frac})"
    )
    talker_llm = TalkerLLM(
        model_path,
        enforce_eager=enforce_eager,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=talker_util,
        process_gpu_memory_fraction=proc_frac,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ctx = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    pull.setsockopt(zmq.LINGER, 0)
    pull.connect(command_connect_addr)

    push = ctx.socket(zmq.PUSH)
    push.setsockopt(zmq.LINGER, 0)
    push.connect(result_connect_addr)

    logger.info(f"[talker_worker] connected to {command_connect_addr}, result {result_connect_addr}")

    step_count = 0
    try:
        while True:
            msg = pull.recv()
            cmd = deserialize_command(msg)

            if cmd.get("cmd") == CMD_SHUTDOWN:
                logger.info("[talker_worker] received shutdown")
                break

            if cmd.get("cmd") == CMD_ADD_REQUEST:
                request_id = cmd["request_id"]
                inputs_embeds_np = cmd["inputs_embeds"]
                sp_dict = cmd.get("sampling_params", {})
                inputs_embeds = [
                    torch.from_numpy(arr).to(device) for arr in inputs_embeds_np
                ]
                sp = _sampling_params_from_dict(sp_dict)
                talker_llm.add_request(inputs_embeds, sp, request_id=request_id)
                continue

            if cmd.get("cmd") == CMD_CLEAR_REQUEST:
                talker_llm.clear_request(cmd["request_id"])
                continue

            if cmd.get("cmd") == CMD_RUN_STEP:
                step_id = cmd["step_id"]
                try:
                    _, _, outputs_all = talker_llm.step_with_outputs()
                    # outputs_all: (request_id, seq_id, completion_token_ids, last_hidden_state, is_finished)
                    payload = serialize_talker_result(step_id, outputs_all)
                    push.send(payload)
                    step_count += 1
                    if step_count % 50 == 1:
                        logger.info(f"[talker_worker] step#{step_count} batch_size={len(outputs_all)}")
                except Exception as e:
                    logger.exception(f"[talker_worker] step failed: {e}")
                    # Send error result so main doesn't hang
                    payload = serialize_talker_result(step_id, [])
                    push.send(payload)
                    raise
                continue

            logger.warning(f"[talker_worker] unknown command: {cmd.get('cmd')}")
    finally:
        pull.close()
        push.close()
        ctx.term()
        try:
            talker_llm.exit()
        except Exception:
            pass
        logger.info("[talker_worker] exited")


def main():
    """CLI for spawning talker worker (used when main spawns as subprocess with args)."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--command_connect", required=True, help="ZMQ address to connect for commands")
    parser.add_argument("--result_connect", required=True, help="ZMQ address to connect for results (main PULL)")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_talker_worker(
        args.command_connect,
        args.result_connect,
        args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
