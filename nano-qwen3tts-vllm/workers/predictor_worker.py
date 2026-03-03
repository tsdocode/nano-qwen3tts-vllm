"""
Predictor worker process: sync loop that receives commands over ZMQ and runs PredictorLLM.
Runs a burst of step() on run_step until no work (matching current in-process behavior).
"""

import logging
import sys

import torch

try:
    import zmq
except ImportError:
    zmq = None

from nano_qwen3tts_vllm.workers.protocol import (
    deserialize_command,
    serialize_predictor_result,
    CMD_ADD_REQUEST,
    CMD_RUN_STEP,
    CMD_CLEAR_REQUEST,
    CMD_SHUTDOWN,
)
from nano_qwen3tts_vllm.llm import PredictorLLM
from nano_qwen3tts_vllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


def _sampling_params_from_dict(d: dict) -> SamplingParams:
    return SamplingParams(
        temperature=d.get("temperature", 0.9),
        max_tokens=d.get("max_tokens", 17),
        ignore_eos=d.get("ignore_eos", False),
        do_sample=d.get("do_sample", True),
        top_k=d.get("top_k", 50),
        top_p=d.get("top_p", 1.0),
    )


def run_predictor_worker(
    command_connect_addr: str,
    result_connect_addr: str,
    model_path: str,
    *,
    gpu_memory_utilization: float = 0.9,
    enforce_eager: bool = False,
    tensor_parallel_size: int = 1,
) -> None:
    """
    Entrypoint for the predictor worker process. Sync loop: recv command → execute → send result.
    On run_step, runs a burst of step() until no work (no yielding).
    """
    if zmq is None:
        raise ImportError("pyzmq is required for predictor worker. pip install pyzmq")

    from nano_qwen3tts_vllm.interface import _compute_memory_split
    mem_cfg = _compute_memory_split(model_path, gpu_memory_utilization)
    pred_util = mem_cfg["pred_util"]
    proc_frac = mem_cfg["process_gpu_memory_fraction"]
    # Leave headroom so KV cache fits: process limit includes non-PyTorch overhead; pred_util=1.0 can OOM.
    pred_util = min(pred_util, 0.85)

    # Cap this process's GPU memory before loading the model.
    if torch.cuda.is_available():
        try:
            set_frac = getattr(torch.cuda, "set_per_process_memory_fraction", None) or getattr(
                getattr(torch.cuda, "memory", None), "set_per_process_memory_fraction", None
            )
            if set_frac is not None:
                set_frac(proc_frac, 0)
                logger.info(f"[predictor_worker] set_per_process_memory_fraction({proc_frac})")
        except Exception as e:
            logger.warning(f"[predictor_worker] set_per_process_memory_fraction failed: {e}")

    logger.info(
        f"[predictor_worker] loading model from {model_path} "
        f"(gpu_util={pred_util}, proc_frac={proc_frac})"
    )
    predictor_llm = PredictorLLM(
        model_path,
        enforce_eager=enforce_eager,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=pred_util,
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

    logger.info(f"[predictor_worker] connected to {command_connect_addr}, result {result_connect_addr}")

    burst_count = 0
    try:
        while True:
            msg = pull.recv()
            cmd = deserialize_command(msg)

            if cmd.get("cmd") == CMD_SHUTDOWN:
                logger.info("[predictor_worker] received shutdown")
                break

            if cmd.get("cmd") == CMD_ADD_REQUEST:
                request_id = cmd["request_id"]
                inputs_embeds_np = cmd["inputs_embeds"]
                sp_dict = cmd.get("sampling_params", {})
                inputs_embeds = [
                    torch.from_numpy(arr).to(device) for arr in inputs_embeds_np
                ]
                sp = _sampling_params_from_dict(sp_dict)
                predictor_llm.add_request(inputs_embeds, sp, request_id=request_id)
                continue

            if cmd.get("cmd") == CMD_CLEAR_REQUEST:
                predictor_llm.clear_request(cmd["request_id"])
                continue

            if cmd.get("cmd") == CMD_RUN_STEP:
                step_id = cmd["step_id"]
                try:
                    # Burst: run step() until no work (sync, no yield)
                    outputs_all = []
                    while predictor_llm.scheduler.waiting or predictor_llm.scheduler.running:
                        output, _ = predictor_llm.step()
                        # output: list of (request_id, seq_id, token_ids)
                        outputs_all.extend(output)
                    payload = serialize_predictor_result(step_id, outputs_all)
                    push.send(payload)
                    burst_count += 1
                    if burst_count % 50 == 1:
                        logger.info(f"[predictor_worker] burst#{burst_count} finished={len(outputs_all)}")
                except Exception as e:
                    logger.exception(f"[predictor_worker] burst failed: {e}")
                    payload = serialize_predictor_result(step_id, [])
                    push.send(payload)
                    raise
                continue

            logger.warning(f"[predictor_worker] unknown command: {cmd.get('cmd')}")
    finally:
        pull.close()
        push.close()
        ctx.term()
        try:
            predictor_llm.exit()
        except Exception:
            pass
        logger.info("[predictor_worker] exited")


def main():
    """CLI for spawning predictor worker."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--command_connect", required=True)
    parser.add_argument("--result_connect", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_predictor_worker(
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
