"""
Async bridge and ZMQ clients for talker and predictor worker processes.
Main process binds PUSH/PULL, spawns workers, runs a result-bridge thread that completes
asyncio Futures when worker results arrive.
"""

import asyncio
import logging
import multiprocessing as mp
import os
import threading
import uuid

try:
    import zmq
except ImportError:
    zmq = None

from nano_qwen3tts_vllm.zmq.utils import find_available_port

from nano_qwen3tts_vllm.workers.protocol import (
    serialize_talker_add_request,
    serialize_talker_run_step,
    serialize_clear_request,
    serialize_shutdown,
    deserialize_talker_result,
    serialize_predictor_add_request,
    serialize_predictor_run_step,
    deserialize_predictor_result,
)

logger = logging.getLogger(__name__)


def _run_result_bridge_thread(
    talker_result_pull,
    predictor_result_pull,
    pending_talker_futures: dict,
    pending_predictor_futures: dict,
    loop: asyncio.AbstractEventLoop,
    stop_event: threading.Event,
) -> None:
    """Thread: poll both result PULL sockets and complete the corresponding Futures."""
    poller = zmq.Poller()
    poller.register(talker_result_pull, zmq.POLLIN)
    poller.register(predictor_result_pull, zmq.POLLIN)
    while not stop_event.is_set():
        try:
            evts = dict(poller.poll(timeout=100))
        except Exception as e:
            if stop_event.is_set():
                break
            logger.exception(f"[result_bridge] poll error: {e}")
            continue
        if talker_result_pull in evts:
            try:
                msg = talker_result_pull.recv()
                step_id, outputs_all = deserialize_talker_result(msg)
                fut = pending_talker_futures.pop(step_id, None)
                if fut is not None and not fut.done():
                    loop.call_soon_threadsafe(fut.set_result, ("talker", outputs_all))
            except Exception as e:
                logger.warning(f"[result_bridge] talker result: {e}")
        if predictor_result_pull in evts:
            try:
                msg = predictor_result_pull.recv()
                step_id, outputs_all = deserialize_predictor_result(msg)
                fut = pending_predictor_futures.pop(step_id, None)
                if fut is not None and not fut.done():
                    loop.call_soon_threadsafe(fut.set_result, ("predictor", outputs_all))
            except Exception as e:
                logger.warning(f"[result_bridge] predictor result: {e}")


def _sampling_params_to_dict(sp) -> dict:
    return {
        "temperature": sp.temperature,
        "max_tokens": sp.max_tokens,
        "ignore_eos": sp.ignore_eos,
        "do_sample": sp.do_sample,
        "top_k": sp.top_k,
        "top_p": sp.top_p,
    }


class TalkerWorkerClient:
    """Sends commands to talker worker and provides run_step_async() returning a Future."""

    def __init__(
        self,
        command_bind_addr: str,
        pending_talker_futures: dict,
        talker_ready: set,
        loop: asyncio.AbstractEventLoop,
    ):
        if zmq is None:
            raise ImportError("pyzmq required")
        self._ctx = zmq.Context()
        self._push = self._ctx.socket(zmq.PUSH)
        self._push.setsockopt(zmq.LINGER, 0)
        self._push.bind(command_bind_addr)
        self._pending = pending_talker_futures
        self._talker_ready = talker_ready
        self._loop = loop

    def send_add_request(self, request_id: str, inputs_embeds, sampling_params) -> None:
        sp_dict = _sampling_params_to_dict(sampling_params)
        payload = serialize_talker_add_request(request_id, inputs_embeds, sp_dict)
        self._push.send(payload)
        self._talker_ready.add(request_id)

    def send_clear_request(self, request_id: str) -> None:
        self._push.send(serialize_clear_request(request_id))

    def run_step_async(self) -> asyncio.Future:
        """Send run_step to worker; return a Future that resolves to outputs_all (list of 5-tuples)."""
        step_id = str(uuid.uuid4())
        future = self._loop.create_future()
        self._pending[step_id] = future
        self._push.send(serialize_talker_run_step(step_id))
        return future

    def send_shutdown(self) -> None:
        self._push.send(serialize_shutdown())

    def close(self) -> None:
        try:
            self._push.close()
            self._ctx.term()
        except Exception:
            pass


class PredictorWorkerClient:
    """Sends commands to predictor worker; run_step_async() returns Future of (outputs_all)."""

    def __init__(
        self,
        command_bind_addr: str,
        pending_predictor_futures: dict,
        predictor_ready: set,
        loop: asyncio.AbstractEventLoop,
    ):
        if zmq is None:
            raise ImportError("pyzmq required")
        self._ctx = zmq.Context()
        self._push = self._ctx.socket(zmq.PUSH)
        self._push.setsockopt(zmq.LINGER, 0)
        self._push.bind(command_bind_addr)
        self._pending = pending_predictor_futures
        self._predictor_ready = predictor_ready
        self._loop = loop

    def send_add_request(self, request_id: str, inputs_embeds, sampling_params) -> None:
        sp_dict = _sampling_params_to_dict(sampling_params)
        payload = serialize_predictor_add_request(request_id, inputs_embeds, sp_dict)
        self._push.send(payload)
        self._predictor_ready.add(request_id)

    def send_clear_request(self, request_id: str) -> None:
        self._push.send(serialize_clear_request(request_id))

    def run_step_async(self) -> asyncio.Future:
        """Send run_step; return Future that resolves to outputs_all (list of (request_id, seq_id, token_ids))."""
        step_id = str(uuid.uuid4())
        future = self._loop.create_future()
        self._pending[step_id] = future
        self._push.send(serialize_predictor_run_step(step_id))
        return future

    def send_shutdown(self) -> None:
        self._push.send(serialize_shutdown())

    def close(self) -> None:
        try:
            self._push.close()
            self._ctx.term()
        except Exception:
            pass


def start_multiprocess_engines(
    model_path: str,
    request_queues: dict,
    queues_lock: asyncio.Lock,
    *,
    gpu_memory_utilization: float = 0.9,
    enforce_eager: bool = False,
    tensor_parallel_size: int = 1,
):
    """
    Bind ZMQ sockets, start result-bridge thread, create clients, spawn talker and predictor
    worker processes. Returns an object with .talker_client, .predictor_client, .stop_async().
    """
    if zmq is None:
        raise ImportError("pyzmq required for multiprocess engines")
    loop = asyncio.get_event_loop()
    base_port = find_available_port(9600, max_attempts=100)
    addrs = {
        "talker_command": f"tcp://127.0.0.1:{base_port}",
        "talker_result": f"tcp://127.0.0.1:{base_port + 1}",
        "predictor_command": f"tcp://127.0.0.1:{base_port + 2}",
        "predictor_result": f"tcp://127.0.0.1:{base_port + 3}",
    }
    ctx = zmq.Context()
    talker_result_pull = ctx.socket(zmq.PULL)
    talker_result_pull.setsockopt(zmq.LINGER, 0)
    talker_result_pull.bind(addrs["talker_result"])
    predictor_result_pull = ctx.socket(zmq.PULL)
    predictor_result_pull.setsockopt(zmq.LINGER, 0)
    predictor_result_pull.bind(addrs["predictor_result"])

    pending_talker = {}
    pending_predictor = {}
    talker_ready = set()
    predictor_ready = set()
    stop_event = threading.Event()
    bridge = threading.Thread(
        target=_run_result_bridge_thread,
        args=(
            talker_result_pull,
            predictor_result_pull,
            pending_talker,
            pending_predictor,
            loop,
            stop_event,
        ),
        daemon=True,
    )
    bridge.start()

    talker_client = TalkerWorkerClient(
        addrs["talker_command"],
        pending_talker,
        talker_ready,
        loop,
    )
    predictor_client = PredictorWorkerClient(
        addrs["predictor_command"],
        pending_predictor,
        predictor_ready,
        loop,
    )

    from nano_qwen3tts_vllm.workers.talker_worker import run_talker_worker
    from nano_qwen3tts_vllm.workers.predictor_worker import run_predictor_worker

    # Split user's GPU_MEMORY_UTILIZATION between the two workers (e.g. 0.3 → 0.15 each).
    per_worker_util = gpu_memory_utilization / 2.0

    ctx_spawn = mp.get_context("spawn")
    talker_proc = ctx_spawn.Process(
        target=run_talker_worker,
        args=(addrs["talker_command"], addrs["talker_result"], model_path),
        kwargs=dict(
            gpu_memory_utilization=per_worker_util,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tensor_parallel_size,
        ),
    )
    predictor_proc = ctx_spawn.Process(
        target=run_predictor_worker,
        args=(addrs["predictor_command"], addrs["predictor_result"], model_path),
        kwargs=dict(
            gpu_memory_utilization=per_worker_util,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tensor_parallel_size,
        ),
    )
    talker_proc.start()
    predictor_proc.start()
    logger.info(
        f"[multiprocess_engines] started talker pid={talker_proc.pid} predictor pid={predictor_proc.pid}"
    )

    class Holder:
        def __init__(self):
            self.talker_client = talker_client
            self.predictor_client = predictor_client
            self.talker_ready = talker_ready
            self.predictor_ready = predictor_ready
            self._stop_event = stop_event
            self._talker_proc = talker_proc
            self._predictor_proc = predictor_proc
            self._request_queues = request_queues
            self._queues_lock = queues_lock

        async def stop_async(self):
            self._stop_event.set()
            talker_client.send_shutdown()
            predictor_client.send_shutdown()
            self._talker_proc.join(timeout=10.0)
            if self._talker_proc.is_alive():
                self._talker_proc.terminate()
                self._talker_proc.join(timeout=5.0)
            self._predictor_proc.join(timeout=10.0)
            if self._predictor_proc.is_alive():
                self._predictor_proc.terminate()
                self._predictor_proc.join(timeout=5.0)
            talker_client.close()
            predictor_client.close()
            try:
                talker_result_pull.close()
                predictor_result_pull.close()
                ctx.term()
            except Exception:
                pass

    return Holder()
