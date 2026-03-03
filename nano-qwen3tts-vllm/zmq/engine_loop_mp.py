"""
Async engine loops when talker and predictor run in separate processes (USE_MULTIPROCESS_ENGINES=1).
Orchestrator: wait until "ready" set matches active requests (or prefill), send run_step to worker,
await result, dispatch to per-request asyncio queues.
"""

import asyncio
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


PREDICTOR_COLLECT_MS = _float_env("PREDICTOR_COLLECT_MS", 3.0)
PREFILL_COLLECT_MS = _float_env("PREFILL_COLLECT_MS", 5.0)


async def run_talker_loop_mp(
    talker_client: Any,
    request_queues: dict,
    queues_lock: asyncio.Lock,
    talker_ready: set,
) -> None:
    """
    Replacement for run_talker_loop when using multiprocess talker.
    Waits until talker_ready matches active requests (or timeout), sends run_step, awaits result,
    dispatches (engine_type, msg_type, payload) to request_queues[request_id].
    """
    step_count = 0
    while True:
        await asyncio.sleep(0.0005)
        async with queues_lock:
            active = set(request_queues.keys())
        if not talker_ready:
            continue
        # Optionally wait for more prefills
        if len(talker_ready) < len(active):
            t_start = time.perf_counter()
            while (time.perf_counter() - t_start) * 1000 < PREFILL_COLLECT_MS:
                await asyncio.sleep(0.001)
                async with queues_lock:
                    active = set(request_queues.keys())
                if talker_ready >= active:
                    break
        if not talker_ready:
            continue
        # Run step with whoever is ready; do not wait for all active, so chunk-2
        # / time-to-first-audio stays low when many requests are concurrent.
        try:
            logger.info(
                f"[talker_loop_mp] run_step active={len(active)} talker_ready={len(talker_ready)} "
                f"request_ids={list(talker_ready)[:3]!r}"
            )
            future = talker_client.run_step_async()
            _, outputs_all = await future
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception(f"[talker_loop_mp] step failed: {e}")
            continue

        if not outputs_all:
            continue

        step_count += 1
        completed_this_step = set()
        for item in outputs_all:
            request_id, seq_id, token_ids, hidden_states, is_finished = item
            completed_this_step.add(request_id)
            payload = {"token_ids": token_ids, "hidden_states": hidden_states}
            async with queues_lock:
                q = request_queues.get(request_id)
            if q is not None:
                try:
                    q.put_nowait(("talker", "token", payload))
                    if step_count <= 1:
                        logger.info(f"[talker_loop_mp] dispatched token to request_id={request_id[:8]} token_ids={token_ids[:5]!r}")
                except asyncio.QueueFull:
                    pass
            if is_finished:
                async with queues_lock:
                    q = request_queues.get(request_id)
                if q is not None:
                    try:
                        q.put_nowait(("talker", "done", {}))
                    except asyncio.QueueFull:
                        pass

        # Clear only the request_ids that were in this step (they've been served)
        talker_ready -= completed_this_step
        if step_count % 50 == 1:
            logger.info(f"[talker_loop_mp] step#{step_count} batch={len(outputs_all)}")


async def run_predictor_loop_mp(
    predictor_client: Any,
    request_queues: dict,
    queues_lock: asyncio.Lock,
    predictor_ready: set,
) -> None:
    """
    Replacement for run_predictor_loop when using multiprocess predictor.
    Waits until predictor_ready is non-empty (and optionally all active have sent),
    sends run_step, awaits burst result, dispatches to request_queues.
    """
    burst_count = 0
    while True:
        await asyncio.sleep(0.0005)
        if not predictor_ready:
            continue
        # Brief yield to let more add_requests arrive (batching)
        async with queues_lock:
            active = set(request_queues.keys())
        if len(predictor_ready) < len(active) and len(active) > 1:
            await asyncio.sleep(PREDICTOR_COLLECT_MS / 1000.0)
            async with queues_lock:
                active = set(request_queues.keys())
                predictor_ready_copy = set(predictor_ready)
        else:
            async with queues_lock:
                active = set(request_queues.keys())
                predictor_ready_copy = set(predictor_ready)

        if not predictor_ready_copy:
            continue

        try:
            future = predictor_client.run_step_async()
            _, outputs_all = await future
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception(f"[predictor_loop_mp] burst failed: {e}")
            continue

        burst_count += 1
        for request_id, seq_id, token_ids in outputs_all:
            payload = {"token_ids": token_ids}
            async with queues_lock:
                q = request_queues.get(request_id)
            if q is not None:
                try:
                    q.put_nowait(("predictor", "token", payload))
                except asyncio.QueueFull:
                    pass
            predictor_ready.discard(request_id)

        if burst_count % 50 == 1:
            logger.info(f"[predictor_loop_mp] burst#{burst_count} finished={len(outputs_all)}")
