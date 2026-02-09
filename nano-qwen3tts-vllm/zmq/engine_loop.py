"""
Async engine step loops: separate loops for talker and predictor so they
run independently as asyncio tasks on the event-loop thread.

GPU step() calls run directly (not via run_in_executor) because:
  - CUDA graph replay takes ~1ms -- run_in_executor adds 1-2ms overhead per call
  - With 15 predictor steps per frame, that's 15-30ms of wasted executor overhead

Key batching strategy:
  - The TALKER waits for all active requests to have decode_input_embeds
    before running, which forces requests into sync.
  - The PREDICTOR yields briefly before each burst to let all interface tasks
    submit their work, improving batching from batch=N-1+1 to batch=N.
  - Once synced, requests stay in lockstep permanently.
"""

import asyncio
import time
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Safety timeout to prevent infinite wait if a client disconnects.
# Normal wait is <20ms (one predictor burst cycle). This is just a safety net.
TALKER_BATCH_WAIT_SAFETY_MS = 20.0  # milliseconds

# Brief sleep before predictor burst to let interface tasks submit work.
# With N CCUs, N interface tasks compete for the event loop after talker
# publishes N outputs.  Each takes ~0.5ms to process + submit predictor.
# 3ms gives enough time for 6+ tasks to submit, dramatically improving batching.
PREDICTOR_COLLECT_MS = 3.0  # milliseconds

# Max time to collect prefill requests before starting the prefill step.
# Without this, step_with_outputs() blocks the event loop for ~47ms during
# each prefill, preventing new HTTP requests and _do_prep from completing.
# With 16 CCUs this causes 5 prefill waves (5 × 47ms = 235ms stagger).
# Collecting first lets all _do_preps finish (~2ms each) so we do ONE big prefill.
PREFILL_COLLECT_MS = 15.0  # milliseconds


def _count_talker_ready(talker_llm) -> int:
    """Count sequences in the talker's running queue that have decode_input_embeds set."""
    count = 0
    for seq in talker_llm.scheduler.running:
        if len(seq) == 0 or seq.decode_input_embeds is not None:
            count += 1
    return count


async def run_talker_loop(
    talker_llm: Any,
    zmq_bridge: Any,
) -> None:
    """
    Run forever as an asyncio task. When the talker has schedulable work,
    call step_with_outputs() and publish tokens / done to ZMQ.

    WAITS for ALL active requests to have decode_input_embeds before running.
    This forces requests into sync: once synced, they stay in lockstep permanently.
    The wait is bounded by one predictor burst cycle (~17ms) and only happens
    during initial sync. After that, all requests arrive within <2ms.
    """
    step_count = 0
    while True:
        total_active = len(talker_llm.scheduler.running)
        ready = _count_talker_ready(talker_llm)
        has_prefill = bool(talker_llm.scheduler.waiting)

        if ready > 0 or has_prefill:
            # ── Prefill collection: yield event loop so pending _do_prep calls
            # can finish and their requests enter scheduler.waiting. Each _do_prep
            # takes ~2ms; without this yield, step_with_outputs() blocks the loop
            # for ~47ms per prefill wave, causing 16 CCU requests to prefill in
            # 5 waves instead of 1 (adds ~200ms stagger).
            if has_prefill:
                t_collect = time.perf_counter()
                prev_waiting = len(talker_llm.scheduler.waiting)
                # Yield at least once (1ms) to let event loop process pending tasks
                await asyncio.sleep(0.001)
                for _ in range(int(PREFILL_COLLECT_MS)):  # up to PREFILL_COLLECT_MS rounds of 1ms
                    curr_waiting = len(talker_llm.scheduler.waiting)
                    if curr_waiting == prev_waiting:
                        break  # No new arrivals in last 1ms → done collecting
                    prev_waiting = curr_waiting
                    await asyncio.sleep(0.001)  # 1ms yield
                final_waiting = len(talker_llm.scheduler.waiting)
                collect_ms = (time.perf_counter() - t_collect) * 1000
                logger.info(
                    f"[talker_loop] prefill_collect: {final_waiting} requests "
                    f"in {collect_ms:.1f}ms"
                )
                # Re-read state after collection
                total_active = len(talker_llm.scheduler.running)
                ready = _count_talker_ready(talker_llm)
                has_prefill = bool(talker_llm.scheduler.waiting)

            # Wait for ALL active sequences to become ready (decode_input_embeds set).
            # This is the key synchronization point: it forces all requests into
            # the same batch, after which they stay in lockstep permanently.
            # Skip waiting if there's a prefill to do (prefills don't need decode_input_embeds).
            if total_active > 1 and total_active > ready and not has_prefill:
                t_wait_start = time.perf_counter()
                while True:
                    await asyncio.sleep(0.0005)  # 0.5ms poll; yields to predictor + interface tasks
                    ready = _count_talker_ready(talker_llm)
                    total_active = len(talker_llm.scheduler.running)
                    has_prefill = bool(talker_llm.scheduler.waiting)
                    if ready >= total_active or has_prefill:
                        break  # All ready or prefill arrived
                    if total_active <= 1:
                        break  # Only 1 request left, no need to batch
                    elapsed = (time.perf_counter() - t_wait_start) * 1000
                    if elapsed >= TALKER_BATCH_WAIT_SAFETY_MS:
                        logger.warning(
                            f"[talker_loop] batch_wait SAFETY timeout: ready={ready}/{total_active} "
                            f"after {elapsed:.1f}ms -- possible client disconnect"
                        )
                        break

            try:
                t0 = time.perf_counter()
                _, _, outputs_all = talker_llm.step_with_outputs()
                t_step = time.perf_counter()

                batch_size = len(outputs_all)
                if batch_size == 0:
                    # schedule() returned nothing (all seqs waiting for decode_input_embeds)
                    await asyncio.sleep(0.0005)  # 0.5ms before retry
                    continue

                step_count += 1

                t_pub = time.perf_counter()
                for tup in outputs_all:
                    request_id, seq_id, token_ids, hidden_states, is_finished = tup
                    zmq_bridge.publish_token("talker", request_id, token_ids, hidden_states)
                    if is_finished:
                        zmq_bridge.publish_done("talker", request_id)
                t_done = time.perf_counter()

                if step_count % 50 == 1 or batch_size > 1:
                    logger.info(
                        f"[talker_loop] step#{step_count} "
                        f"batch={batch_size} active={total_active} "
                        f"gpu={((t_step - t0) * 1000):.2f}ms "
                        f"publish={((t_done - t_pub) * 1000):.2f}ms "
                        f"total={((t_done - t0) * 1000):.2f}ms"
                    )
            except Exception as e:
                raise e
            await asyncio.sleep(0)
        else:
            await asyncio.sleep(0)


async def run_predictor_loop(
    predictor_llm: Any,
    zmq_bridge: Any,
    talker_llm: Any = None,
) -> None:
    """
    Run forever as an asyncio task. When the predictor has pending work,
    run a TIGHT decode loop -- all decode steps execute without yielding
    (since decode steps don't need external input).

    Before starting each burst, yields briefly (PREDICTOR_COLLECT_MS) to let
    interface tasks submit their work. This dramatically improves batching:
    without the yield, the predictor grabs 7/8 requests immediately and runs
    the 8th in a separate burst (batch=7+1 pattern). With the yield, all 8
    interface tasks get event loop time to submit, and the predictor runs
    all 8 in one burst (batch=8).
    """
    step_count = 0
    while True:
        pred_waiting = len(predictor_llm.scheduler.waiting)
        pred_running = len(predictor_llm.scheduler.running)
        has_work = pred_waiting > 0 or pred_running > 0

        if has_work:
            # Brief yield before starting burst: let interface tasks that are
            # ready to submit predictor work do so.  This closes the batch=N-1+1
            # gap by giving ALL interface tasks event loop time.
            # Only sleep when there are multiple active talker sequences
            # (i.e., we expect more submissions to arrive).
            n_talker_active = len(talker_llm.scheduler.running) if talker_llm else 0
            if n_talker_active > 1 and pred_waiting + pred_running < n_talker_active:
                await asyncio.sleep(PREDICTOR_COLLECT_MS / 1000.0)
                # Re-check after sleep
                pred_waiting = len(predictor_llm.scheduler.waiting)
                pred_running = len(predictor_llm.scheduler.running)
                has_work = pred_waiting > 0 or pred_running > 0
                if not has_work:
                    await asyncio.sleep(0)
                    continue

            try:
                t_loop_start = time.perf_counter()
                steps_in_burst = 0
                finished_outputs = []

                # TIGHT LOOP: run all steps until no more work, without yielding
                while True:
                    if not (predictor_llm.scheduler.waiting or predictor_llm.scheduler.running):
                        break

                    outputs, _ = predictor_llm.step()
                    steps_in_burst += 1

                    for request_id, seq_id, token_ids in outputs:
                        finished_outputs.append((request_id, seq_id, token_ids))

                    if not predictor_llm.scheduler.running and not predictor_llm.scheduler.waiting:
                        break

                t_loop_end = time.perf_counter()

                # Publish finished outputs
                t_pub = time.perf_counter()
                for request_id, seq_id, token_ids in finished_outputs:
                    zmq_bridge.publish_token("predictor", request_id, token_ids, None)
                t_done = time.perf_counter()

                step_count += 1
                batch_size = max(pred_waiting, pred_running)
                logger.info(
                    f"[predictor_loop] burst#{step_count} "
                    f"steps={steps_in_burst} batch={batch_size} "
                    f"finished={len(finished_outputs)} "
                    f"gpu_total={((t_loop_end - t_loop_start) * 1000):.2f}ms "
                    f"publish={((t_done - t_pub) * 1000):.2f}ms"
                )
            except Exception as e:
                raise e
            await asyncio.sleep(0)
        else:
            await asyncio.sleep(0)
