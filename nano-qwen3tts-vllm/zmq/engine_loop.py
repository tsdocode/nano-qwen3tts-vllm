"""
Async engine step loops: separate loops for talker and predictor so they
run independently as asyncio tasks on the event-loop thread.
No threads — step() runs on the same thread as the asyncio event loop
so CUDA graphs are not hurt.
"""

import asyncio
from typing import Any


async def run_talker_loop(
    talker_llm: Any,
    zmq_bridge: Any,
) -> None:
    """
    Run forever as an asyncio task.  When the talker has pending work,
    call step_with_outputs() and publish tokens / done to ZMQ.
    Yields to the event loop when idle via asyncio.sleep(0.001).
    """
    while True:
        has_work = bool(talker_llm.scheduler.waiting or talker_llm.scheduler.running)
        if has_work:
            try:
                _, _, outputs_all = talker_llm.step_with_outputs()
                for tup in outputs_all:
                    request_id, seq_id, token_ids, hidden_states, is_finished = tup
                    zmq_bridge.publish_token("talker", request_id, token_ids, hidden_states)
                    if is_finished:
                        zmq_bridge.publish_done("talker", request_id)
            except Exception as e:
                raise e
            await asyncio.sleep(0)
        else:
            await asyncio.sleep(0.001)


async def run_predictor_loop(
    predictor_llm: Any,
    zmq_bridge: Any,
) -> None:
    """
    Run forever as an asyncio task.  When the predictor has pending work,
    call step() and publish tokens to ZMQ.
    Yields to the event loop when idle via asyncio.sleep(0.001).
    """
    while True:
        has_work = bool(predictor_llm.scheduler.waiting or predictor_llm.scheduler.running)
        if has_work:
            try:
                outputs, _ = predictor_llm.step()
                for request_id, seq_id, token_ids in outputs:
                    zmq_bridge.publish_token("predictor", request_id, token_ids, None)
            except Exception as e:
                raise e
            await asyncio.sleep(0)
        else:
            await asyncio.sleep(0.001)
