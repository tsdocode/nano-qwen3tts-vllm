"""
Shared command and result protocol for main ↔ worker (talker / predictor).
Uses pickle + numpy for tensors so we avoid sending CUDA tensors across processes.
"""

import pickle
from typing import Any, Optional

import numpy as np
import torch


# Command types
CMD_ADD_REQUEST = "add_request"
CMD_RUN_STEP = "run_step"
CMD_CLEAR_REQUEST = "clear_request"
CMD_SHUTDOWN = "shutdown"


def _tensor_to_numpy(x: Any) -> Any:
    """Convert torch tensors to numpy for serialization. BFloat16 -> float32 (numpy has no bfloat16)."""
    if hasattr(x, "cpu"):
        t = x.detach().cpu()
        if t.dtype == torch.bfloat16:
            t = t.float()
        return t.numpy()
    if isinstance(x, np.ndarray):
        return x
    return x


def _numpy_to_tensor(x: Any, device: str = "cuda") -> Any:
    """Convert numpy back to torch (called in worker)."""
    if isinstance(x, np.ndarray):
        import torch
        return torch.from_numpy(x).to(device)
    return x


# ---- Talker ----

def serialize_talker_add_request(
    request_id: str,
    inputs_embeds: Any,  # list of tensors
    sampling_params: dict,
) -> bytes:
    """Serialize add_request for talker. inputs_embeds: list of [B, T, D] tensors."""
    embeds_list = [_tensor_to_numpy(t) for t in inputs_embeds]
    obj = {
        "cmd": CMD_ADD_REQUEST,
        "request_id": request_id,
        "inputs_embeds": embeds_list,
        "sampling_params": sampling_params,
    }
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def serialize_talker_run_step(step_id: str) -> bytes:
    """Serialize run_step command."""
    return pickle.dumps({"cmd": CMD_RUN_STEP, "step_id": step_id}, protocol=pickle.HIGHEST_PROTOCOL)


def serialize_clear_request(request_id: str) -> bytes:
    """Serialize clear_request command (same for talker and predictor)."""
    return pickle.dumps({"cmd": CMD_CLEAR_REQUEST, "request_id": request_id}, protocol=pickle.HIGHEST_PROTOCOL)


def serialize_shutdown() -> bytes:
    """Serialize shutdown command."""
    return pickle.dumps({"cmd": CMD_SHUTDOWN}, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_command(payload: bytes) -> dict:
    """Deserialize any command from main."""
    return pickle.loads(payload)


def serialize_talker_result(
    step_id: str,
    outputs_all: list[tuple[str, Any, list[int], Any, bool]],
) -> bytes:
    """
    Serialize talker step result. outputs_all: list of
    (request_id, seq_id, token_ids, last_hidden_state, is_finished).
    last_hidden_state: tensor -> numpy.
    """
    out = []
    for request_id, seq_id, token_ids, last_hidden_state, is_finished in outputs_all:
        h = _tensor_to_numpy(last_hidden_state) if last_hidden_state is not None else None
        out.append((request_id, seq_id, token_ids, h, is_finished))
    return pickle.dumps({"step_id": step_id, "outputs_all": out}, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_talker_result(payload: bytes) -> tuple[str, list]:
    """Returns (step_id, outputs_all). hidden_states stay numpy for main (main converts if needed)."""
    obj = pickle.loads(payload)
    return obj["step_id"], obj["outputs_all"]


# ---- Predictor ----

def serialize_predictor_add_request(
    request_id: str,
    inputs_embeds: Any,
    sampling_params: dict,
) -> bytes:
    """Serialize add_request for predictor."""
    embeds_list = [_tensor_to_numpy(t) for t in inputs_embeds]
    obj = {
        "cmd": CMD_ADD_REQUEST,
        "request_id": request_id,
        "inputs_embeds": embeds_list,
        "sampling_params": sampling_params,
    }
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def serialize_predictor_run_step(step_id: str) -> bytes:
    """Serialize run_step for predictor (worker will run burst of step() until no work)."""
    return pickle.dumps({"cmd": CMD_RUN_STEP, "step_id": step_id}, protocol=pickle.HIGHEST_PROTOCOL)


def serialize_predictor_result(
    step_id: str,
    outputs_all: list[tuple[str, Any, list[int]]],
) -> bytes:
    """
    Serialize predictor burst result. outputs_all: list of (request_id, seq_id, token_ids).
    """
    return pickle.dumps({"step_id": step_id, "outputs_all": outputs_all}, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_predictor_result(payload: bytes) -> tuple[str, list]:
    """Returns (step_id, outputs_all)."""
    obj = pickle.loads(payload)
    return obj["step_id"], obj["outputs_all"]
