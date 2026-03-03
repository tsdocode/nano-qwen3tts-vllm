"""
Load only embedding layers for the main process when using ZMQ/async mode.
Avoids loading full talker and predictor models in the main process to prevent
CUDA OOM (main process + two engine processes would load models 3x on GPU).
"""

import os
import json
import torch
from torch import nn

from nano_qwen3tts_vllm.config import Qwen3TTSConfig
from nano_qwen3tts_vllm.models.qwen3_tts_talker import Qwen3TTSTalkerResizeMLP
from safetensors.torch import load_file


def load_embeddings_only(model_path: str, device: str = "cpu"):
    """
    Load config and only the embedding layers needed for request preparation
    in the main process (text_embedding, input_embedding, text_projection from
    talker; codec_embedding from predictor). Keeps main process GPU memory minimal.

    Returns:
        model_config: Qwen3TTSConfig (full config for prepare_inputs etc.)
        text_embedding: nn.Embedding (talker)
        input_embedding: nn.Embedding (talker codec)
        text_projection: Qwen3TTSTalkerResizeMLP (talker)
        predictor_input_embeddings: nn.ModuleList of nn.Embedding (predictor codec)
    """
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    model_config = Qwen3TTSConfig(**config_dict)
    talker_config = model_config.talker_config
    predictor_config = talker_config.code_predictor_config

    state_dict = load_file(os.path.join(model_path, "model.safetensors"))

    # --- Talker embedding modules (same structure as full model for load_state_dict) ---
    text_embedding = nn.Embedding(
        talker_config.text_vocab_size, talker_config.text_hidden_size
    )
    input_embedding = nn.Embedding(talker_config.vocab_size, talker_config.hidden_size)
    text_projection = Qwen3TTSTalkerResizeMLP(
        talker_config.text_hidden_size,
        talker_config.text_hidden_size,
        talker_config.hidden_size,
        talker_config.hidden_act,
        bias=True,
    )
    # Minimal talker state: keys like model.text_embedding.weight, model.codec_embedding.weight, text_projection.*
    talker_sd = {}
    for k, v in state_dict.items():
        if not k.startswith("talker.") or k.startswith("talker.code_predictor."):
            continue
        key = k.replace("talker.", "", 1)
        talker_sd[key] = v
    # HF checkpoints may use embed_tokens; map to codec_embedding (same as talker convert_state_dict)
    if "model.embed_tokens.weight" in talker_sd and "model.codec_embedding.weight" not in talker_sd:
        talker_sd["model.codec_embedding.weight"] = talker_sd["model.embed_tokens.weight"]

    # Build a minimal module that has model.text_embedding, model.codec_embedding, text_projection
    class _TalkerEmbeddings(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.text_embedding = text_embedding
            self.model.codec_embedding = input_embedding
            self.text_projection = text_projection

    talker_emb = _TalkerEmbeddings()
    talker_emb.load_state_dict(talker_sd, strict=False)

    # --- Predictor codec_embedding (ModuleList of num_code_groups - 1 embeddings) ---
    n_code = predictor_config.num_code_groups - 1
    predictor_codec = nn.ModuleList(
        [
            nn.Embedding(predictor_config.vocab_size, talker_config.hidden_size)
            for _ in range(n_code)
        ]
    )
    predictor_sd = {}
    for k, v in state_dict.items():
        if not k.startswith("talker.code_predictor."):
            continue
        key = k.replace("talker.code_predictor.", "", 1)
        predictor_sd[key] = v

    class _PredictorCodecOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.codec_embedding = predictor_codec

    pred_emb = _PredictorCodecOnly()
    pred_emb.load_state_dict(predictor_sd, strict=False)

    # Move to device
    d = torch.device(device)
    model_config = model_config
    text_embedding = talker_emb.model.text_embedding.to(d)
    input_embedding = talker_emb.model.codec_embedding.to(d)
    text_projection = talker_emb.text_projection.to(d)
    predictor_input_embeddings = pred_emb.model.codec_embedding.to(d)

    return (
        model_config,
        text_embedding,
        input_embedding,
        text_projection,
        predictor_input_embeddings,
    )
