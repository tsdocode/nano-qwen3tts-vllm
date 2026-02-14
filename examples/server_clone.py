#!/usr/bin/env python3
"""
High-Performance Async Voice Clone Server
- True concurrency
- Continuous batching (ZMQ engine loop)
- Single model instance
- No duplicate GPU load
- Stable streaming
"""

import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nano_qwen3tts_vllm.interface import Qwen3TTSInterface
from nano_qwen3tts_vllm.zmq import ZMQOutputBridge

# ---------------- CONFIG ---------------- #

TARGET_SAMPLE_RATE = 24000
MODEL_PATH = os.environ.get(
    "QWEN3_TTS_MODEL_PATH",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clone-async-server")

# ---------------- GLOBALS ---------------- #

interface: Qwen3TTSInterface = None
decode_lock = threading.Lock()

# ---------------- FASTAPI ---------------- #

app = FastAPI(title="Qwen3 Clone Async Server", version="2.0")


class CloneRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: Optional[str] = "English"
    ref_audio: str
    ref_text: Optional[str] = None
    x_vector_only_mode: bool = False


# ---------------- LIFESPAN ---------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):

    global interface

    logger.info("Initializing ZMQ bridge...")
    zmq_bridge = ZMQOutputBridge(auto_find_port=True)

    logger.info("Loading model (single instance, ZMQ mode)...")

    interface = Qwen3TTSInterface.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH,
        enforce_eager=False,
        tensor_parallel_size=1,
        zmq_bridge=zmq_bridge,
    )

    await interface.start_zmq_tasks()

    logger.info("Model + ZMQ engine ready.")
    yield

    logger.info("Shutting down...")
    await interface.stop_zmq_tasks()
    interface.shutdown()


app.router.lifespan_context = lifespan


# ---------------- AUDIO UTILS ---------------- #

def float_to_pcm16(wav: np.ndarray) -> np.ndarray:
    wav = np.clip(wav, -1.0, 1.0)
    return (wav * 32767).astype(np.int16)


def resample_to_24k(wav: np.ndarray, sr: int):
    if sr == TARGET_SAMPLE_RATE:
        return wav
    n_new = int(len(wav) * TARGET_SAMPLE_RATE / sr)
    return np.interp(
        np.linspace(0, len(wav) - 1, n_new),
        np.arange(len(wav)),
        wav,
    ).astype(np.float32)


# ---------------- STREAM GENERATOR ---------------- #

async def generate_stream(req: CloneRequest):

    tokenizer = interface.speech_tokenizer
    prev_len = 0
    accumulated_codes = []

    try:
        async for chunk in interface.generate_voice_clone_async(
            text=req.text,
            language=req.language,
            ref_audio=req.ref_audio,
            ref_text=req.ref_text,
            x_vector_only_mode=req.x_vector_only_mode,
        ):

            accumulated_codes.append(chunk)

            # Continuous decode every 4 chunks
            if len(accumulated_codes) % 4 == 0:

                def decode_batch():
                    with decode_lock:
                        wavs, sr = tokenizer.decode(
                            [{"audio_codes": accumulated_codes}]
                        )
                    wav = resample_to_24k(wavs[0], sr)
                    return float_to_pcm16(wav)

                pcm16 = await asyncio.get_event_loop().run_in_executor(
                    None, decode_batch
                )

                out = pcm16[prev_len:].tobytes()
                prev_len = len(pcm16)

                if out:
                    yield out

        # Final decode if remaining
        if accumulated_codes:

            def decode_final():
                with decode_lock:
                    wavs, sr = tokenizer.decode(
                        [{"audio_codes": accumulated_codes}]
                    )
                wav = resample_to_24k(wavs[0], sr)
                return float_to_pcm16(wav)

            pcm16 = await asyncio.get_event_loop().run_in_executor(
                None, decode_final
            )

            out = pcm16[prev_len:].tobytes()
            if out:
                yield out

    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------- ENDPOINT ---------------- #

@app.post("/v1/audio/clone")
async def clone_endpoint(req: CloneRequest):
    return StreamingResponse(
        generate_stream(req),
        media_type="audio/L16",
        headers={"Sample-Rate": str(TARGET_SAMPLE_RATE)},
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine": "ZMQ async batching",
        "model": MODEL_PATH,
    }


# ---------------- RUN ---------------- #

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
