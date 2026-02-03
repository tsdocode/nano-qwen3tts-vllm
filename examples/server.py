"""FastAPI server for Qwen3-TTS text-to-speech generation.

Env:
  USE_ZMQ=1              - Use ZMQ (async engine loop + async queue).
  QWEN3_TTS_MODEL_PATH   - Model directory or HuggingFace model ID (e.g., Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice).
  HOST, PORT             - Server bind address.
"""

import asyncio
import logging
import os
import time
import threading
from contextlib import asynccontextmanager
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Output format: 16-bit PCM at 24 kHz
TARGET_SAMPLE_RATE = 24000

logger = logging.getLogger(__name__)

# Ensure log messages appear on console (works when run as uvicorn server:app or python server.py)
if not logging.getLogger().handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(_handler)
    logging.getLogger().setLevel(logging.DEBUG if os.environ.get("DEBUG_TTS") else logging.INFO)

# Lazy imports to avoid loading heavy models at module load
_interface = None
_tokenizer = None
_zmq_bridge = None
_decode_lock = threading.Lock()


def _use_zmq():
    """True if server should use ZMQ (background engine loop + queue-based generate)."""
    return os.environ.get("USE_ZMQ", "1").lower() in ("1", "true", "yes")


def get_interface():
    """Get or initialize the Qwen3TTSInterface (with or without ZMQ based on USE_ZMQ env)."""
    global _interface, _zmq_bridge
    if _interface is None:
        from nano_qwen3tts_vllm.interface import Qwen3TTSInterface
        model_path = os.environ.get("QWEN3_TTS_MODEL_PATH", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        
        # Check if it's a local path or HuggingFace model ID
        import os as os_module
        if os_module.isdir(model_path) or os_module.isfile(model_path):
            # Local path - use regular init
            if _use_zmq():
                from nano_qwen3tts_vllm.zmq import ZMQOutputBridge
                import warnings
                # Auto-find port if default is in use
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    _zmq_bridge = ZMQOutputBridge(auto_find_port=True)
                    if w:
                        for warning in w:
                            logger.warning(str(warning.message))
                _interface = Qwen3TTSInterface(
                    model_path=model_path,
                    zmq_bridge=_zmq_bridge,
                    enforce_eager=False,
                )
            else:
                _interface = Qwen3TTSInterface(model_path=model_path)
        else:
            # HuggingFace model ID - use from_pretrained
            if _use_zmq():
                from nano_qwen3tts_vllm.zmq import ZMQOutputBridge
                import warnings
                # Auto-find port if default is in use
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    _zmq_bridge = ZMQOutputBridge(auto_find_port=True)
                    if w:
                        for warning in w:
                            logger.warning(str(warning.message))
                _interface = Qwen3TTSInterface.from_pretrained(
                    pretrained_model_name_or_path=model_path,
                    zmq_bridge=_zmq_bridge,
                    enforce_eager=False,
                )
            else:
                _interface = Qwen3TTSInterface.from_pretrained(
                    pretrained_model_name_or_path=model_path,
                    enforce_eager=False,
                )
    return _interface


def get_tokenizer():
    """Get or initialize the Qwen3TTSTokenizer for decoding audio codes."""
    global _tokenizer
    if _tokenizer is None:
        from nano_qwen3tts_vllm.utils.speech_tokenizer_cudagraph import SpeechTokenizerCUDAGraph

        _tokenizer = SpeechTokenizerCUDAGraph(
            "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            device="cuda:0",
        )
        
    return _tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: warm up model and start ZMQ tasks when USE_ZMQ. Shutdown: stop ZMQ tasks and close bridge."""
    interface = get_interface()
    get_tokenizer()
    if _use_zmq() and interface.zmq_bridge is not None:
        await interface.start_zmq_tasks()
        
    generate_speech_stream(SpeechRequest(text="Hello, this is a test.", language="English", speaker="Vivian"))
    yield
    if _use_zmq() and _interface is not None and _interface.zmq_bridge is not None:
        await _interface.stop_zmq_tasks()
        if _zmq_bridge is not None:
            _zmq_bridge.close()


app = FastAPI(
    title="Qwen3-TTS API",
    description="Text-to-speech generation using Qwen3-TTS with vLLM-style optimizations",
    version="0.1.0",
    lifespan=lifespan,
)


class SpeechRequest(BaseModel):
    """Request body for speech generation."""

    text: str = Field(..., min_length=1, description="Text to synthesize")
    language: str = Field(default="English", description="Language of the text")
    speaker: str = Field(default="Vivian", description="Speaker name")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


def _float_to_pcm16(wav: np.ndarray) -> np.ndarray:
    """Convert float32 [-1, 1] to int16 PCM."""
    wav = np.clip(wav, -1.0, 1.0)
    return (wav * 32767).astype(np.int16)


def _resample_to_24k(wav: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample waveform to 24 kHz if needed."""
    if orig_sr == TARGET_SAMPLE_RATE:
        return wav
    n_orig = len(wav)
    n_new = int(round(n_orig * TARGET_SAMPLE_RATE / orig_sr))
    if n_new == 0:
        return wav
    indices = np.linspace(0, n_orig - 1, n_new, dtype=np.float64)
    return np.interp(indices, np.arange(n_orig), wav).astype(np.float32)


def _decode_batch(tokenizer, audio_codes: list) -> tuple[np.ndarray, int]:
    """Decode cumulative audio_codes to PCM16 @ 24kHz. Returns (pcm16, sample_rate)."""
    with _decode_lock:
        wav_list, sr = tokenizer.decode([{"audio_codes": audio_codes}])
    wav = wav_list[0]
    wav_24k = _resample_to_24k(wav, sr)
    return _float_to_pcm16(wav_24k), TARGET_SAMPLE_RATE


async def generate_speech_stream(request: SpeechRequest):
    """
    Streaming decode: producer (generation) and consumer (decode) run concurrently.
    Uses single asyncio.Queue + run_in_executor — no decode thread, no call_soon_threadsafe.
    When consumer awaits decode in executor, event loop runs producer (overlap).
    """
    interface = get_interface()
    tokenizer = get_tokenizer()
    loop = asyncio.get_event_loop()
    codes_queue: asyncio.Queue[list | None] = asyncio.Queue(maxsize=2)  # backpressure
    async def producer() -> None:
        audio_codes = []
        first_chunk_time = None
        last_chunk_time = None
        try:
            async for audio_code in interface.generate_custom_voice_async(
                text=request.text,
                language=request.language,
                speaker=request.speaker,
            ):
                current_time = time.time()
                if first_chunk_time is None:
                    first_chunk_time = current_time
                if last_chunk_time is not None:
                    inner_latency = current_time - last_chunk_time
                    print(f"[producer] inner chunk latency: {inner_latency*1000:.2f}ms")
                last_chunk_time = current_time
                
                audio_codes.append(audio_code)
                if len(audio_codes) % 4 == 0:  # decode every 4 chunks
                    await codes_queue.put(list(audio_codes))
            
            if first_chunk_time is not None:
                first_chunk_latency = last_chunk_time - first_chunk_time
                print(f"[producer] first chunk latency: {first_chunk_latency*1000:.2f}ms")
            
            # final batch if not already sent (e.g. 13 chunks: sent at 12, need 13)
            if audio_codes and len(audio_codes) % 4 != 0:
                await codes_queue.put(list(audio_codes))
        finally:
            await codes_queue.put(None)  # sentinel

    producer_task = asyncio.create_task(producer())
    prev_len_24k = 0

    try:
        while True:
            item = await codes_queue.get()
            if item is None:
                break
            # run_in_executor: decode in thread pool; event loop runs producer meanwhile
            pcm16, _ = await loop.run_in_executor(
                None,
                lambda c=item: _decode_batch(tokenizer, c),
            )
            chunk = pcm16[prev_len_24k:].tobytes()
            prev_len_24k = len(pcm16)
            if chunk:
                yield chunk
    finally:
        await producer_task


@app.post("/v1/audio/speech", response_class=StreamingResponse)
async def generate_speech(request: SpeechRequest):
    """
    Generate speech from text.
    Returns raw PCM 16-bit mono at 24 kHz (audio/L16).
    Uses generate_custom_voice_async (requires USE_ZMQ=1).
    """
    try:
        return StreamingResponse(
            generate_speech_stream(request),
            media_type="audio/L16",
            headers={"Sample-Rate": str(TARGET_SAMPLE_RATE)},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """API info."""
    return {
        "name": "Qwen3-TTS API",
        "docs": "/docs",
        "health": "/health",
        "speech": "POST /v1/audio/speech (PCM16, 24 kHz mono)",
        "zmq": _use_zmq(),
    }


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    if _use_zmq():
        logger.info("Starting Qwen3-TTS API with ZMQ (async engine loop).")
    uvicorn.run(app, host=host, port=port)
