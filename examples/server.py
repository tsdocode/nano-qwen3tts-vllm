"""FastAPI server for Qwen3-TTS text-to-speech generation.

Env:
  USE_ZMQ=1              - Use ZMQ (async engine loop + async queue).
  QWEN3_TTS_MODEL_PATH   - Model directory or HuggingFace model ID (e.g., Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice).
  HOST, PORT             - Server bind address.
"""

import asyncio
import logging
import os
import pickle
import time
import threading
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import functools
# Output format: 16-bit PCM at 24 kHz
TARGET_SAMPLE_RATE = 24000

logger = logging.getLogger(__name__)

# Ensure log messages appear on console (works when run as uvicorn server:app or python server.py)
if not logging.getLogger().handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(_handler)
    logging.getLogger().setLevel(logging.DEBUG if os.environ.get("DEBUG_TTS") else logging.INFO)

# Lazy imports to avoid loading heavy models at module load
_interface = None
_tokenizer = None
_zmq_bridge = None

# --- Batched decode system (replaces _decode_lock) ---
# Instead of serializing decode with a threading.Lock, requests submit to a
# queue and a background worker batches them into fewer decode calls.
_decode_queue: asyncio.Queue = None
_decode_worker_task: asyncio.Task = None
# Fallback lock for sync warm-up path only (not used in async serving)
_decode_lock = threading.Lock()

# Default speakers for CustomVoice model
DEFAULT_SPEAKERS = [
   
]

# Voice clones directory (same as gradio app)
VOICES_DIR = Path(__file__).parent / "voices"
VOICES_DIR.mkdir(exist_ok=True)


def _use_zmq():
    """True if server should use ZMQ (background engine loop + queue-based generate)."""
    return os.environ.get("USE_ZMQ", "1").lower() in ("1", "true", "yes")


def get_interface():
    """Get or initialize the Qwen3TTSInterface (with or without ZMQ based on USE_ZMQ env)."""
    global _interface, _zmq_bridge
    if _interface is None:
        from nano_qwen3tts_vllm.interface import Qwen3TTSInterface
        model_path = os.environ.get("QWEN3_TTS_MODEL_PATH", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        
        # Check if it's a local path or HuggingFace model ID
        if os.path.isdir(model_path) or os.path.isfile(model_path):
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
    """Startup: warm up model, start ZMQ tasks and decode worker. Shutdown: stop all."""
    global _decode_queue, _decode_worker_task

    interface = get_interface()
    get_tokenizer()
    if _use_zmq() and interface.zmq_bridge is not None:
        await interface.start_zmq_tasks()

    # Start batched decode worker
    _decode_queue = asyncio.Queue()
    _decode_worker_task = asyncio.create_task(_decode_worker_loop())

    # Warmup: run concurrent requests to compile CUDA kernels for all batch sizes.
    # Without this, the first batch=8 request triggers kernel compilation (~400ms extra).
    # We ramp from 1→8 so kernels for every batch size 1..8 are cached.
    async def _warmup_one(req):
        try:
            async for _ in generate_speech_stream(req):
                pass
        except Exception as e:
            logger.warning(f"[warmup] error (non-fatal): {e}")

    warmup_text = "Hello, this is a warmup test."
    # Use voice clone if Anna exists, otherwise default speaker
    if is_voice_clone("Anna"):
        warmup_req = SpeechRequest(text=warmup_text, language="English", speaker="Anna")
    else:
        warmup_req = SpeechRequest(text=warmup_text, language="English", speaker="Vivian")

    # Batch=1 warmup (also warms _do_prep, prefill, predictor, decode)
    logger.info("[warmup] batch=1 ...")
    await _warmup_one(warmup_req)

    # Batch=8 warmup (compiles kernels for larger matmul shapes)
    logger.info("[warmup] batch=8 ...")
    await asyncio.gather(*[_warmup_one(warmup_req) for _ in range(8)])
    logger.info("[warmup] done.")

    yield

    # Stop decode worker
    if _decode_queue is not None:
        await _decode_queue.put(None)  # sentinel
    if _decode_worker_task is not None:
        await _decode_worker_task

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


@app.get("/voices")
async def list_voices():
    """
    List all available voices (default speakers + voice clones).
    Returns a dictionary with 'default' and 'clones' keys.
    """
    default_voices = DEFAULT_SPEAKERS
    voice_clones = get_voice_clones()
    
    return {
        "default": default_voices,
        "clones": voice_clones,
        "all": default_voices + voice_clones,
    }


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


def _decode_batch_sync(tokenizer, audio_codes: list) -> tuple[np.ndarray, int]:
    """Sync decode (for warm-up only). Uses _decode_lock."""
    with _decode_lock:
        wav_list, sr = tokenizer.decode([{"audio_codes": audio_codes}])
    wav = wav_list[0]
    wav_24k = _resample_to_24k(wav, sr)
    return _float_to_pcm16(wav_24k), TARGET_SAMPLE_RATE

async def _decode_worker_loop():
    """Background asyncio task: collects decode requests from _decode_queue,
    micro-batches them, and runs decode in an executor.

    Each queue item is a dict: {"audio_codes": list, "future": asyncio.Future}.
    The worker waits for the first item, then drains any additional queued items
    into a single batched decode call, delivering results to each future.
    """
    decode_start = time.time()
    tokenizer = get_tokenizer()
    loop = asyncio.get_event_loop()
    while True:
        # Wait for first request
        item = await _decode_queue.get()
        if item is None:
            break
        batch = [item]

        # Drain any queued requests (micro-batch)
        while not _decode_queue.empty():
            try:
                extra = _decode_queue.get_nowait()
                if extra is None:
                    break
                batch.append(extra)
            except asyncio.QueueEmpty:
                break

        # Batched decode: combine all into one call
        all_inputs = [{"audio_codes": req["audio_codes"]} for req in batch]

        def _do_decode(inputs=all_inputs):
            return tokenizer.decode(inputs)

        try:
            decode_start = time.time()
            wav_results, sr = await loop.run_in_executor(None, _do_decode)
            decode_latency = (time.time() - decode_start) * 1000
            logger.info(f"[decoder] batch_size={len(batch)} latency={decode_latency:.2f}ms")
            for req, wav in zip(batch, wav_results):
                wav_24k = _resample_to_24k(wav, sr)
                pcm16 = _float_to_pcm16(wav_24k)
                if not req["future"].done():
                    req["future"].set_result((pcm16, TARGET_SAMPLE_RATE))
        except Exception as e:
            for req in batch:
                if not req["future"].done():
                    req["future"].set_exception(e)
                    
        logger.info(f"[decoder] total latency: {time.time() - decode_start:.2f}s")


async def _decode_batched(audio_codes: list) -> tuple[np.ndarray, int]:
    """Submit a decode request to the batched worker and await the result."""
    future = asyncio.get_event_loop().create_future()
    await _decode_queue.put({"audio_codes": audio_codes, "future": future})
    return await future


def _decode_inline(audio_codes: list) -> tuple[np.ndarray, int]:
    """Decode directly on the calling thread (no executor).

    Used for FIRST-chunk decode to avoid the ~130ms scheduling delay caused by
    run_in_executor callback delivery competing with GPU prefills on the event loop.
    Single-code CUDA-graph decode takes ~15ms, acceptable for inline use.
    """
    tokenizer = get_tokenizer()
    with torch.inference_mode():
        wav_list, sr = tokenizer.chunked_decode([{"audio_codes": audio_codes}])
    wav_24k = _resample_to_24k(wav_list[0], sr)
    return _float_to_pcm16(wav_24k), TARGET_SAMPLE_RATE


def get_voice_clones():
    """Get list of saved voice clone names from the voices directory."""
    if not VOICES_DIR.exists():
        return []
    voice_files = list(VOICES_DIR.glob("*.pkl"))
    voice_names = [f.stem for f in voice_files]
    return sorted(voice_names)


def is_voice_clone(speaker: str) -> bool:
    """Check if speaker is a voice clone (exists in voices directory)."""
    return (VOICES_DIR / f"{speaker}.pkl").exists()


@functools.lru_cache(maxsize=128)
def load_voice_clone_prompt(speaker: str):
    """Load voice clone prompt from pickle file."""
    voice_path = VOICES_DIR / f"{speaker}.pkl"
    if not voice_path.exists():
        raise ValueError(f"Voice clone '{speaker}' not found")
    with open(voice_path, 'rb') as f:
        return pickle.load(f)




async def generate_speech_stream(request: SpeechRequest):
    """
    Streaming decode: first chunk decoded inline for minimal latency,
    subsequent chunks use producer/consumer with batched decode worker.

    Key insight: putting the first code through codes_queue → consumer adds
    ~130ms scheduling delay (consumer competes with talker/predictor for event
    loop time). By awaiting the first code directly via __anext__() and decoding
    inline, we eliminate that delay entirely.
    """
    try:
        interface = get_interface()
        tokenizer = get_tokenizer()
        loop = asyncio.get_event_loop()
        start_time = time.time()

        # --- Create the async generator (voice clone or custom voice) ---
        if is_voice_clone(request.speaker):
            voice_clone_prompt = load_voice_clone_prompt(request.speaker)
            gen = interface.generate_voice_clone_async(
                text=request.text,
                language=request.language,
                voice_clone_prompt=voice_clone_prompt,
            )
        else:
            gen = interface.generate_custom_voice_async(
                text=request.text,
                language=request.language,
                speaker=request.speaker,
            )

        # --- FIRST CHUNK: collect 2 codes, decode inline, yield immediately ---
        # Waiting for 2 codes gives a longer initial audio segment for smoother
        # playback start, at the cost of one extra decode cycle (~50ms).
        FIRST_CHUNK_CODES = 2
        first_codes = []
        async for audio_code in gen:
            first_codes.append(audio_code)
            if len(first_codes) >= FIRST_CHUNK_CODES:
                break
        if not first_codes:
            return  # generator produced nothing

        t_first_code = time.time()
        pcm16_first, _ = _decode_inline(first_codes)
        t_first_decoded = time.time()
        logger.info(
            f"[stream] first chunk codes={len(first_codes)} "
            f"latency: {(t_first_code - start_time)*1000:.1f}ms "
            f"decode: {(t_first_decoded - t_first_code)*1000:.1f}ms "
            f"total: {(t_first_decoded - start_time)*1000:.1f}ms"
        )
        prev_len_24k = len(pcm16_first)
        yield pcm16_first.tobytes()

        # --- SUBSEQUENT CHUNKS: producer task + batched decode worker ---
        codes_queue: asyncio.Queue[list | None] = asyncio.Queue()  # unbounded

        async def producer() -> None:
            audio_codes = list(first_codes)  # first codes already yielded
            last_chunk_time = t_first_code
            try:
                async for audio_code in gen:
                    current_time = time.time()
                    inner_latency = current_time - last_chunk_time
                    logger.info(f"[producer] inner chunk latency: {inner_latency*1000:.2f}ms")
                    last_chunk_time = current_time

                    audio_codes.append(audio_code)
                    if len(audio_codes) % 4 == 0:
                        await codes_queue.put(list(audio_codes))
                # final partial batch
                if audio_codes and len(audio_codes) % 4 != 0:
                    await codes_queue.put(list(audio_codes))
            finally:
                await codes_queue.put(None)  # sentinel

        producer_task = asyncio.create_task(producer())

        try:
            while True:
                item = await codes_queue.get()
                if item is None:
                    break
                if _decode_queue is not None:
                    pcm16, _ = await _decode_batched(item)
                else:
                    pcm16, _ = await loop.run_in_executor(
                        None,
                        lambda c=item: _decode_batch_sync(tokenizer, c),
                    )
                chunk = pcm16[prev_len_24k:].tobytes()
                prev_len_24k = len(pcm16)
                if chunk:
                    yield chunk
        finally:
            await producer_task
    except StopAsyncIteration:
        # Generator produced no codes at all
        pass
    except Exception as e:
        logger.error(f"[generate_speech_stream] Error: {e}")
        import traceback
        traceback.print_exc()
        raise e

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
        "voices": "GET /voices (list all available voices)",
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
