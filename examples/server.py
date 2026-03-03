"""FastAPI server for Qwen3-TTS text-to-speech generation.

Env:
  QWEN3_TTS_MODEL_PATH   - Model directory or HuggingFace model ID (e.g., Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice).
  HOST, PORT             - Server bind address.
  (Uses multiprocess engines: talker and predictor in separate processes. Multi-worker: uvicorn examples.server:app --workers N.)

  First-chunk / mixed-load stability (when external requests join running benchmark):
  PREFILL_COLLECT_MS     - Max ms to collect prefills before running one (default 15). Lower (e.g. 5) reduces
                           first-chunk latency and stall for existing streams when a single new request joins.
  PREDICTOR_COLLECT_MS   - Ms to yield before predictor burst so interface tasks can submit (default 3).
                           Slightly higher (e.g. 5) can help a new request's first chunk batch with others.
  TALKER_BATCH_WAIT_SAFETY_MS - Safety timeout for talker batch sync (default 20). Rarely needs changing.

  DECODER_MP_WORKER         - If 1 (default), window-batch decode runs in a dedicated process so the
                              event loop is not blocked and decoder runs fast (no GIL contention).
                              Set to 0 to use in-process executor instead.
"""

import asyncio
import logging
import os
import pickle
import re
import time
import threading
import uuid
import wave
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
import torch
import multiprocessing as mp
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import functools
# Output format: 16-bit PCM at 24 kHz
TARGET_SAMPLE_RATE = 24000

# Leading silence sent immediately at stream start (ms)
SILENCE_MS = int(os.environ.get("STREAM_LEADING_SILENCE_MS", "50"))
_SILENCE_PCM16: np.ndarray = None  # cached 50ms silence, int16

def _get_leading_silence_bytes() -> bytes:
    """Return 50ms (or SILENCE_MS) of silence as PCM16 bytes at TARGET_SAMPLE_RATE."""
    global _SILENCE_PCM16
    if _SILENCE_PCM16 is None:
        n_samples = int(TARGET_SAMPLE_RATE * (SILENCE_MS / 1000.0))
        _SILENCE_PCM16 = np.zeros(n_samples, dtype=np.int16)
    return _SILENCE_PCM16.tobytes()

# Streaming decode: decode only (context + new) codes per chunk, trim to new part
STREAMING_CHUNK_SIZE = int(os.environ.get("STREAMING_CHUNK_SIZE", "4"))
STREAMING_CONTEXT_SIZE = int(os.environ.get("STREAMING_CONTEXT_SIZE", "8"))
# First N audio chunks use smaller size for lower start latency; then use STREAMING_CHUNK_SIZE
FIRST_CHUNK_COUNT = int(os.environ.get("FIRST_CHUNK_COUNT", "8"))
FIRST_CHUNK_SIZE = int(os.environ.get("FIRST_CHUNK_SIZE", "4"))
# After this many codes we switch from FIRST_CHUNK_SIZE to STREAMING_CHUNK_SIZE
_first_codes_threshold = FIRST_CHUNK_COUNT * FIRST_CHUNK_SIZE  # 8*2 = 16
# Voice clone: prefix decoded sequence with last N frames from ref_code so tokenizer has enough context; we skip this prefix when yielding audio
VOICE_CLONE_CODE_PREFIX_FRAMES = int(os.environ.get("VOICE_CLONE_CODE_PREFIX_FRAMES", "16"))

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

# --- Batched decode system (replaces _decode_lock) ---
# Instead of serializing decode with a threading.Lock, requests submit to a
# queue and a background worker batches them into fewer decode calls.
_decode_queue: asyncio.Queue = None
_decode_worker_task: asyncio.Task = None
# Fallback lock for sync warm-up path only (not used in async serving)
_decode_lock = threading.Lock()

# --- Multiprocessing decoder worker (dedicated process for window decode) ---
# When DECODER_MP_WORKER=1, window-batch decode runs in a separate process so the
# main event loop is never blocked and there is no GIL contention.
_mp_decoder_request_queue: mp.Queue = None
_mp_decoder_result_queue: mp.Queue = None
_mp_decoder_process: mp.Process = None
_mp_decoder_ctx = None  # spawn context

# Default speakers for CustomVoice model
DEFAULT_SPEAKERS = [
   
]

# Voice clones directory (same as gradio app)
VOICES_DIR = Path(__file__).parent / "voices"
VOICES_DIR.mkdir(exist_ok=True)

# Debug: save full audio per generation (filename = first 6 words)
DEBUG_AUDIO_DIR = Path(__file__).parent / "debug"
DEBUG_SAVE_AUDIO = os.environ.get("DEBUG_SAVE_AUDIO", "1").lower() in ("1", "true", "yes")


def _debug_audio_filename(text: str) -> str:
    """First 6 words of sentence, sanitized for filesystem."""
    words = text.strip().split()[:6]
    name = "_".join(words) if words else "empty"
    name = re.sub(r"[^\w\-_]", "", name)
    return (name or "empty") + ".wav"


def _save_debug_audio(chunks: list[bytes], text: str) -> None:
    """Write full PCM chunks to debug/first_6_words.wav."""
    if not DEBUG_SAVE_AUDIO or not chunks:
        return
    try:
        DEBUG_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        raw = b"".join(chunks)
        path = DEBUG_AUDIO_DIR / _debug_audio_filename(text)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(TARGET_SAMPLE_RATE)
            wav.writeframes(raw)
        logger.info(f"[debug] saved audio to {path}")
    except Exception as e:
        logger.warning(f"[debug] failed to save audio: {e}")


def get_interface():
    """Get or initialize the Qwen3TTSInterface (multiprocess engines only)."""
    global _interface
    if _interface is None:
        from nano_qwen3tts_vllm.interface import Qwen3TTSInterface
        model_path = os.environ.get("QWEN3_TTS_MODEL_PATH", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        gpu_mem_util = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.9"))

        if os.path.isdir(model_path) or os.path.isfile(model_path):
            _interface = Qwen3TTSInterface(
                model_path=model_path,
                enforce_eager=False,
                gpu_memory_utilization=gpu_mem_util,
            )
        else:
            _interface = Qwen3TTSInterface.from_pretrained(
                pretrained_model_name_or_path=model_path,
                enforce_eager=False,
                gpu_memory_utilization=gpu_mem_util,
            )
    return _interface


def get_tokenizer():
    """Get or initialize the Qwen3TTSTokenizer for decoding audio codes."""
    global _tokenizer
    if _tokenizer is None:
        from nano_qwen3tts_vllm.utils.speech_tokenizer_cudagraph import SpeechTokenizerCUDAGraph

        num_graph_lengths = int(os.environ.get("DECODER_GRAPH_LENGTHS", "50"))
        _tokenizer = SpeechTokenizerCUDAGraph(
            "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            device="cuda:0",
            num_graph_lengths=num_graph_lengths,
        )
    return _tokenizer


def _decoder_worker_process(request_queue: mp.Queue, result_queue: mp.Queue) -> None:
    """Dedicated process: load tokenizer once, then get jobs from request_queue,
    run decode_window_batched, put (job_id, wav_list, sr) on result_queue.
    Top-level function for multiprocessing spawn.
    """
    from nano_qwen3tts_vllm.utils.speech_tokenizer_cudagraph import SpeechTokenizerCUDAGraph
    num_graph_lengths = int(os.environ.get("DECODER_GRAPH_LENGTHS", "50"))
    tokenizer = SpeechTokenizerCUDAGraph(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        device="cuda:0",
        num_graph_lengths=num_graph_lengths,
    )
    while True:
        job = request_queue.get()
        if job is None:
            break
        job_id = job["job_id"]
        batch_inputs = job["batch"]
        try:
            wav_list, sr = tokenizer.decode_window_batched(batch_inputs)
            result_queue.put({"job_id": job_id, "wav_list": wav_list, "sr": sr, "error": None})
        except Exception as e:
            result_queue.put({"job_id": job_id, "wav_list": None, "sr": None, "error": e})


def _samples_per_frame() -> int:
    """Number of PCM samples per codebook frame (for trimming prefix audio)."""
    return int(get_tokenizer().tokenizer.model.decoder.total_upsample)


def _decode_window_fallback(tokenizer, batch_inputs: list) -> tuple[list, int]:
    """Decode full audio_codes then trim to new part (when decode_window_batched is not available).

    Each item in batch_inputs is {"audio_codes": list, "left_context_frames": int}.
    Returns (list of trimmed float32 wavs, sample_rate).
    """
    spf = int(tokenizer.tokenizer.model.decoder.total_upsample)
    wavs = []
    sr = None
    for r in batch_inputs:
        codes = r["audio_codes"]
        left = r.get("left_context_frames") or 0
        wav_list, sr = tokenizer.decode([{"audio_codes": codes}])
        wav = np.asarray(wav_list[0], dtype=np.float32)
        if left > 0:
            skip = left * spf
            wav = wav[skip:] if skip < len(wav) else np.array([], dtype=np.float32)
        wavs.append(wav)
    return wavs, sr


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: warm up model, start ZMQ tasks and decode worker. Shutdown: stop all."""
    global _decode_queue, _decode_worker_task
    global _mp_decoder_request_queue, _mp_decoder_result_queue, _mp_decoder_process, _mp_decoder_ctx

    interface = get_interface()
    get_tokenizer()
    await interface.start_zmq_tasks()

    # Optional: start multiprocessing decoder worker (requires tokenizer.decode_window_batched; clone uses fallback in main process)
    if os.environ.get("DECODER_MP_WORKER", "0").lower() in ("1", "true", "yes"):
        _mp_decoder_ctx = mp.get_context("spawn")
        _mp_decoder_request_queue = _mp_decoder_ctx.Queue()
        _mp_decoder_result_queue = _mp_decoder_ctx.Queue()
        _mp_decoder_process = _mp_decoder_ctx.Process(
            target=_decoder_worker_process,
            args=(_mp_decoder_request_queue, _mp_decoder_result_queue),
        )
        _mp_decoder_process.start()
        logger.info("[decoder] started MP worker process for window-batch decode")
    else:
        _mp_decoder_ctx = None
        _mp_decoder_request_queue = None
        _mp_decoder_result_queue = None
        _mp_decoder_process = None

    # Start batched decode worker (asyncio task that collects from _decode_queue)
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
    await asyncio.gather(*[_warmup_one(warmup_req) for _ in range(16)])
    logger.info("[warmup] done.")

    yield

    # Stop decode worker
    if _decode_queue is not None:
        await _decode_queue.put(None)  # sentinel
    if _decode_worker_task is not None:
        await _decode_worker_task

    # Stop MP decoder worker process
    if _mp_decoder_request_queue is not None and _mp_decoder_process is not None and _mp_decoder_process.is_alive():
        _mp_decoder_request_queue.put(None)
        _mp_decoder_process.join(timeout=10.0)
        if _mp_decoder_process.is_alive():
            _mp_decoder_process.terminate()
            _mp_decoder_process.join(timeout=5.0)
        logger.info("[decoder] MP worker process stopped")

    if _interface is not None:
        await _interface.stop_zmq_tasks()


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
    volume: float = Field(default=1.0, ge=0.0, le=3.0, description="Volume gain (1.0 = normal, e.g. 1.5 = 50% louder)")


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


def _apply_volume_pcm16(pcm16: np.ndarray, volume: float) -> np.ndarray:
    """Apply volume gain to int16 PCM (in-place style); volume 1.0 = no change."""
    if volume == 1.0:
        return pcm16
    scaled = (pcm16.astype(np.float32) * volume).round()
    return np.clip(scaled, -32768, 32767).astype(np.int16)


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

    Queue item: {"audio_codes": list, "future": asyncio.Future, "left_context_frames": int | None}.
    If left_context_frames is set, decode only the window and trim (streaming); else full decode.
    """
    tokenizer = get_tokenizer()
    loop = asyncio.get_event_loop()
    while True:
        item = await _decode_queue.get()
        if item is None:
            break
        batch = [item]

        while not _decode_queue.empty():
            try:
                extra = _decode_queue.get_nowait()
                if extra is None:
                    break
                batch.append(extra)
            except asyncio.QueueEmpty:
                break

        # Split into window (streaming) and full decode requests
        batch_window = [r for r in batch if r.get("left_context_frames") is not None]
        batch_full = [r for r in batch if r.get("left_context_frames") is None]

        decode_start = time.time()
        try:
            if batch_full:
                all_inputs = [{"audio_codes": r["audio_codes"]} for r in batch_full]

                def _do_full(inputs=all_inputs):
                    return tokenizer.decode(inputs)

                t_before_exec = time.time()
                wav_results, sr = await loop.run_in_executor(None, _do_full)
                exec_s = time.time() - t_before_exec
                for req, wav in zip(batch_full, wav_results):
                    wav_24k = _resample_to_24k(wav, sr)
                    pcm16 = _float_to_pcm16(wav_24k)
                    if not req["future"].done():
                        req["future"].set_result((pcm16, TARGET_SAMPLE_RATE))
                total_s = time.time() - decode_start
                logger.info(
                    f"[decoder] batch_size={len(batch_full)} full executor_ms={exec_s*1000:.2f} total_ms={total_s*1000:.2f}"
                )

            if batch_window:
                batch_inputs = [
                    {"audio_codes": r["audio_codes"], "left_context_frames": r["left_context_frames"]}
                    for r in batch_window
                ]
                t_before = time.time()
                decode_window_batched = getattr(tokenizer, "decode_window_batched", None)
                if _mp_decoder_process is not None and _mp_decoder_process.is_alive() and decode_window_batched is not None:
                    # Send to dedicated MP worker when tokenizer supports window batch decode.
                    job_id = str(uuid.uuid4())
                    _mp_decoder_request_queue.put({"job_id": job_id, "batch": batch_inputs})
                    result = await loop.run_in_executor(None, _mp_decoder_result_queue.get)
                    exec_s = time.time() - t_before
                    if result.get("error") is not None:
                        for req in batch_window:
                            if not req["future"].done():
                                req["future"].set_exception(result["error"])
                    else:
                        wav_list, sr = result["wav_list"], result["sr"]
                        for req, wav in zip(batch_window, wav_list):
                            wav_24k = _resample_to_24k(wav, sr)
                            pcm16 = _float_to_pcm16(wav_24k)
                            if not req["future"].done():
                                req["future"].set_result((pcm16, TARGET_SAMPLE_RATE))
                else:
                    # Main process: use decode_window_batched if available, else full decode + trim.
                    def _do_window(inputs=batch_inputs, tok=tokenizer):
                        if getattr(tok, "decode_window_batched", None) is not None:
                            return tok.decode_window_batched(inputs)
                        return _decode_window_fallback(tok, inputs)
                    wav_list, sr = await loop.run_in_executor(None, _do_window)
                    exec_s = time.time() - t_before
                    for req, wav in zip(batch_window, wav_list):
                        wav_24k = _resample_to_24k(wav, sr)
                        pcm16 = _float_to_pcm16(wav_24k)
                        if not req["future"].done():
                            req["future"].set_result((pcm16, TARGET_SAMPLE_RATE))
                total_s = time.time() - decode_start
                logger.info(
                    f"[decoder] batch_size={len(batch_window)} (window batched) mp_ms={exec_s*1000:.2f} total_ms={total_s*1000:.2f}"
                )
        except Exception as e:
            for req in batch:
                if not req["future"].done():
                    req["future"].set_exception(e)


async def _decode_batched(audio_codes: list, left_context_frames: int = None) -> tuple[np.ndarray, int]:
    """Submit a decode request to the batched worker and await the result.

    If left_context_frames is set, only the window (context+new) is decoded and trimmed (streaming).
    """
    future = asyncio.get_event_loop().create_future()
    payload = {"audio_codes": audio_codes, "future": future}
    if left_context_frames is not None:
        payload["left_context_frames"] = left_context_frames
    await _decode_queue.put(payload)
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


def get_voice_clone_code_prefix(voice_clone_prompt: dict, num_frames: int = VOICE_CLONE_CODE_PREFIX_FRAMES) -> list:
    """Extract last num_frames code frames from voice clone prompt ref_code for use as decode prefix.

    ref_code is [time, 16] (tensor or ndarray). Returns list of num_frames frames, each frame
    a list of 16 ints, to match the format expected by the tokenizer (list of frames).
    Gives the tokenizer enough context so the first decoded chunks are stable; we skip
    this prefix when yielding audio to the client.
    """
    ref_code = voice_clone_prompt.get("ref_code")
    if ref_code is None:
        return []
    n = min(num_frames, len(ref_code))
    if n <= 0:
        return []
    if hasattr(ref_code, "cpu"):
        last = ref_code[-n:].cpu()
        return last.tolist()
    if hasattr(ref_code, "tolist"):
        return ref_code[-n:].tolist()
    return list(ref_code[-n:])




async def generate_speech_stream(request: SpeechRequest):
    """
    Streaming decode: producer feeds code chunks to a queue; consumer decodes
    (window + context) and yields PCM audio chunks.

    CANCELLATION SAFETY: when the client disconnects mid-stream, we must:
      1. Cancel the producer task (stops consuming from gen immediately)
      2. Close gen (triggers interface.py's finally → clear_request)
    Without this, a ghost sequence stays in scheduler.running and causes
    200ms batch_wait timeouts on every subsequent talker step.
    """
    request_start_time = time.time()
    request_status = "success"
    gen = None
    producer_task = None
    debug_chunks: list[bytes] = []
    first_audio_observed = False
    try:
        # Send leading silence immediately so client gets audio right away
        silence = _get_leading_silence_bytes()
        debug_chunks.append(silence)
        yield silence
        await asyncio.sleep(0)  # flush first chunk to client

        interface = get_interface()
        tokenizer = get_tokenizer()
        loop = asyncio.get_event_loop()
        start_time = time.time()

        # --- Create the async generator (voice clone or custom voice) ---
        text = "..." + request.text
        logger.info(f"[stream] request text length={len(request.text)} speaker={request.speaker!r} -> feeding to model")
        code_prefix: list = []
        prefix_len = 0
        if is_voice_clone(request.speaker):
            voice_clone_prompt = load_voice_clone_prompt(request.speaker)
            code_prefix = get_voice_clone_code_prefix(voice_clone_prompt)
            prefix_len = len(code_prefix)
            if prefix_len:
                logger.info(f"[voice_clone] using {prefix_len} ref_code frames as decode prefix (first chunks will be skipped)")
            gen = interface.generate_voice_clone_async(
                text=text,
                language=request.language,
                voice_clone_prompt=voice_clone_prompt,
            )
        else:
            gen = interface.generate_custom_voice_async(
                text=text,
                language=request.language,
                speaker=request.speaker,
            )

        # --- Producer feeds code chunks; consumer decodes and yields audio ---
        codes_queue: asyncio.Queue[list | None] = asyncio.Queue()  # unbounded
        prev_code_pos = 0

        async def producer() -> None:
            audio_codes: list = []
            last_chunk_time = start_time
            first_put = True
            try:
                async for audio_code in gen:
                    current_time = time.time()
                    inner_latency = current_time - last_chunk_time
                    logger.info(f"[producer] inner chunk latency: {inner_latency*1000:.2f}ms")
                    last_chunk_time = current_time

                    audio_codes.append(audio_code)
                    n = len(audio_codes)
                    # First chunks (1-8): yield every 2 codes. Later: yield every 4.
                    if n <= _first_codes_threshold:
                        if n % FIRST_CHUNK_SIZE == 0:
                            await codes_queue.put(code_prefix + list(audio_codes))
                            if first_put:
                                logger.info("[producer] first batch put, codes=%d", n)
                                first_put = False
                    else:
                        if n % STREAMING_CHUNK_SIZE == 0:
                            await codes_queue.put(code_prefix + list(audio_codes))
                # Final partial chunk
                if audio_codes:
                    n = len(audio_codes)
                    if n <= _first_codes_threshold and n % FIRST_CHUNK_SIZE != 0:
                        await codes_queue.put(code_prefix + list(audio_codes))
                    elif n > _first_codes_threshold and n % STREAMING_CHUNK_SIZE != 0:
                        await codes_queue.put(code_prefix + list(audio_codes))
            except asyncio.CancelledError:
                logger.warning("[producer] cancelled (client likely disconnected)")
                raise
            except Exception as e:
                logger.exception("[producer] exception: %s", e)
                raise
            finally:
                logger.info("[producer] done, total_codes=%d", len(audio_codes))
                await codes_queue.put(None)  # sentinel

        producer_task = asyncio.create_task(producer())
        logger.info("[stream] producer started, consumer waiting for codes from model ...")

        chunk_index = 0
        last_audio_chunk_time = time.time()
        try:
            while True:
                item = await codes_queue.get()
                if item is None:
                    break
                if len(item) <= prev_code_pos:
                    continue

                t_chunk_start = time.time()
                # Decode only window [prev_code_pos - context : end], trim to new part
                start = max(0, prev_code_pos - STREAMING_CONTEXT_SIZE)
                window = item[start:]
                context_frames = prev_code_pos - start
                is_first_chunk = prev_code_pos == 0
                # When using voice clone prefix: decode with context (full window), then we ignore prefix audio when yielding below
                if prefix_len > 0 and is_first_chunk:
                    left_context_frames = 0  # get full decode; we trim prefix at yield
                else:
                    left_context_frames = context_frames

                if _decode_queue is not None:
                    pcm16, _ = await _decode_batched(window, left_context_frames=left_context_frames)
                else:
                    # No decode queue: decode full window then trim (tokenizer may not have decode_window)
                    wav_list, sr = _decode_window_fallback(
                        tokenizer,
                        [{"audio_codes": window, "left_context_frames": left_context_frames}],
                    )
                    wav_24k = _resample_to_24k(wav_list[0], sr)
                    pcm16 = _float_to_pcm16(wav_24k)

                # Ignore prefix audio when yielding to user (voice clone: drop first prefix_len frames of decoded audio)
                if prefix_len > 0 and is_first_chunk and len(pcm16) > 0:
                    skip_samples = prefix_len * _samples_per_frame()
                    if skip_samples < len(pcm16):
                        pcm16 = pcm16[skip_samples:]
                    else:
                        pcm16 = np.array([], dtype=pcm16.dtype)

                prev_code_pos = len(item)
                if len(pcm16) > 0:
                    pcm16 = _apply_volume_pcm16(pcm16, request.volume)
                    chunk_index += 1
                    decode_post_s = time.time() - t_chunk_start
                    interval_s = t_chunk_start - last_audio_chunk_time
                    last_audio_chunk_time = time.time()
                    logger.info(
                        f"[stream] audio chunk #{chunk_index} decode_post_ms={decode_post_s*1000:.2f} "
                        f"interval_since_prev_ms={interval_s*1000:.2f} samples={len(pcm16)}"
                    )
                    chunk_bytes = pcm16.tobytes()
                    if not first_audio_observed:
                        first_audio_observed = True
                    debug_chunks.append(chunk_bytes)
                    yield chunk_bytes
                    await asyncio.sleep(0)  # yield so ASGI can flush chunk to client
        finally:
            n_chunks = len(debug_chunks)
            n_bytes = sum(len(c) for c in debug_chunks)
            logger.info(f"[stream] ended: {n_chunks} chunks, {n_bytes} bytes total")
            _save_debug_audio(debug_chunks, request.text)
            # Cancel producer immediately -- don't wait for it to finish naturally.
            # Without cancel(), a disconnected client leaves the producer running
            # for seconds, keeping the ghost sequence in scheduler.running.
            producer_task.cancel()
            try:
                await producer_task
            except (asyncio.CancelledError, Exception):
                pass
    except StopAsyncIteration:
        # Generator produced no codes at all
        pass
    except asyncio.CancelledError:
        logger.warning("[stream] stream task cancelled (client likely disconnected)")
        raise
    except Exception as e:
        logger.error(f"[generate_speech_stream] Error: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        # ALWAYS close the interface generator to trigger its cleanup
        # (clear_request in interface.py's finally block removes the sequence
        # from scheduler.running, preventing ghost sequence / batch_wait timeouts).
        if gen is not None:
            try:
                await gen.aclose()
            except Exception:
                pass  # may already be closed from normal EOS

@app.post("/v1/audio/speech", response_class=StreamingResponse)
async def generate_speech(request: SpeechRequest):
    """
    Generate speech from text.
    Returns raw PCM 16-bit mono at 24 kHz (audio/L16).
    Uses async generation (multiprocess talker + predictor).
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
        "multiprocess_engines": True,
    }


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    logger.info("Starting Qwen3-TTS API with multiprocess engines (talker + predictor workers).")
    uvicorn.run(app, host=host, port=port)
