"""
Batch TTS generation using async API (concurrent streams).

Runs multiple generate_custom_voice_async() calls concurrently with asyncio.gather,
so the multiprocess talker/predictor can batch and interleave work.

Usage:
  cd nano-qwen3tts-vllm
  python examples/batch_async_example.py --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --batch-size 4
  python examples/batch_async_example.py --model-path /path/to/model --batch-size 8 --output-dir ./batch_out

Arguments:
  --model-path   - Model directory or HuggingFace model ID (required)
  --batch-size   - Number of concurrent generation streams (default: 4)
  --output-dir   - Directory to write WAV files (default: ./batch_output)
  --language     - Language (default: English)
  --speaker      - Speaker name (default: Vivian)

Env:
  QWEN3_TTS_MODEL_PATH, GPU_MEMORY_UTILIZATION

Requires: pyzmq, nano_qwen3tts_vllm, qwen_tts, soundfile.
"""

import argparse
import asyncio
import os
import sys
import time

# Add parent so we can import nano_qwen3tts_vllm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Default sentences for batch (short and long)
DEFAULT_TEXTS = [
    "Hello, this is the first sentence.",
    "The quick brown fox jumps over the lazy dog.",
    "Welcome to batch async generation with Qwen3-TTS.",
    "Multiple streams run concurrently on the same engine.",
    "Text to speech can be fast with batching.",
    "Async generation allows overlapping compute and memory transfer.",
    "We use asyncio.gather to run several generations at once.",
    "Each stream yields code chunks that are decoded to audio.",
]


async def collect_codes(interface, text: str, language: str, speaker: str, index: int):
    """Run one async generation and return (index, list of code chunks, first_chunk_latency_s)."""
    codes = []
    first_chunk_time = None
    start = time.perf_counter()
    try:
        async for chunk in interface.generate_custom_voice_async(
            text=text, language=language, speaker=speaker
        ):
            codes.append(chunk)
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter() - start
    except Exception as e:
        return (index, None, None, str(e))
    return (index, codes, first_chunk_time, None)


async def main():
    parser = argparse.ArgumentParser(description="Batch async TTS generation")
    parser.add_argument("--model-path", type=str, default=None, help="Model path or HuggingFace ID")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of concurrent streams")
    parser.add_argument("--output-dir", type=str, default="./batch_output", help="Output WAV directory")
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument("--speaker", type=str, default="Vivian")
    parser.add_argument(
        "--texts",
        type=str,
        nargs="+",
        default=None,
        help="Custom texts (otherwise use built-in list up to batch-size)",
    )
    args = parser.parse_args()

    model_path = args.model_path or os.environ.get("QWEN3_TTS_MODEL_PATH")
    if not model_path:
        print("Error: --model-path or QWEN3_TTS_MODEL_PATH required")
        sys.exit(1)

    batch_size = args.batch_size
    texts = args.texts if args.texts else DEFAULT_TEXTS[:batch_size]
    if len(texts) < batch_size:
        # Repeat or pad with last
        while len(texts) < batch_size:
            texts.append(texts[-1] if texts else "Hello.")
    else:
        texts = texts[:batch_size]

    os.makedirs(args.output_dir, exist_ok=True)
    gpu_util = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.9"))

    print("=" * 60)
    print("Batch async TTS generation")
    print("=" * 60)
    print(f"Model:      {model_path}")
    print(f"Batch size: {batch_size}")
    print(f"Output dir: {args.output_dir}")
    print()

    from nano_qwen3tts_vllm.interface import Qwen3TTSInterface

    print("[1/3] Loading interface and starting engines...")
    if os.path.isdir(model_path) or os.path.isfile(model_path):
        interface = Qwen3TTSInterface(
            model_path=model_path,
            enforce_eager=False,
            gpu_memory_utilization=gpu_util,
        )
    else:
        interface = Qwen3TTSInterface.from_pretrained(
            pretrained_model_name_or_path=model_path,
            enforce_eager=False,
            gpu_memory_utilization=gpu_util,
        )
    await interface.start_zmq_tasks()
    print("      Done.")

    print(f"[2/3] Running {batch_size} concurrent streams...")
    t0 = time.perf_counter()
    tasks = [
        collect_codes(interface, text, args.language, args.speaker, i)
        for i, text in enumerate(texts)
    ]
    results = await asyncio.gather(*tasks)
    total_s = time.perf_counter() - t0
    print(f"      All streams finished in {total_s:.3f}s")

    await interface.stop_zmq_tasks()

    # Summarize
    first_chunk_latencies = []
    for r in results:
        idx, codes, first_s, err = r
        if err:
            print(f"      [{idx}] Error: {err}")
            continue
        if first_s is not None:
            first_chunk_latencies.append(first_s)
        print(f"      [{idx}] {len(codes)} chunks, first chunk: {first_s*1000:.0f}ms" if first_s else f"      [{idx}] {len(codes)} chunks")

    if first_chunk_latencies:
        print(f"      Time to first chunk (min/avg/max): {min(first_chunk_latencies)*1000:.0f} / {sum(first_chunk_latencies)/len(first_chunk_latencies)*1000:.0f} / {max(first_chunk_latencies)*1000:.0f} ms")

    print("[3/3] Decoding to WAV...")
    try:
        from qwen_tts import Qwen3TTSTokenizer
        import soundfile as sf

        tokenizer = Qwen3TTSTokenizer.from_pretrained(
            "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            device_map="cuda:0",
        )
        for r in results:
            idx, codes, _, err = r
            if err or not codes:
                continue
            wav_list, sr = tokenizer.decode([{"audio_codes": codes}])
            wav = wav_list[0]
            out_path = os.path.join(args.output_dir, f"batch_{idx}.wav")
            sf.write(out_path, wav, sr)
            print(f"      Saved {out_path} ({len(wav)/sr:.2f}s)")
        print("      Done.")
    except ImportError as e:
        print(f"      Skip WAV (install qwen_tts and soundfile): {e}")

    print()
    print("=" * 60)
    print("Batch async example finished.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        sys.exit(130)
