"""
Standalone test for Qwen3TTSInterface with async generation (multiprocess only).

Talker and predictor run in separate processes; main event loop is never blocked.

Usage:
  cd nano-qwen3tts-vllm  # or nano-qwen3tts-vllm/examples
  python examples/test_interface_zmq.py --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice

Arguments:
  --model-path          - Model directory or HuggingFace model ID (required)
  --text                - Text to synthesize (optional)
  --language            - Language of the text (default: English)
  --speaker             - Speaker name (default: Vivian)
  --output              - Output WAV path (default: output_zmq_test.wav)

Env (optional, overridden by args):
  QWEN3_TTS_MODEL_PATH  - model dir or HuggingFace model ID
  TEST_TEXT             - text to synthesize
  OUT_WAV               - output WAV path
  GPU_MEMORY_UTILIZATION - GPU memory fraction (default 0.9)

Requires: pyzmq, nano_qwen3tts_vllm, qwen_tts (tokenizer), soundfile.
"""

import argparse
import asyncio
import os
import sys
import time
import logging


logger = logging.getLogger(__name__)

# Ensure log messages appear on console (works when run as uvicorn server:app or python server.py)
if not logging.getLogger().handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(_handler)
    logging.getLogger().setLevel(logging.DEBUG if os.environ.get("DEBUG_TTS") else logging.INFO)



# Add parent so we can import nano_qwen3tts_vllm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


async def main():
    parser = argparse.ArgumentParser(
        description="Test Qwen3TTSInterface with multiprocess async generation"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Model directory or HuggingFace model ID (e.g., Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice). "
             "Can also be set via QWEN3_TTS_MODEL_PATH env var.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to synthesize. Can also be set via TEST_TEXT env var.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        help="Language of the text (default: English)",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="Vivian",
        help="Speaker name (default: Vivian)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output WAV path. Can also be set via OUT_WAV env var. (default: output_zmq_test.wav)",
    )
    args = parser.parse_args()

    os.environ["USE_MULTIPROCESS_ENGINES"] = "1"

    # Get values from args or env vars with defaults
    model_path = args.model_path or os.environ.get("QWEN3_TTS_MODEL_PATH", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    text = args.text or os.environ.get("TEST_TEXT", "Hi there, this is a ZMQ test.")
    language = args.language
    speaker = args.speaker
    out_wav = args.output or os.environ.get("OUT_WAV", "output_zmq_test.wav")

    print("=" * 60)
    print("Qwen3-TTS Interface test (multiprocess: talker + predictor in separate processes)")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Text:  {text}")
    print(f"Output: {out_wav}")
    print()

    gpu_mem_util = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.9"))
    print("[1/4] GPU_MEMORY_UTILIZATION={}".format(gpu_mem_util))

    from nano_qwen3tts_vllm.interface import Qwen3TTSInterface
    print("[2/4] Loading interface...")

    if os.path.isdir(model_path) or os.path.isfile(model_path):
        print(f"      Using local model path: {model_path}")
        interface = Qwen3TTSInterface(
            model_path=model_path,
            enforce_eager=False,
            gpu_memory_utilization=gpu_mem_util,
        )
    else:
        print(f"      Detected HuggingFace model ID: {model_path}")
        print(f"      Downloading if needed...")
        interface = Qwen3TTSInterface.from_pretrained(
            pretrained_model_name_or_path=model_path,
            enforce_eager=False,
            gpu_memory_utilization=gpu_mem_util,
        )

    await interface.start_zmq_tasks()
    print("      ✓ Multiprocess engines started (talker + predictor workers + async loops).")

    print("[3/4] Generating (async: add_request + await queue.get)...")
    audio_codes = []
    first_chunk_time = None
    last_chunk_time = None
    start_time = time.time()
    try:
        async for chunk in interface.generate_custom_voice_async(text=text, language=language, speaker=speaker):
            current_time = time.time()
            if first_chunk_time is None:
                first_chunk_time = current_time
                first_chunk_latency = current_time - start_time
                print(f"      chunk #{len(audio_codes) + 1}: {len(chunk)} codes (first chunk latency: {first_chunk_latency:.3f}s)")
            else:
                inner_latency = current_time - last_chunk_time if last_chunk_time else 0
                print(f"      chunk #{len(audio_codes) + 1}: {len(chunk)} codes (inner latency: {inner_latency:.3f}s)")
            audio_codes.append(chunk)
            last_chunk_time = current_time
        
        total_time = time.time() - start_time
        print(f"      ✓ Received {len(audio_codes)} chunks total in {total_time:.3f}s")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"      ✗ Error generating: {e}")
        sys.exit(1)

    if not audio_codes:
        print("[FAIL] No audio chunks received.")
        sys.exit(1)

    await interface.stop_zmq_tasks()

    print("[4/4] Decoding to WAV...")
    try:
        from qwen_tts import Qwen3TTSTokenizer
        tokenizer = Qwen3TTSTokenizer.from_pretrained(
            "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            device_map="cuda:0",
        )
        wav_list, sr = tokenizer.decode([{"audio_codes": audio_codes}])
        wav = wav_list[0]
        import soundfile as sf
        sf.write(out_wav, wav, sr)
        duration = len(wav) / sr
        print(f"      ✓ Saved {out_wav} ({duration:.2f}s @ {sr} Hz)")
    except ImportError:
        print(f"      ⚠ Warning: qwen_tts not available. Skipping WAV decoding.")
        print(f"      Audio codes collected: {len(audio_codes)} chunks")
        print(f"      Install qwen_tts to enable WAV decoding: pip install qwen-tts")

    print()
    print("=" * 60)
    print("✓ OK: Interface test passed.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Test cancelled by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[FAIL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
