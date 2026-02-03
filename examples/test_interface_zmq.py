"""
Standalone test for Qwen3TTSInterface with ZMQ (asyncio engine loop + asyncio queue).

Usage:
  cd nano-qwen3tts-vllm/examples
  python test_interface_zmq.py --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice

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

Requires: pyzmq (with asyncio), msgpack, nano_qwen3tts_vllm, qwen_tts (tokenizer), soundfile.
"""

import argparse
import asyncio
import os
import sys
import time


# Add parent so we can import nano_qwen3tts_vllm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


async def main():
    parser = argparse.ArgumentParser(
        description="Test Qwen3TTSInterface with ZMQ (asyncio engine loop + asyncio queue)"
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
    
    # Get values from args or env vars with defaults
    model_path = args.model_path or os.environ.get("QWEN3_TTS_MODEL_PATH", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    text = args.text or os.environ.get("TEST_TEXT", "Hi there, this is a ZMQ test.")
    language = args.language
    speaker = args.speaker
    out_wav = args.output or os.environ.get("OUT_WAV", "output_zmq_test.wav")

    print("=" * 60)
    print("Qwen3-TTS Interface + ZMQ test (asyncio engine loop)")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Text:  {text}")
    print(f"Output: {out_wav}")
    print()

    from nano_qwen3tts_vllm.zmq import ZMQOutputBridge
    import warnings
    
    print("[1/4] Initializing ZMQ bridge...")
    try:
        # Capture warnings to show port changes
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bridge = ZMQOutputBridge(auto_find_port=True)
            if w:
                for warning in w:
                    print(f"      ⚠ {warning.message}")
        print(f"      ✓ ZMQ bridge bound to {bridge.bind_address}")
    except Exception as e:
        print(f"      ✗ Failed to initialize ZMQ bridge: {e}")
        sys.exit(1)

    from nano_qwen3tts_vllm.interface import Qwen3TTSInterface
    print("[2/4] Loading interface with zmq_bridge...")
    
    # Check if it's a local path or HuggingFace model ID
    if os.path.isdir(model_path) or os.path.isfile(model_path):
        # Local path - use regular init
        print(f"      Using local model path: {model_path}")
        interface = Qwen3TTSInterface(model_path=model_path, zmq_bridge=bridge, enforce_eager=False)
    else:
        # HuggingFace model ID - use from_pretrained
        print(f"      Detected HuggingFace model ID: {model_path}")
        print(f"      Downloading if needed...")
        interface = Qwen3TTSInterface.from_pretrained(
            pretrained_model_name_or_path=model_path,
            zmq_bridge=bridge,
            enforce_eager=False,
        )
    
    await interface.start_zmq_tasks()
    print("      ✓ ZMQ dispatcher and engine loop tasks are running (no threads).")

    print("[3/4] Generating (async: add_request + await queue.get)...")
    audio_codes = []
    first_chunk_time = None
    last_chunk_time = None
    start_time = time.time()
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
    print("✓ OK: ZMQ interface test passed.")
    print("=" * 60)
    bridge.close()


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
