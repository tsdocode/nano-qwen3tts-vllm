"""
Standalone test for Qwen3TTSInterface with ZMQ (asyncio engine loop + asyncio queue).

Usage:
  cd nano-qwen3tts-vllm/examples
  python test_interface_zmq.py

Env (optional):
  QWEN3_TTS_MODEL_PATH  - model dir (default: .../qwen3tts)
  TEST_TEXT             - text to synthesize
  OUT_WAV               - output WAV path (default: output_zmq_test.wav)

Requires: pyzmq (with asyncio), msgpack, nano_qwen3tts_vllm, qwen_tts (tokenizer), soundfile.
"""

import asyncio
import os
import sys
import time


# Add parent so we can import nano_qwen3tts_vllm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


async def main():
    model_path = os.environ.get("QWEN3_TTS_MODEL_PATH", "/home/sang/work/weights/qwen3tts")
    text = os.environ.get("TEST_TEXT", "Hi there, this is a ZMQ test.")
    language = "English"
    speaker = "Vivian"
    out_wav = os.environ.get("OUT_WAV", "output_zmq_test.wav")

    print("=" * 60)
    print("Qwen3-TTS Interface + ZMQ test (asyncio engine loop)")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Text:  {text}")
    print(f"Output: {out_wav}")
    print()

    from nano_qwen3tts_vllm.zmq import ZMQOutputBridge
    bridge = ZMQOutputBridge()
    print("[1/4] ZMQ bridge bound to", bridge.bind_address)

    from nano_qwen3tts_vllm.interface import Qwen3TTSInterface
    print("[2/4] Loading interface with zmq_bridge...")
    interface = Qwen3TTSInterface(model_path=model_path, zmq_bridge=bridge, enforce_eager=False)
    await interface.start_zmq_tasks()
    print("      ZMQ dispatcher and engine loop tasks are running (no threads).")

    print("[3/4] Generating (async: add_request + await queue.get)...")
    audio_codes = []
    first_chunk_time = None
    last_chunk_time = None
    start_time= time.time()
    async for chunk in interface.generate_custom_voice_async(text=text, language=language, speaker=speaker):
        current_time = time.time()
        if first_chunk_time is None:
            first_chunk_time = current_time
            print(f"      chunk #{len(audio_codes) + 1}: {len(chunk)} codes (first chunk latency: {(current_time - start_time):.3f}s)")
        else:
            inner_latency = current_time - last_chunk_time if last_chunk_time else 0
            print(f"      chunk #{len(audio_codes) + 1}: {len(chunk)} codes (inner latency: {inner_latency:.3f}s)")
        audio_codes.append(chunk)
        last_chunk_time = current_time
    print(f"      Received {len(audio_codes)} chunks total.")

    if not audio_codes:
        print("[FAIL] No audio chunks received.")
        sys.exit(1)

    await interface.stop_zmq_tasks()

    print("[4/4] Decoding to WAV...")
    from qwen_tts import Qwen3TTSTokenizer
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        device_map="cuda:0",
    )
    wav_list, sr = tokenizer.decode([{"audio_codes": audio_codes}])
    wav = wav_list[0]
    import soundfile as sf
    sf.write(out_wav, wav, sr)
    print(f"      Saved {out_wav} ({len(wav) / sr:.2f}s @ {sr} Hz)")

    print()
    print("OK: ZMQ interface test passed.")
    bridge.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
