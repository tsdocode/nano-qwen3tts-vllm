"""
Quick benchmark comparison script.

Simple side-by-side comparison of nano-vllm vs original Qwen3-TTS.

Note: This benchmark uses Voice Design mode for consistent comparison.
nano-vllm uses streaming API (generate -> decode), while original returns audio directly.

Usage:
    python quick_benchmark.py --model-path Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "nano-qwen3tts-vllm"))
sys.path.insert(0, str(Path(__file__).parent / "Qwen3-TTS"))

TEST_TEXT = "Hello world! This is a benchmark test to compare the performance of different implementations."
VOICE_DESIGN_TEXT = "A professional female voice with clear pronunciation"


def benchmark_nano_vllm(model_path: str, num_runs: int = 5):
    """Benchmark nano-vllm."""
    from nano_qwen3tts_vllm.interface import Qwen3TTSInterface
    
    print("\n" + "="*80)
    print("Loading nano-qwen3tts-vllm...")
    print("="*80)
    
    load_start = time.time()
    interface = Qwen3TTSInterface.from_pretrained(
        pretrained_model_name_or_path=model_path,
        enforce_eager=False,
        tensor_parallel_size=1,
    )
    load_time = time.time() - load_start
    print(f"✓ Model loaded in {load_time:.2f}s")
    
    # Warmup
    print("Warming up...")
    warmup_codes = list(interface.generate_voice_design(
        text="warmup",
        language="English",
        instruct=VOICE_DESIGN_TEXT,
    ))
    
    # Benchmark
    print(f"\nRunning {num_runs} iterations...")
    times = []
    audio_durations = []
    
    for i in range(num_runs):
        start = time.time()
        
        # Generate codec chunks
        audio_codes = list(interface.generate_voice_design(
            text=TEST_TEXT,
            language="English",
            instruct=VOICE_DESIGN_TEXT,
        ))
        
        # Decode to audio
        wavs, sr = interface.speech_tokenizer.decode([{"audio_codes": audio_codes}])
        
        elapsed = time.time() - start
        audio_duration = len(wavs[0]) / sr
        
        times.append(elapsed)
        audio_durations.append(audio_duration)
        print(f"  Run {i+1}: {elapsed:.3f}s (audio: {audio_duration:.2f}s, RTF: {elapsed/audio_duration:.3f})")
    
    avg_time = sum(times) / len(times)
    avg_duration = sum(audio_durations) / len(audio_durations)
    avg_rtf = avg_time / avg_duration
    
    print(f"\nResults:")
    print(f"  Avg generation time: {avg_time:.3f}s")
    print(f"  Avg audio duration:  {avg_duration:.2f}s")
    print(f"  Avg RTF:             {avg_rtf:.3f}")
    
    return avg_time, avg_rtf


def benchmark_original(model_path: str, num_runs: int = 5):
    """Benchmark original Qwen3-TTS."""
    import torch
    from qwen_tts import Qwen3TTSModel
    
    print("\n" + "="*80)
    print("Loading original Qwen3-TTS...")
    print("="*80)
    
    load_start = time.time()
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    load_time = time.time() - load_start
    print(f"✓ Model loaded in {load_time:.2f}s")
    
    # Warmup
    print("Warming up...")
    model.generate_voice_design(
        text="warmup",
        language="English",
        instruct=VOICE_DESIGN_TEXT,
    )
    
    # Benchmark
    print(f"\nRunning {num_runs} iterations...")
    times = []
    audio_durations = []
    
    for i in range(num_runs):
        start = time.time()
        wavs = model.generate_voice_design(
            text=TEST_TEXT,
            language="English",
            instruct=VOICE_DESIGN_TEXT,
        )
        elapsed = time.time() - start
        # sr is returned from generate_voice_design, but fallback to speech_tokenizer
        if isinstance(wavs, tuple):
            wavs, sr = wavs
        else:
            sr = model.speech_tokenizer.sample_rate if hasattr(model, 'speech_tokenizer') else 24000
        audio_duration = len(wavs[0]) / sr
        
        times.append(elapsed)
        audio_durations.append(audio_duration)
        print(f"  Run {i+1}: {elapsed:.3f}s (audio: {audio_duration:.2f}s, RTF: {elapsed/audio_duration:.3f})")
    
    avg_time = sum(times) / len(times)
    avg_duration = sum(audio_durations) / len(audio_durations)
    avg_rtf = avg_time / avg_duration
    
    print(f"\nResults:")
    print(f"  Avg generation time: {avg_time:.3f}s")
    print(f"  Avg audio duration:  {avg_duration:.2f}s")
    print(f"  Avg RTF:             {avg_rtf:.3f}")
    
    return avg_time, avg_rtf


def main():
    parser = argparse.ArgumentParser(description="Quick benchmark comparison")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        help="Model path or HuggingFace ID (VoiceDesign model recommended)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of runs",
    )
    args = parser.parse_args()
    
    print("\n" + "#"*80)
    print("QUICK BENCHMARK: nano-vllm vs Original Qwen3-TTS")
    print("#"*80)
    print(f"Model: {args.model_path}")
    print(f"Mode: Voice Design")
    print(f"Voice: {VOICE_DESIGN_TEXT}")
    print(f"Test text: {TEST_TEXT}")
    print(f"Number of runs: {args.num_runs}")
    
    # Benchmark nano-vllm
    nano_time, nano_rtf = benchmark_nano_vllm(args.model_path, args.num_runs)
    
    # Benchmark original
    original_time, original_rtf = benchmark_original(args.model_path, args.num_runs)
    
    # Comparison
    print("\n" + "#"*80)
    print("COMPARISON")
    print("#"*80)
    speedup = original_time / nano_time
    rtf_improvement = original_rtf / nano_rtf
    
    print(f"\n{'Metric':<30} {'nano-vllm':<15} {'Original':<15} {'Improvement':<15}")
    print("-"*80)
    print(f"{'Avg Generation Time':<30} {nano_time:.3f}s        {original_time:.3f}s        {speedup:.2f}x faster")
    print(f"{'Avg RTF':<30} {nano_rtf:.3f}          {original_rtf:.3f}          {rtf_improvement:.2f}x better")
    
    print(f"\n🚀 nano-vllm is {speedup:.2f}x faster than original Qwen3-TTS!")
    print(f"📊 RTF improved by {rtf_improvement:.2f}x")
    print("#"*80 + "\n")


if __name__ == "__main__":
    main()
