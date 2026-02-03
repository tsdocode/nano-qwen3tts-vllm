"""
Example demonstrating Voice Design feature.

Voice Design allows you to generate speech with natural language voice/style instructions
using the Voice Design model (e.g., Qwen3-TTS-12Hz-1.7B-VoiceDesign).

Usage:
    # Using HuggingFace model ID (automatically downloads if needed)
    python examples/voice_design_example.py \
        --model-path Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
        --output-dir ./output
    
    # Using local model path
    python examples/voice_design_example.py \
        --model-path /path/to/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
        --output-dir ./output
"""

import argparse
import os
import sys
import time
from pathlib import Path

import soundfile as sf

# Add parent directory to path to import nano_qwen3tts_vllm
sys.path.insert(0, str(Path(__file__).parent.parent))

from nano_qwen3tts_vllm.interface import Qwen3TTSInterface


def main():
    parser = argparse.ArgumentParser(
        description="Example demonstrating Voice Design feature"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to Voice Design model or HuggingFace model ID (e.g., Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign or /path/to/local/model)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for generated audio files",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    
    print("=" * 60)
    print("Voice Design Example")
    print("=" * 60)
    
    # Initialize Voice Design model
    print(f"\nLoading Voice Design model from: {args.model_path}")
    
    # Check if it's a HuggingFace model ID or local path
    # Use from_pretrained which handles both cases automatically
    if os.path.isdir(args.model_path) or os.path.isfile(args.model_path):
        # Local path - use regular init
        interface = Qwen3TTSInterface(
            model_path=args.model_path,
            enforce_eager=False,
            tensor_parallel_size=1,
        )
    else:
        # Likely a HuggingFace model ID, use from_pretrained
        print("  Detected HuggingFace model ID, downloading if needed...")
        interface = Qwen3TTSInterface.from_pretrained(
            pretrained_model_name_or_path=args.model_path,
            enforce_eager=False,
            tensor_parallel_size=1,
        )
    print("✓ Model loaded successfully")
    
    # Example 1: Single voice design generation
    print("\n" + "-" * 60)
    print("[Example 1] Generating speech with voice design instruction")
    print("-" * 60)
    
    text = "H-hey! You dropped your... uh... calculus notebook? I mean, I think it's yours? Maybe?"
    instruct = "Male, 17 years old, tenor range, gaining confidence - deeper breath support now, though vowels still tighten when nervous"
    
    print(f"Text: {text}")
    print(f"Instruct: {instruct}")
    
    start_time = time.time()
    wavs, sr = interface.generate_voice_design(
        text=text,
        instruct=instruct,
        language="English",
    )
    elapsed = time.time() - start_time
    
    output_path = output_dir / "voice_design_example_1.wav"
    sf.write(str(output_path), wavs[0], sr)
    print(f"✓ Generated in {elapsed:.2f}s")
    print(f"  Saved to: {output_path}")
    
    # Example 2: Different voice characteristics
    print("\n" + "-" * 60)
    print("[Example 2] Different voice characteristics")
    print("-" * 60)
    
    examples = [
        {
            "text": "Hello, this is a test of voice design with a cheerful voice.",
            "instruct": "Female, 25 years old, cheerful and energetic",
            "output": "voice_design_cheerful.wav",
        },
        {
            "text": "This is a professional and calm voice demonstration.",
            "instruct": "Male, 40 years old, calm and professional",
            "output": "voice_design_professional.wav",
        },
        {
            "text": "I can sound like a young character with excitement!",
            "instruct": "Female, 18 years old, excited and enthusiastic",
            "output": "voice_design_excited.wav",
        },
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"  Text: {example['text']}")
        print(f"  Instruct: {example['instruct']}")
        
        start_time = time.time()
        wavs, sr = interface.generate_voice_design(
            text=example["text"],
            instruct=example["instruct"],
            language="English",
        )
        elapsed = time.time() - start_time
        
        output_path = output_dir / example["output"]
        sf.write(str(output_path), wavs[0], sr)
        print(f"  ✓ Generated in {elapsed:.2f}s, saved to: {output_path}")
    
    # Example 3: Batch generation
    print("\n" + "-" * 60)
    print("[Example 3] Batch generation with different instructions")
    print("-" * 60)
    
    texts = [
        "Hello, this is a test of voice design.",
        "I can generate different voices with natural language descriptions.",
        "Batch processing allows generating multiple voices efficiently.",
    ]
    instructs = [
        "Female, 25 years old, cheerful and energetic",
        "Male, 40 years old, calm and professional",
        "Female, 30 years old, friendly and warm",
    ]
    
    print(f"Generating {len(texts)} samples in batch...")
    start_time = time.time()
    wavs, sr = interface.generate_voice_design(
        text=texts,
        instruct=instructs,
        language=["English"] * len(texts),
    )
    elapsed = time.time() - start_time
    
    for i, wav in enumerate(wavs):
        output_path = output_dir / f"voice_design_batch_{i+1}.wav"
        sf.write(str(output_path), wav, sr)
        print(f"✓ Batch item {i+1} saved to: {output_path}")
    
    print(f"\n✓ Batch generation completed in {elapsed:.2f}s ({elapsed/len(texts):.2f}s per sample)")
    
    # Example 4: Different languages
    print("\n" + "-" * 60)
    print("[Example 4] Multi-language voice design")
    print("-" * 60)
    
    multilingual_examples = [
        {
            "text": "Hello, this is English voice design.",
            "instruct": "Male, 30 years old, clear and articulate",
            "language": "English",
            "output": "voice_design_english.wav",
        },
        {
            "text": "你好，这是中文语音设计测试。",
            "instruct": "Female, 25 years old, gentle and soft",
            "language": "Chinese",
            "output": "voice_design_chinese.wav",
        },
    ]
    
    for example in multilingual_examples:
        print(f"\nLanguage: {example['language']}")
        print(f"  Text: {example['text']}")
        print(f"  Instruct: {example['instruct']}")
        
        start_time = time.time()
        wavs, sr = interface.generate_voice_design(
            text=example["text"],
            instruct=example["instruct"],
            language=example["language"],
        )
        elapsed = time.time() - start_time
        
        output_path = output_dir / example["output"]
        sf.write(str(output_path), wavs[0], sr)
        print(f"  ✓ Generated in {elapsed:.2f}s, saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print(f"\nAll audio files saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
