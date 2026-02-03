"""
Example demonstrating Voice Clone feature.

Voice Clone allows you to clone a voice from reference audio and generate new speech
with that voice using the Base model (e.g., Qwen3-TTS-12Hz-1.7B-Base).

Usage:
    # With reference audio file
    python examples/voice_clone_example.py \
        --model-path /path/to/Qwen3-TTS-12Hz-1.7B-Base \
        --ref-audio reference.wav \
        --ref-text "Reference text corresponding to the audio" \
        --output-dir ./output
    
    # x_vector_only_mode (speaker embedding only, no reference text needed)
    python examples/voice_clone_example.py \
        --model-path /path/to/Qwen3-TTS-12Hz-1.7B-Base \
        --ref-audio reference.wav \
        --x-vector-only \
        --output-dir ./output
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# Add parent directory to path to import nano_qwen3tts_vllm
sys.path.insert(0, str(Path(__file__).parent.parent))

from nano_qwen3tts_vllm.interface import Qwen3TTSInterface


def main():
    parser = argparse.ArgumentParser(
        description="Example demonstrating Voice Clone feature"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to Base model or HuggingFace model ID (e.g., Qwen/Qwen3-TTS-12Hz-1.7B-Base or /path/to/local/model)",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        required=True,
        help="Path to reference audio file",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default="",
        help="Reference text corresponding to ref-audio (required for ICL mode, optional for x_vector_only_mode)",
    )
    parser.add_argument(
        "--x-vector-only",
        action="store_true",
        help="Use x_vector_only_mode (speaker embedding only, no ICL). ref-text not required.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for generated audio files",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.x_vector_only and not args.ref_text:
        parser.error("--ref-text is required when not using --x-vector-only mode (ICL mode requires reference text)")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    
    print("=" * 60)
    print("Voice Clone Example")
    print("=" * 60)
    
    # Initialize Base model for voice cloning
    print(f"\nLoading Base model from: {args.model_path}")
    
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
    
    # Load reference audio
    print(f"\nLoading reference audio from: {args.ref_audio}")
    ref_audio, ref_sr = sf.read(args.ref_audio)
    if ref_audio.ndim > 1:
        ref_audio = np.mean(ref_audio, axis=-1)
    print(f"✓ Reference audio loaded: {len(ref_audio)} samples at {ref_sr}Hz")
    
    # Create voice clone prompt from reference audio
    print("\n" + "-" * 60)
    print("[Step 1] Creating voice clone prompt from reference audio")
    print("-" * 60)
    
    mode_str = "x_vector_only_mode" if args.x_vector_only else "ICL mode"
    print(f"Mode: {mode_str}")
    if args.ref_text:
        print(f"Reference text: {args.ref_text}")
    
    start_time = time.time()
    voice_clone_prompt = interface.create_voice_clone_prompt(
        ref_audio=(ref_audio, ref_sr),
        ref_text=args.ref_text if args.ref_text else None,
        x_vector_only_mode=args.x_vector_only,
    )
    elapsed = time.time() - start_time
    
    print(f"✓ Voice clone prompt created in {elapsed:.2f}s")
    print(f"  - Speaker embedding shape: {voice_clone_prompt['ref_spk_embedding'][0].shape}")
    if voice_clone_prompt['ref_code'][0] is not None:
        print(f"  - Reference code shape: {voice_clone_prompt['ref_code'][0].shape}")
    else:
        print(f"  - Reference code: None (x_vector_only_mode)")
    print(f"  - ICL mode: {voice_clone_prompt['icl_mode'][0]}")
    print(f"  - x_vector_only_mode: {voice_clone_prompt['x_vector_only_mode'][0]}")
    
    # Example 1: Single voice clone generation
    print("\n" + "-" * 60)
    print("[Example 1] Generating speech with cloned voice (single)")
    print("-" * 60)
    
    sentence1 = "No problem! I actually... kinda finished those already? If you want to compare answers or something..."
    
    print(f"Text: {sentence1}")
    start_time = time.time()
    wavs, sr = interface.generate_voice_clone(
        text=sentence1,
        language="English",
        voice_clone_prompt=voice_clone_prompt,
    )
    elapsed = time.time() - start_time
    
    output_path = output_dir / "voice_clone_example_1.wav"
    sf.write(str(output_path), wavs[0], sr)
    print(f"✓ Generated in {elapsed:.2f}s")
    print(f"  Saved to: {output_path}")
    
    # Example 2: Another single generation with same prompt
    print("\n" + "-" * 60)
    print("[Example 2] Generating another sentence with same cloned voice")
    print("-" * 60)
    
    sentence2 = "What? No! I mean yes but not like... I just think you're... your titration technique is really precise!"
    
    print(f"Text: {sentence2}")
    start_time = time.time()
    wavs, sr = interface.generate_voice_clone(
        text=sentence2,
        language="English",
        voice_clone_prompt=voice_clone_prompt,
    )
    elapsed = time.time() - start_time
    
    output_path = output_dir / "voice_clone_example_2.wav"
    sf.write(str(output_path), wavs[0], sr)
    print(f"✓ Generated in {elapsed:.2f}s")
    print(f"  Saved to: {output_path}")
    
    # Example 3: Batch voice clone generation
    print("\n" + "-" * 60)
    print("[Example 3] Batch generation with cloned voice")
    print("-" * 60)
    
    sentences = [
        "No problem! I actually... kinda finished those already? If you want to compare answers or something...",
        "What? No! I mean yes but not like... I just think you're... your titration technique is really precise!",
        "This is a third sentence generated with the same cloned voice.",
    ]
    
    print(f"Generating {len(sentences)} samples in batch...")
    start_time = time.time()
    wavs, sr = interface.generate_voice_clone(
        text=sentences,
        language=["English"] * len(sentences),
        voice_clone_prompt=voice_clone_prompt,
    )
    elapsed = time.time() - start_time
    
    for i, wav in enumerate(wavs):
        output_path = output_dir / f"voice_clone_batch_{i+1}.wav"
        sf.write(str(output_path), wav, sr)
        print(f"✓ Batch item {i+1} saved to: {output_path}")
    
    print(f"\n✓ Batch generation completed in {elapsed:.2f}s ({elapsed/len(sentences):.2f}s per sample)")
    
    # Example 4: Compare ICL mode vs x_vector_only_mode (if ref_text was provided)
    if args.ref_text and not args.x_vector_only:
        print("\n" + "-" * 60)
        print("[Example 4] Comparing ICL mode vs x_vector_only_mode")
        print("-" * 60)
        
        test_text = "This sentence compares ICL mode with x_vector_only mode."
        
        # ICL mode (already created)
        print("\nICL mode (with reference code):")
        start_time = time.time()
        wavs_icl, sr = interface.generate_voice_clone(
            text=test_text,
            language="English",
            voice_clone_prompt=voice_clone_prompt,
        )
        elapsed_icl = time.time() - start_time
        output_path_icl = output_dir / "voice_clone_icl_mode.wav"
        sf.write(str(output_path_icl), wavs_icl[0], sr)
        print(f"  ✓ Generated in {elapsed_icl:.2f}s, saved to: {output_path_icl}")
        
        # x_vector_only_mode
        print("\nx_vector_only_mode (speaker embedding only):")
        voice_clone_prompt_xvec = interface.create_voice_clone_prompt(
            ref_audio=(ref_audio, ref_sr),
            ref_text=None,
            x_vector_only_mode=True,
        )
        start_time = time.time()
        wavs_xvec, sr = interface.generate_voice_clone(
            text=test_text,
            language="English",
            voice_clone_prompt=voice_clone_prompt_xvec,
        )
        elapsed_xvec = time.time() - start_time
        output_path_xvec = output_dir / "voice_clone_xvec_only.wav"
        sf.write(str(output_path_xvec), wavs_xvec[0], sr)
        print(f"  ✓ Generated in {elapsed_xvec:.2f}s, saved to: {output_path_xvec}")
        print(f"\n  Compare the two files to hear the difference!")
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print(f"\nAll audio files saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
