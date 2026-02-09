"""
Gradio app for Neto's Voice Clone feature.

This app allows you to:
1. Clone voices from reference audio and save them with custom names
2. Generate speech using saved voice clones (ICL mode only)

Usage:
    python examples/gradio_voice_clone.py --model-path Qwen/Neto's-12Hz-1.7B-Base
    
    # Or with local model
    python examples/gradio_voice_clone.py --model-path /path/to/local/model
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import gradio as gr
import torch

# Add parent directory to path to import nano_qwen3tts_vllm
sys.path.insert(0, str(Path(__file__).parent.parent))

from nano_qwen3tts_vllm.interface import Qwen3TTSInterface


# Global model instance
interface = None
VOICES_DIR = Path(__file__).parent / "voices"
VOICES_DIR.mkdir(exist_ok=True)


def get_saved_voices():
    """Get list of saved voice names from the voices directory."""
    if not VOICES_DIR.exists():
        return []
    voice_files = list(VOICES_DIR.glob("*.pkl"))
    voice_names = [f.stem for f in voice_files]
    return sorted(voice_names)


def clone_voice(ref_audio_path: str, ref_text: str, speaker_name: str) -> str:
    """
    Clone a voice from reference audio and save it.
    
    Args:
        ref_audio_path: Path to reference audio file
        ref_text: Reference text corresponding to the audio
        speaker_name: Name to save the voice clone as
    
    Returns:
        Success or error message
    """
    if not ref_audio_path:
        return "❌ Please upload a reference audio file."
    
    if not ref_text or not ref_text.strip():
        return "❌ Please provide reference text (required for ICL mode)."
    
    if not speaker_name or not speaker_name.strip():
        return "❌ Please provide a speaker name."
    
    # Sanitize speaker name for filename
    safe_name = "".join(c for c in speaker_name if c.isalnum() or c in (' ', '-', '_')).strip()
    if not safe_name:
        return "❌ Speaker name must contain at least one alphanumeric character."
    
    try:
        # Load reference audio
        ref_audio, ref_sr = sf.read(ref_audio_path)
        if ref_audio.ndim > 1:
            ref_audio = np.mean(ref_audio, axis=-1)
        
        # Create voice clone prompt (ICL mode only)
        voice_clone_prompt = interface.create_voice_clone_prompt(
            ref_audio=(ref_audio, ref_sr),
            ref_text=ref_text.strip(),
            x_vector_only_mode=False,  # ICL mode only
        )
        
        # Ensure tensors are on CPU for pickling
        if isinstance(voice_clone_prompt.get('ref_code'), torch.Tensor):
            voice_clone_prompt['ref_code'] = voice_clone_prompt['ref_code'].cpu()
        
        # Handle ref_spk_embedding (can be tensor or list of tensors)
        spk_emb = voice_clone_prompt.get('ref_spk_embedding')
        if isinstance(spk_emb, torch.Tensor):
            voice_clone_prompt['ref_spk_embedding'] = spk_emb.cpu()
        elif isinstance(spk_emb, list):
            voice_clone_prompt['ref_spk_embedding'] = [emb.cpu() if isinstance(emb, torch.Tensor) else emb for emb in spk_emb]
        
        # Save as pickle
        voice_path = VOICES_DIR / f"{safe_name}.pkl"
        with open(voice_path, 'wb') as f:
            pickle.dump(voice_clone_prompt, f)
        
        return f"✅ Voice cloned successfully! Saved as '{safe_name}'"
    
    except Exception as e:
        return f"❌ Error cloning voice: {str(e)}"


# Global state to store voice clone prompt for saving
_current_voice_clone_prompt = None
_current_is_new_voice = False


def generate_speech(
    text: str, 
    voice_selection: str, 
    language: str,
    ref_audio_path: str = None,
    ref_text: str = None,
) -> tuple:
    """
    Generate speech using a saved voice clone or new voice clone.
    
    Args:
        text: Text to synthesize
        voice_selection: Selected voice name or "New Voice Clone"
        language: Language selection
        ref_audio_path: Path to reference audio (for new voice clone)
        ref_text: Reference text (for new voice clone)
    
    Returns:
        Tuple of (sample_rate, audio_array) for Gradio Audio component, or None on error
    """
    global _current_voice_clone_prompt, _current_is_new_voice
    
    if not text or not text.strip():
        _current_is_new_voice = False
        return None, gr.update(visible=False), gr.update(visible=False), "❌ Please enter text to synthesize."
    
    if not voice_selection:
        _current_is_new_voice = False
        return None, gr.update(visible=False), gr.update(visible=False), "❌ Please select a voice."
    
    try:
        voice_clone_prompt = None
        
        if voice_selection == "New Voice Clone":
            # Create new voice clone
            if not ref_audio_path:
                _current_is_new_voice = False
                return None, gr.update(visible=False), gr.update(visible=False), "❌ Please upload reference audio for new voice clone."
            
            if not ref_text or not ref_text.strip():
                _current_is_new_voice = False
                return None, gr.update(visible=False), gr.update(visible=False), "❌ Please provide reference text (required for ICL mode)."
            
            # Load reference audio
            ref_audio, ref_sr = sf.read(ref_audio_path)
            if ref_audio.ndim > 1:
                ref_audio = np.mean(ref_audio, axis=-1)
            
            # Create voice clone prompt (ICL mode only)
            voice_clone_prompt = interface.create_voice_clone_prompt(
                ref_audio=(ref_audio, ref_sr),
                ref_text=ref_text.strip(),
                x_vector_only_mode=False,  # ICL mode only
            )
            
            # Store for potential saving
            _current_voice_clone_prompt = voice_clone_prompt
            _current_is_new_voice = True
        else:
            # Load existing voice clone
            voice_path = VOICES_DIR / f"{voice_selection}.pkl"
            if not voice_path.exists():
                _current_is_new_voice = False
                return None, gr.update(visible=False), gr.update(visible=False), f"❌ Voice '{voice_selection}' not found."
            
            with open(voice_path, 'rb') as f:
                voice_clone_prompt = pickle.load(f)
            
            _current_voice_clone_prompt = None
            _current_is_new_voice = False
        
        # Generate codec chunks
        audio_codes = list(interface.generate_voice_clone(
            text=text.strip(),
            language=language,
            voice_clone_prompt=voice_clone_prompt,
        ))
        
        # Decode to audio
        wavs, sr = interface.speech_tokenizer.decode([{"audio_codes": audio_codes}])
        
        # Return audio and show save section if new voice
        status_msg = "✅ Generated successfully!" if not _current_is_new_voice else "✅ Generated successfully! You can now save this voice clone if you're happy with it."
        return (
            (sr, wavs[0]), 
            gr.update(visible=_current_is_new_voice), 
            gr.update(visible=_current_is_new_voice),
            status_msg
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error generating speech: {e}")
        _current_is_new_voice = False
        return None, gr.update(visible=False), gr.update(visible=False), f"❌ Error: {str(e)}"


def save_voice_clone(speaker_name: str) -> str:
    """
    Save the current voice clone prompt to a pickle file.
    
    Args:
        speaker_name: Name to save the voice clone as
    
    Returns:
        Success or error message
    """
    global _current_voice_clone_prompt, _current_is_new_voice
    
    if not _current_voice_clone_prompt:
        return "❌ No voice clone to save. Please generate audio with a new voice clone first."
    
    if not speaker_name or not speaker_name.strip():
        return "❌ Please provide a speaker name."
    
    # Sanitize speaker name for filename
    safe_name = "".join(c for c in speaker_name if c.isalnum() or c in (' ', '-', '_')).strip()
    if not safe_name:
        return "❌ Speaker name must contain at least one alphanumeric character."
    
    try:
        # Create a copy for saving (ensure tensors are on CPU)
        voice_clone_prompt = {}
        for key, value in _current_voice_clone_prompt.items():
            if isinstance(value, torch.Tensor):
                voice_clone_prompt[key] = value.cpu()
            elif isinstance(value, list):
                voice_clone_prompt[key] = [emb.cpu() if isinstance(emb, torch.Tensor) else emb for emb in value]
            else:
                voice_clone_prompt[key] = value
        
        # Save as pickle
        voice_path = VOICES_DIR / f"{safe_name}.pkl"
        with open(voice_path, 'wb') as f:
            pickle.dump(voice_clone_prompt, f)
        
        _current_voice_clone_prompt = None
        _current_is_new_voice = False
        
        return f"✅ Voice cloned successfully! Saved as '{safe_name}.pkl'"
    
    except Exception as e:
        return f"❌ Error saving voice clone: {str(e)}"


def update_voice_dropdown():
    """Update the voice dropdown with current saved voices."""
    voices = get_saved_voices()
    choices = ["New Voice Clone"] + voices if voices else ["New Voice Clone"]
    return gr.Dropdown(choices=choices, value=choices[0])


def toggle_new_voice_fields(voice_selection: str):
    """Show/hide fields for new voice clone based on selection."""
    if voice_selection == "New Voice Clone":
        return gr.update(visible=True), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False)


def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="Neto's Voice Clone") as demo:
        gr.Markdown(
            """
            # 🎤 Neto's Voice Clone
            
            Generate speech with existing voices or create new voice clones. Save new voices after you're happy with the generation!
            """
        )
        
        with gr.Row():
            with gr.Column():
                voice_dropdown = gr.Dropdown(
                    label="Select Voice",
                    choices=["New Voice Clone"] + get_saved_voices(),
                    value="New Voice Clone",
                )
                refresh_button = gr.Button("🔄 Refresh Voices", size="sm")
                
                # New voice clone fields (initially visible)
                with gr.Group(visible=True) as new_voice_group:
                    ref_audio_input = gr.Audio(
                        label="Reference Audio (for new voice clone)",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    ref_text_input = gr.Textbox(
                        label="Reference Text (for new voice clone)",
                        placeholder="Enter the text corresponding to the reference audio...",
                        lines=3,
                    )
                
                text_input = gr.Textbox(
                    label="Text to Synthesize",
                    placeholder="Enter the text you want to generate...",
                    lines=5,
                )
                language_input = gr.Dropdown(
                    label="Language",
                    choices=["Auto", "English", "Chinese", "Japanese", "Korean", "French", "German", "Spanish"],
                    value="English",
                )
                generate_button = gr.Button("Generate Speech", variant="primary")
                
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2,
                )
                
                # Save voice clone section (initially hidden)
                with gr.Group(visible=False) as save_voice_group:
                    gr.Markdown("### 💾 Save Voice Clone")
                    gr.Markdown("If you're happy with the generated audio, you can save this voice clone for future use.")
                    save_speaker_name_input = gr.Textbox(
                        label="Speaker Name",
                        placeholder="e.g., John, Anna, Narrator",
                    )
                    save_button = gr.Button("Save Voice Clone", variant="secondary")
                    save_output = gr.Textbox(
                        label="Save Status",
                        interactive=False,
                    )
            
            with gr.Column():
                audio_output = gr.Audio(
                    label="Generated Audio",
                    type="numpy",
                )
        
        # Update voice dropdown when refresh is clicked
        refresh_button.click(
            fn=update_voice_dropdown,
            outputs=voice_dropdown,
        )
        
        # Show/hide new voice fields based on selection
        voice_dropdown.change(
            fn=toggle_new_voice_fields,
            inputs=voice_dropdown,
            outputs=[new_voice_group, ref_audio_input],
        )
        
        # Generate speech
        generate_button.click(
            fn=generate_speech,
            inputs=[text_input, voice_dropdown, language_input, ref_audio_input, ref_text_input],
            outputs=[audio_output, save_voice_group, save_speaker_name_input, status_output],
        )
        
        # Save voice clone
        save_button.click(
            fn=save_voice_clone,
            inputs=save_speaker_name_input,
            outputs=save_output,
        ).then(
            fn=update_voice_dropdown,
            outputs=voice_dropdown,
        )
        
        gr.Markdown(
            """
            ---
            **Note:** This app uses ICL (In-Context Learning) mode only, which requires reference text.
            Voice clones sync with the realtime api
            """
        )
    
    return demo


def main():
    parser = argparse.ArgumentParser(
        description="Gradio app for Voice Clone feature"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to Base model or HuggingFace model ID (e.g., Qwen/Neto's-12Hz-1.7B-Base or /path/to/local/model)",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="0.0.0.0",
        help="Server name (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Server port (default: 7860)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.2,
        help="Fraction of GPU memory to use (default: 0.9). Automatically split between Talker and Predictor.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link",
    )
    
    args = parser.parse_args()
    
    global interface
    
    gpu_mem_util = float(os.environ.get("GPU_MEMORY_UTILIZATION", str(args.gpu_memory_utilization)))
    
    print("=" * 60)
    print("Loading Neto's Base Model for Voice Cloning")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"GPU memory utilization: {gpu_mem_util}")
    
    # Initialize Base model for voice cloning
    if os.path.isdir(args.model_path) or os.path.isfile(args.model_path):
        # Local path - use regular init
        interface = Qwen3TTSInterface(
            model_path=args.model_path,
            enforce_eager=False,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_mem_util,
        )
    else:
        # Likely a HuggingFace model ID, use from_pretrained
        print("  Detected HuggingFace model ID, downloading if needed...")
        interface = Qwen3TTSInterface.from_pretrained(
            pretrained_model_name_or_path=args.model_path,
            enforce_eager=False,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_mem_util,
        )
    
    print("✓ Model loaded successfully")
    print(f"✓ Voices directory: {VOICES_DIR.absolute()}")
    print("=" * 60)
    
    # Create and launch Gradio interface
    demo = create_interface()
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
