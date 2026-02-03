"""
Simple Gradio app for Qwen3-TTS Voice Design feature.

Usage:
    python gradio_voice_design.py --model-path Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
    
    # Or with local model
    python gradio_voice_design.py --model-path /path/to/local/model
"""

import argparse
import os
import numpy as np
import gradio as gr
from nano_qwen3tts_vllm.interface import Qwen3TTSInterface


# Global model instance
interface = None


def generate_speech(text: str, instruct: str, language: str) -> tuple:
    """
    Generate speech from text and voice instruction.
    
    Args:
        text: Text to synthesize
        instruct: Voice design instruction (e.g., "Male, 30 years old, deep voice")
        language: Language selection
    
    Returns:
        Tuple of (sample_rate, audio_array) for Gradio Audio component
    """
    if not text or not instruct:
        return None
    
    try:
        # Generate codec chunks
        audio_codes = list(interface.generate_voice_design(
            text=text,
            instruct=instruct,
            language=language,
        ))
        
        # Decode to audio
        wavs, sr = interface.speech_tokenizer.decode([{"audio_codes": audio_codes}])
        
        # Return in format expected by Gradio Audio: (sample_rate, numpy_array)
        return (sr, wavs[0])
    
    except Exception as e:
        print(f"Error generating speech: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Gradio app for Qwen3-TTS Voice Design")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        help="Path to Voice Design model or HuggingFace model ID"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=17861,
        help="Port to run the server on"
    )
    parser.add_argument(
        "--cuda-device",
        type=str,
        default="0",
        help="CUDA device to use (e.g., '0', '1', '2')"
    )
    
    args = parser.parse_args()
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    
    # Initialize model
    global interface
    print(f"Loading Voice Design model from: {args.model_path}")
    
    if os.path.isdir(args.model_path) or os.path.isfile(args.model_path):
        # Local path
        interface = Qwen3TTSInterface(
            model_path=args.model_path,
            enforce_eager=False,
            tensor_parallel_size=1,
        )
    else:
        # HuggingFace model ID
        interface = Qwen3TTSInterface.from_pretrained(
            pretrained_model_name_or_path=args.model_path,
            enforce_eager=False,
            tensor_parallel_size=1,
        )
    
    print("✓ Model loaded successfully")
    
    # Define example voice instructions
    example_instructions = [
        ["Hello, this is a demonstration of the voice design system.", 
         "Male, 35 years old, deep and authoritative voice", 
         "English"],
        ["Welcome to our text-to-speech service!", 
         "Female, 25 years old, cheerful and energetic", 
         "English"],
        ["Thank you for using our system.", 
         "Male, 40 years old, calm and professional", 
         "English"],
        ["I'm excited to show you what I can do!", 
         "Female, 20 years old, enthusiastic and young", 
         "English"],
    ]
    
    # Create Gradio interface
    with gr.Blocks(title="Qwen3-TTS Voice Design") as demo:
        gr.Markdown("# 🎙️ Qwen3-TTS Voice Design")
        gr.Markdown(
            "Generate speech with custom voice characteristics using natural language instructions. "
            "Describe the desired voice (gender, age, tone, etc.) and the system will synthesize speech matching your description."
        )
        
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Text to Synthesize",
                    placeholder="Enter the text you want to convert to speech...",
                    lines=3,
                    value="Hello, this is a test of the voice design system."
                )
                
                instruct_input = gr.Textbox(
                    label="Voice Design Instruction",
                    placeholder="Describe the desired voice (e.g., 'Male, 30 years old, deep voice')",
                    lines=2,
                    value="Male, 35 years old, deep and authoritative voice"
                )
                
                language_input = gr.Dropdown(
                    label="Language",
                    choices=["Auto", "English", "Chinese", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"],
                    value="English"
                )
                
                generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
            
            with gr.Column():
                audio_output = gr.Audio(
                    label="Generated Speech",
                    type="numpy"
                )
        
        # Examples section
        gr.Markdown("### 📝 Example Instructions")
        gr.Examples(
            examples=example_instructions,
            inputs=[text_input, instruct_input, language_input],
            outputs=audio_output,
            fn=generate_speech,
            cache_examples=False
        )
        
        # Tips section
        with gr.Accordion("💡 Tips for Voice Instructions", open=False):
            gr.Markdown("""
            ### How to write effective voice instructions:
            
            **Basic Format:**
            - Gender: Male / Female
            - Age: e.g., "25 years old", "middle-aged", "elderly"
            - Tone: deep, soft, cheerful, calm, energetic, professional, etc.
            
            **Good Examples:**
            - `Male, 30 years old, deep and authoritative voice`
            - `Female, 25 years old, soft and gentle voice`
            - `Male, 40 years old, calm and professional`
            - `Female, 20 years old, excited and enthusiastic`
            
            **Tips:**
            - Be specific about gender and age
            - Include 1-2 descriptive adjectives for tone/style
            - Keep instructions concise (1-2 sentences)
            - Experiment with different combinations!
            """)
        
        # Connect button to function
        generate_btn.click(
            fn=generate_speech,
            inputs=[text_input, instruct_input, language_input],
            outputs=audio_output
        )
    
    # Launch the app
    print(f"\n🚀 Launching Gradio app on port {args.port}")
    print(f"   Access locally at: http://localhost:{args.port}")
    if args.share:
        print(f"   Public link will be generated...")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
