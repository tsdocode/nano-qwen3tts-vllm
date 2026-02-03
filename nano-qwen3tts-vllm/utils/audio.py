"""Audio processing utilities for Qwen3-TTS."""
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List
from transformers import AutoConfig, AutoModel, AutoProcessor
from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor

# Import original speech tokenizer
try:
    import sys
    import os
    # Add Qwen3-TTS to path if not already there
    qwen_tts_path = os.path.expanduser('/home/sang/work/Qwen3-TTS')
    if os.path.exists(qwen_tts_path) and qwen_tts_path not in sys.path:
        sys.path.insert(0, qwen_tts_path)
    
    from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer as _Qwen3TTSTokenizer
    # Import to trigger AutoConfig registration for qwen3_tts model types
    try:
        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
        AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

    except:
        pass  # Registration may have already happened
    HAS_SPEECH_TOKENIZER = True
except ImportError as e:
    HAS_SPEECH_TOKENIZER = False
    _Qwen3TTSTokenizer = None
    print(f"Warning: qwen_tts not installed. Speech tokenizer unavailable. Error: {e}")


class SpeechTokenizer:
    """Wrapper for Qwen3-TTS speech tokenizer (codec encoder/decoder).
    
    This handles:
    - Audio encoding: waveform → 16 codebooks
    - Audio decoding: 16 codebooks → waveform
    - Audio I/O: loading and saving audio files
    
    The speech tokenizer uses a neural codec to compress audio into
    16 discrete codebook indices per timestep.
    """
    
    def __init__(self, model_path: str, dtype=torch.bfloat16):
        """Initialize speech tokenizer.
        
        Args:
            model_path: Path to model directory containing speech tokenizer
            dtype: Data type for model (default: bfloat16)
        """
        if not HAS_SPEECH_TOKENIZER:
            raise ImportError(
                "qwen_tts package not found. Install with:\n"
                "pip install git+https://github.com/QwenLM/Qwen3-TTS.git"
            )
        
        print(f"Loading speech tokenizer from {model_path}...")
        # Load using Qwen3TTSTokenizer.from_pretrained
        self.tokenizer = _Qwen3TTSTokenizer.from_pretrained(
            model_path,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            dtype=dtype,
        )
        
        # Get sample rate from config or feature extractor
        if hasattr(self.tokenizer.config, 'sample_rate'):
            self.sample_rate = self.tokenizer.config.sample_rate
        elif hasattr(self.tokenizer.feature_extractor, 'sampling_rate'):
            self.sample_rate = self.tokenizer.feature_extractor.sampling_rate
        else:
            # Default to 12.5kHz for 12hz model
            self.sample_rate = 12500
        
        self.device = self.tokenizer.device
        
        print(f"Speech tokenizer loaded: sample_rate={self.sample_rate}Hz, device={self.device}")
    
    @torch.inference_mode()
    def encode(self, audio: Union[torch.Tensor, np.ndarray], sr: int = None) -> torch.Tensor:
        """Encode audio waveform to codec IDs.
        
        Args:
            audio: Audio waveform tensor [batch, samples] or [samples], or numpy array
            sr: Sample rate of input audio (required for numpy input)
            
        Returns:
            Codec IDs tensor [batch, num_codebooks=16, time]
        """
        # Convert torch to numpy if needed
        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
            if audio.dim() == 1:
                audio_np = audio_np[np.newaxis, :]  # Add batch dim
        else:
            audio_np = audio
            if audio_np.ndim == 1:
                audio_np = audio_np[np.newaxis, :]
        
        # Use sr if not provided
        if sr is None:
            sr = self.sample_rate
        
        # Encode using Qwen3TTSTokenizer API
        # It expects numpy arrays or list of numpy arrays
        result = self.tokenizer.encode(audio_np, sr=sr)
        
        # Extract codec IDs from result
        # Result has 'audio_codes' key which is a list of tensors [time, 16]
        if hasattr(result, 'audio_codes'):
            audio_codes_list = result.audio_codes
        elif isinstance(result, dict) and 'audio_codes' in result:
            audio_codes_list = result['audio_codes']
        else:
            raise ValueError(f"Unexpected tokenizer output format: {type(result)}")
        
        # Stack to [batch, 16, time] format
        # audio_codes_list is list of [time, 16], we need [batch, 16, time]
        codec_ids = torch.stack([codes.transpose(0, 1) for codes in audio_codes_list])
        
        return codec_ids
    
    @torch.inference_mode()
    def decode(self, audio_codes: Union[List[List[int]], torch.Tensor, List[dict]]) -> Tuple[List[np.ndarray], int]:
        """Decode codec IDs to audio waveform.
        
        Args:
            audio_codes: Can be:
                - List of codebook_id chunks: [[c0_book0, c0_book1, ..., c0_book15], [c1_book0, ...], ...]
                - Tensor [batch, num_codebooks=16, time]
                - List of dicts: [{"audio_codes": codes}] where codes is list or tensor
            
        Returns:
            Tuple of (audio_list, sample_rate)
            - audio_list: List of numpy arrays [samples]
            - sample_rate: Sample rate of output audio
        """
        # Handle different input formats
        if isinstance(audio_codes, list):
            if len(audio_codes) > 0 and isinstance(audio_codes[0], dict):
                # Format: [{"audio_codes": ...}]
                audio_codes = audio_codes[0]["audio_codes"]
            
            # Convert list of chunks to tensor
            if isinstance(audio_codes, list) and len(audio_codes) > 0:
                # List of codec chunks [[c0_b0, ..., c0_b15], [c1_b0, ..., c1_b15], ...]
                # Convert to tensor [time, 16]
                codec_tensor = torch.tensor(audio_codes, dtype=torch.long)  # [time, 16]
                # Reshape to [1, 16, time] for batch processing
                codec_ids = codec_tensor.transpose(0, 1).unsqueeze(0)  # [1, 16, time]
            else:
                codec_ids = audio_codes
        else:
            codec_ids = audio_codes
        
        # codec_ids should now be [batch, 16, time]
        batch_size = codec_ids.shape[0]
        
        # Convert from [batch, 16, time] to list of dicts with 'audio_codes' key
        inputs = []
        for i in range(batch_size):
            # codec_ids[i] is [16, time], transpose to [time, 16]
            codes = codec_ids[i]  # [16, time]
            if codes.dim() == 2:
                codes = codes.transpose(0, 1)  # [time, 16]
            # Tokenizer expects dict with 'audio_codes' key
            inputs.append({'audio_codes': codes})
        
        # Decode
        audio_list, sr = self.tokenizer.decode(inputs)
        
        return audio_list, sr
    
    def load_audio(
        self, 
        audio_path: Union[str, Path],
        target_sr: int = None,
    ) -> torch.Tensor:
        """Load audio file and resample if needed.
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (default: self.sample_rate)
            
        Returns:
            Audio tensor [1, samples]
        """
        if target_sr is None:
            target_sr = self.sample_rate
        
        # Load audio
        waveform, sr = torchaudio.load(str(audio_path))
        
        # Resample if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return waveform
    
    def save_audio(
        self, 
        audio: torch.Tensor, 
        output_path: Union[str, Path],
        sample_rate: int = None,
    ):
        """Save audio waveform to file.
        
        Args:
            audio: Audio tensor [1, samples] or [samples]
            output_path: Path to save audio file
            sample_rate: Sample rate (default: self.sample_rate)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # Ensure proper shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Move to CPU
        audio = audio.cpu()
        
        # Save
        torchaudio.save(str(output_path), audio, sample_rate)
    
    def to_numpy(self, audio: torch.Tensor) -> np.ndarray:
        """Convert audio tensor to numpy array.
        
        Args:
            audio: Audio tensor [1, samples] or [samples]
            
        Returns:
            Numpy array [samples] (mono audio)
        """
        if audio.dim() > 1:
            audio = audio.squeeze(0)
        return audio.cpu().numpy()


def load_reference_audio(
    audio_path: str, 
    model_path: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience function to load and encode reference audio.
    
    Args:
        audio_path: Path to reference audio file
        model_path: Path to model directory
        
    Returns:
        Tuple of (audio_waveform, codec_ids)
    """
    tokenizer = SpeechTokenizer(model_path)
    audio = tokenizer.load_audio(audio_path)
    codec_ids = tokenizer.encode(audio)
    return audio, codec_ids


def decode_and_save(
    codec_ids: torch.Tensor,
    output_path: str,
    model_path: str,
):
    """Convenience function to decode and save audio.
    
    Args:
        codec_ids: Codec IDs tensor [batch, 16, time]
        output_path: Path to save audio file
        model_path: Path to model directory
    """
    tokenizer = SpeechTokenizer(model_path)
    audio = tokenizer.decode(codec_ids)
    tokenizer.save_audio(audio[0], output_path)
