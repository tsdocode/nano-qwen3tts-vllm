import asyncio
import base64
import io
import os
import queue
import threading
import uuid
import torch
import torch.multiprocessing as mp
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf
import time
import torch
from librosa.filters import mel as librosa_mel_fn

from nano_qwen3tts_vllm.utils.prompt import prepare_custom_voice_prompt, _ensure_list, _tokenize_texts
from nano_qwen3tts_vllm.processor import Qwen3TTSProcessor
from nano_qwen3tts_vllm.utils.generation import prepare_inputs, generate_speaker_prompt, generate_icl_prompt
from nano_qwen3tts_vllm.llm import TalkerLLM, PredictorLLM
from nano_qwen3tts_vllm.sampling_params import SamplingParams
from nano_qwen3tts_vllm.engine.engine_core import EngineRequest

try:
    from huggingface_hub import snapshot_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False
    snapshot_download = None

try:
    from nano_qwen3tts_vllm.utils.audio import SpeechTokenizer
    HAS_SPEECH_TOKENIZER = True
except ImportError:
    HAS_SPEECH_TOKENIZER = False
    SpeechTokenizer = None


torch.manual_seed(42)
# Use Qwen3-TTS processor when available so tokenization matches Qwen3TTSModel.generate_custom_voice exactly
def _get_processor(model_path: str):
    try:
        from qwen_tts.core.models import Qwen3TTSProcessor as Qwen3TTSProcessorHF
        return Qwen3TTSProcessorHF.from_pretrained(model_path, fix_mistral_regex=True)
    except ImportError:
        return Qwen3TTSProcessor.from_pretrained(model_path, fix_mistral_regex=True)


class Qwen3TTSInterface:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[str] = None,
        revision: Optional[str] = None,
        enforce_eager: bool = False,
        tensor_parallel_size: int = 1,
        zmq_bridge=None,
    ):
        """Load Qwen3TTSInterface from HuggingFace model repository or local path.
        
        This method automatically downloads the model from HuggingFace if it's not
        available locally. If the path is already a local directory, it will use
        that directly.
        
        Args:
            pretrained_model_name_or_path: HuggingFace model ID (e.g., "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
                or local directory path.
            cache_dir: Directory to cache downloaded models. If None, uses HuggingFace default.
            force_download: Whether to force re-download even if model exists in cache.
            local_files_only: If True, only use local files and don't attempt to download.
            token: HuggingFace token for private models. Can also be set via HF_TOKEN env var.
            revision: Git revision/branch/tag to download. Defaults to "main".
            enforce_eager: Whether to enforce eager mode (disable CUDA graphs).
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            zmq_bridge: Optional ZMQ bridge for async generation.
        
        Returns:
            Qwen3TTSInterface instance.
        
        Example:
            >>> # Download from HuggingFace
            >>> interface = Qwen3TTSInterface.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
            
            >>> # Use local path
            >>> interface = Qwen3TTSInterface.from_pretrained("/path/to/local/model")
            
            >>> # With custom cache directory
            >>> interface = Qwen3TTSInterface.from_pretrained(
            ...     "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            ...     cache_dir="/custom/cache/dir"
            ... )
        """
        # Check if it's already a local directory
        # Also check if it's an absolute path or exists as a file/directory
        if os.path.isdir(pretrained_model_name_or_path) or os.path.isfile(pretrained_model_name_or_path):
            model_path = pretrained_model_name_or_path
            if os.path.isfile(model_path):
                # If it's a file, use the parent directory
                model_path = os.path.dirname(model_path)
            print(f"Using local model directory: {model_path}")
        elif os.path.isabs(pretrained_model_name_or_path):
            # Absolute path that doesn't exist - might be invalid
            raise ValueError(
                f"Model path '{pretrained_model_name_or_path}' does not exist. "
                "Please provide a valid local path or HuggingFace model ID."
            )
        else:
            # Download from HuggingFace
            if not HAS_HF_HUB:
                raise ImportError(
                    "huggingface_hub is required for downloading models. "
                    "Install it with: pip install huggingface_hub"
                )
            
            if local_files_only:
                # Try to find in cache
                from huggingface_hub import try_to_load_from_cache
                model_path = try_to_load_from_cache(
                    pretrained_model_name_or_path,
                    revision=revision or "main",
                    cache_dir=cache_dir,
                )
                if model_path is None:
                    raise ValueError(
                        f"Model {pretrained_model_name_or_path} not found in cache and "
                        "local_files_only=True. Please download it first or set local_files_only=False."
                    )
                if not os.path.isdir(model_path):
                    # It's a file, get the parent directory
                    model_path = os.path.dirname(model_path)
            else:
                print(f"Downloading model from HuggingFace: {pretrained_model_name_or_path}")
                if revision:
                    print(f"  Revision: {revision}")
                if cache_dir:
                    print(f"  Cache directory: {cache_dir}")
                
                model_path = snapshot_download(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                    token=token or os.getenv("HF_TOKEN"),
                    revision=revision or "main",
                )
                print(f"✓ Model downloaded to: {model_path}")
        
        # Initialize interface with the model path
        return cls(
            model_path=model_path,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tensor_parallel_size,
            zmq_bridge=zmq_bridge,
        )
    
    def __init__(self, model_path: str, enforce_eager: bool = False, tensor_parallel_size: int = 1, zmq_bridge=None):
        self.model_path = model_path
        self.enforce_eager = enforce_eager
        self.tensor_parallel_size = tensor_parallel_size
        self.zmq_bridge = zmq_bridge
        
        # For sync mode (non-ZMQ): initialize engines in main process
        if zmq_bridge is None:
            self.talker_llm = TalkerLLM(model_path, enforce_eager=enforce_eager, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.3)
            self.predictor_llm = PredictorLLM(model_path, enforce_eager=enforce_eager, tensor_parallel_size=tensor_parallel_size)
            self.processor = _get_processor(model_path)
            self.model_config = self.talker_llm.model_runner.full_config
            
            self.text_embedding = self.talker_llm.model_runner.model.get_text_embeddings()
            self.input_embedding = self.talker_llm.model_runner.model.get_input_embeddings()
            self.text_projection = self.talker_llm.model_runner.model.text_projection
            
            self.predictor_input_embeddings = self.predictor_llm.model_runner.model.model.codec_embedding
        else:
            # For async/ZMQ mode: engines run in separate processes. Load only embedding
            # layers in the main process to avoid loading full talker+predictor (would cause
            # CUDA OOM when engine processes also load their models).
            from nano_qwen3tts_vllm.utils.embedding_loader import load_embeddings_only
            _device = "cuda" if torch.cuda.is_available() else "cpu"
            (
                self.model_config,
                self.text_embedding,
                self.input_embedding,
                self.text_projection,
                self.predictor_input_embeddings,
            ) = load_embeddings_only(model_path, device=_device)
            self.processor = _get_processor(model_path)

            # Create multiprocessing queues for sending requests to engine processes
            mp_ctx = mp.get_context('spawn')
            self._talker_request_queue = mp_ctx.Queue()
            self._predictor_request_queue = mp_ctx.Queue()
            
            # Start engine core processes
            # Use separate ZMQ ports for talker and predictor to avoid conflicts
            from nano_qwen3tts_vllm.engine.engine_core import talker_engine_core_loop, predictor_engine_core_loop
            
            # Parse base address and create separate bind addresses for each engine.
            # Main process zmq_bridge may already be bound to base_addr (e.g. 9555),
            # so use base_port+1 and base_port+2 for talker and predictor to avoid conflict.
            base_addr = zmq_bridge.bind_address
            if ":" in base_addr:
                proto_host, port = base_addr.rsplit(":", 1)
                base_port = int(port)
                talker_addr = f"{proto_host}:{base_port + 1}"
                predictor_addr = f"{proto_host}:{base_port + 2}"
            else:
                talker_addr = base_addr
                predictor_addr = base_addr
            
            # Two processes share one GPU: give each 50% budget so both get full KV cache (e.g. 24GB → 12GB each)
            process_gpu_fraction = 0.3
            gpu_util = 0.9
            self._talker_process = mp_ctx.Process(
                target=talker_engine_core_loop,
                args=(
                    self._talker_request_queue,
                    talker_addr,
                    model_path,
                ),
                kwargs={
                    'enforce_eager': enforce_eager,
                    'tensor_parallel_size': tensor_parallel_size,
                    'gpu_memory_utilization': gpu_util,
                    'process_gpu_memory_fraction': process_gpu_fraction,
                    'distributed_port': 2433,
                },
                daemon=False,
            )
            
            self._predictor_process = mp_ctx.Process(
                target=predictor_engine_core_loop,
                args=(
                    self._predictor_request_queue,
                    predictor_addr,
                    model_path,
                ),
                kwargs={
                    'enforce_eager': enforce_eager,
                    'tensor_parallel_size': tensor_parallel_size,
                    'gpu_memory_utilization': gpu_util,
                    'process_gpu_memory_fraction': process_gpu_fraction,
                    'distributed_port': 2434,
                },
                daemon=False,
            )
            
            # Store addresses for dispatcher
            self._talker_zmq_address = talker_addr
            self._predictor_zmq_address = predictor_addr
            
            self._talker_process.start()
            self._predictor_process.start()
            print(f"[Interface] Started talker process PID={self._talker_process.pid}")
            print(f"[Interface] Started predictor process PID={self._predictor_process.pid}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize speech tokenizer and speaker encoder if available
        self.speech_tokenizer = None
        self.speaker_encoder = None
        self._init_speech_components()

        # ZMQ path: asyncio queues for receiving outputs from ZMQ dispatcher
        self._request_queues: dict[str, asyncio.Queue] = {}
        self._queues_lock = asyncio.Lock()
        self._zmq_tasks: list[asyncio.Task] = []
        self._zmq_inbox: queue.Queue | None = None
        self._zmq_tasks_started = False
        # Serialize request prep (GPU work) so event loop can run while another request prepares.
        self._prep_lock = threading.Lock()
    
    def _init_speech_components(self):
        """Initialize speech tokenizer and speaker encoder from model if available."""
        try:
            # Try to load speech tokenizer
            if HAS_SPEECH_TOKENIZER:
                self.speech_tokenizer = SpeechTokenizer("Qwen/Qwen3-TTS-Tokenizer-12Hz", dtype=torch.bfloat16)
            
            # Try to load speaker encoder from model
            # Check if speaker_encoder exists in the model
            try:
                from qwen_tts.core.models import Qwen3TTSForConditionalGeneration
                # Try to load just to check if speaker encoder exists
                # We'll load it lazily when needed
                self._speaker_encoder_available = True
            except:
                self._speaker_encoder_available = False
        except Exception as e:
            print(f"Warning: Could not initialize speech components: {e}")
            self.speech_tokenizer = None
            self._speaker_encoder_available = False
    
    def _load_speaker_encoder(self):
        """Lazily load speaker encoder from model."""
        if self.speaker_encoder is not None:
            return self.speaker_encoder
        
        if not self._speaker_encoder_available:
            raise RuntimeError("Speaker encoder not available for this model")
        
        try:
            from qwen_tts.core.models import Qwen3TTSForConditionalGeneration
            # Load model just to get speaker encoder
            # This is a bit inefficient, but necessary for now
            temp_model = Qwen3TTSForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map="cpu",  # Load to CPU first
            )
            if hasattr(temp_model, 'speaker_encoder') and temp_model.speaker_encoder is not None:
                self.speaker_encoder = temp_model.speaker_encoder.to(self.device)
                # Get dtype from model
                if hasattr(temp_model, 'dtype'):
                    self.speaker_encoder = self.speaker_encoder.to(temp_model.dtype)
                return self.speaker_encoder
            else:
                self._speaker_encoder_available = False
                raise RuntimeError("Model does not have speaker_encoder")
        except Exception as e:
            print(f"Warning: Could not load speaker encoder: {e}")
            self._speaker_encoder_available = False
            raise
    
    def _build_ref_text(self, text: str) -> str:
        """Build reference text format for ICL mode.
        
        Args:
            text: Reference text string.
        
        Returns:
            Formatted reference text string.
        """
        return f"<|im_start|>assistant\n{text}<|im_end|>\n"
    
    def _is_url(self, x: str) -> bool:
        """Check if string is a URL."""
        try:
            result = urlparse(x)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _is_probably_base64(self, x: str) -> bool:
        """Check if string is probably base64 encoded."""
        try:
            if isinstance(x, str) and len(x) > 100:
                base64.b64decode(x.split(",")[-1] if "," in x else x)
                return True
        except:
            pass
        return False
    
    def _decode_base64_to_wav_bytes(self, b64: str) -> bytes:
        """Decode base64 string to WAV bytes."""
        if "," in b64 and b64.strip().startswith("data:"):
            b64 = b64.split(",", 1)[1]
        return base64.b64decode(b64)
    
    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        """Load audio from path/URL/base64 to numpy array."""
        if self._is_url(x):
            with urllib.request.urlopen(x) as resp:
                audio_bytes = resp.read()
            with io.BytesIO(audio_bytes) as f:
                audio, sr = sf.read(f, dtype="float32", always_2d=False)
        elif self._is_probably_base64(x):
            wav_bytes = self._decode_base64_to_wav_bytes(x)
            with io.BytesIO(wav_bytes) as f:
                audio, sr = sf.read(f, dtype="float32", always_2d=False)
        else:
            audio, sr = librosa.load(x, sr=None, mono=True)
        
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        
        return audio.astype(np.float32), int(sr)
    
    def _normalize_audio_inputs(self, audios: Union[Any, List[Any]]) -> List[Tuple[np.ndarray, int]]:
        """Normalize audio inputs into list of (waveform, sr) tuples.
        
        Supports:
        - str: wav path / URL / base64 audio string
        - (np.ndarray, sr): waveform + sampling rate tuple
        - list of the above
        """
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]
        
        normalized = []
        for item in items:
            if isinstance(item, str):
                wav, sr = self._load_audio_to_np(item)
                normalized.append((wav, sr))
            elif isinstance(item, tuple) and len(item) == 2:
                wav, sr = item
                if isinstance(wav, torch.Tensor):
                    wav = wav.cpu().numpy()
                if wav.ndim > 1:
                    wav = np.mean(wav, axis=-1)
                normalized.append((wav.astype(np.float32), int(sr)))
            elif isinstance(item, np.ndarray):
                raise ValueError("numpy array provided without sampling rate. Use (np.ndarray, sr) tuple.")
            else:
                raise ValueError(f"Unsupported audio input type: {type(item)}")
        
        return normalized
    
    @torch.inference_mode()
    def extract_speaker_embedding(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """Extract speaker embedding from audio.
        
        Args:
            audio: Audio waveform as numpy array.
            sr: Sample rate (must be 24000).
        
        Returns:
            Speaker embedding tensor [D].
        """
        assert sr == 24000, "Only support 24kHz audio"
        
        # Load speaker encoder if not already loaded
        speaker_encoder = self._load_speaker_encoder()
        
        # Compute mel spectrogram
        mels = self._mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000
        ).transpose(1, 2)
        
        # Get dtype from speaker encoder
        dtype = next(speaker_encoder.parameters()).dtype
        speaker_embedding = speaker_encoder(mels.to(self.device).to(dtype))[0]
        return speaker_embedding
    
    def _mel_spectrogram(
        self,
        y: torch.Tensor,
        n_fft: int,
        num_mels: int,
        sampling_rate: int,
        hop_size: int,
        win_size: int,
        fmin: int,
        fmax: int = None,
        center: bool = False,
    ) -> torch.Tensor:
        """Calculate mel spectrogram (from Qwen3-TTS)."""
        device = y.device
        
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        
        mel_basis = torch.from_numpy(mel).float().to(device)
        hann_window = torch.hann_window(win_size).to(device)
        
        padding = (n_fft - hop_size) // 2
        y = torch.nn.functional.pad(
            y.unsqueeze(1), (padding, padding), mode="reflect"
        ).squeeze(1)
        
        spec = torch.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        
        # Compute magnitude: sqrt(real^2 + imag^2)
        spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
        
        # Apply mel filterbank: mel_basis @ spec
        # spec shape: [batch, freq_bins, time] or [freq_bins, time]
        # mel_basis shape: [num_mels, freq_bins]
        # Result: [batch, num_mels, time] or [num_mels, time]
        mel_spec = torch.matmul(mel_basis, spec)  # matmul handles batch dimension correctly
        
        return mel_spec
    
    def _codebook_ids_to_audio(self, codebook_ids_list: List[List[int]]) -> Tuple[List[np.ndarray], int]:
        """Convert codebook_ids chunks to audio format.
        
        Args:
            codebook_ids_list: List of codebook_id chunks, each chunk is [codebook0, codebook1, ..., codebook15].
        
        Returns:
            Tuple of (wavs: List[np.ndarray], sr: int).
        """
        if self.speech_tokenizer is None:
            raise RuntimeError("speech_tokenizer not available. Cannot decode audio.")
        
        # Convert list of chunks to tensor format [batch, 16, time]
        # Each chunk is [codebook0, codebook1, ..., codebook15]
        # Stack chunks along time dimension
        if not codebook_ids_list:
            return [], self.speech_tokenizer.sample_rate
        
        # Convert to tensor: [time, 16]
        codebook_tensor = torch.tensor(codebook_ids_list, dtype=torch.long)  # [time, 16]
        
        # Transpose to [16, time] and add batch dim -> [1, 16, time]
        codebook_tensor = codebook_tensor.transpose(0, 1).unsqueeze(0)  # [1, 16, time]
        
        # Decode using speech tokenizer
        wavs, sr = self.speech_tokenizer.decode(codebook_tensor)
        return wavs, sr
    
    def create_voice_clone_prompt(
        self,
        ref_audio: Any,
        ref_text: Optional[str] = None,
        x_vector_only_mode: bool = False,
    ) -> Dict[str, Any]:
        """Build voice-clone prompt from reference audio (and optionally reference text).
        
        Args:
            ref_audio: Reference audio. Can be:
                - str: wav path / URL / base64
                - (np.ndarray, sr): waveform + sampling rate tuple
            ref_text: Reference transcript. Required when x_vector_only_mode=False (ICL mode).
            x_vector_only_mode: Whether to use speaker embedding only. If False, ICL mode will be used.
        
        Returns:
            voice_clone_prompt dict with keys: ref_code, ref_spk_embedding, x_vector_only_mode, icl_mode, ref_text.
        """
        if self.speech_tokenizer is None:
            raise RuntimeError("speech_tokenizer not available. Cannot create voice clone prompt.")
        
        if not x_vector_only_mode:
            if ref_text is None or ref_text == "":
                raise ValueError("ref_text is required when x_vector_only_mode=False (ICL mode).")
        
        # Normalize audio input
        wav, sr = self._normalize_audio_inputs([ref_audio])[0]
        
        # Encode audio to codes
        enc = self.speech_tokenizer.tokenizer.encode(wav, sr=sr)
        # enc.audio_codes[0] is [time, 16]
        ref_code = enc.audio_codes[0].cpu()
        
        # Resample to 24kHz for speaker encoder if needed
        wav_resample = wav
        if sr != 24000:
            wav_resample = librosa.resample(
                y=wav_resample.astype(np.float32),
                orig_sr=int(sr),
                target_sr=24000
            )
        
        # Extract speaker embedding
        spk_emb = self.extract_speaker_embedding(audio=wav_resample, sr=24000)
        
        return {
            "ref_code": None if x_vector_only_mode else ref_code,
            "ref_spk_embedding": spk_emb,
            "x_vector_only_mode": bool(x_vector_only_mode),
            "icl_mode": bool(not x_vector_only_mode),
            "ref_text": ref_text,  # Store ref_text for later use in generate_voice_clone
        }
    
    def generate_voice_clone(
        self,
        text: str,
        language: str = None,
        ref_audio: Optional[Any] = None,
        ref_text: Optional[str] = None,
        x_vector_only_mode: bool = False,
        voice_clone_prompt: Optional[Dict[str, Any]] = None,
        non_streaming_mode: bool = True,
    ):
        """Generate speech using voice clone (yields codec chunks).
        
        This is a generator that yields codebook_id chunks. Use SpeechTokenizer to decode.
        
        Args:
            text: Text to synthesize (single string only, no batch support).
            language: Language for the sample (default: "Auto").
            ref_audio: Reference audio for prompt building. Required if voice_clone_prompt is not provided.
            ref_text: Reference transcript used for ICL mode (required when x_vector_only_mode=False).
            x_vector_only_mode: If True, only speaker embedding is used (ignores ref_text/ref_code).
            voice_clone_prompt: Pre-built voice clone prompt dict from create_voice_clone_prompt.
            non_streaming_mode: Using non-streaming text input.
        
        Yields:
            Codebook ID chunks (List[int]). Use SpeechTokenizer.decode() to convert to audio.
            
        Example:
            chunks = list(interface.generate_voice_clone(text="Hello", voice_clone_prompt=prompt))
            wavs, sr = interface.speech_tokenizer.decode([{"audio_codes": chunks}])
        """
        if self.zmq_bridge is not None:
            raise RuntimeError("generate_voice_clone does not support ZMQ bridge. Use sync mode.")
        
        if language is None:
            language = "Auto"
        
        # Build or use voice_clone_prompt
        if voice_clone_prompt is None:
            if ref_audio is None:
                raise ValueError("Either `voice_clone_prompt` or `ref_audio` must be provided.")
            voice_clone_prompt = self.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode
            )
            ref_text_for_ids = ref_text
        else:
            # When voice_clone_prompt is provided, check if ICL mode is enabled
            # If so, use ref_text from prompt or provided ref_text parameter
            icl_mode_enabled = voice_clone_prompt.get("icl_mode", False)
            if icl_mode_enabled:
                # Prefer ref_text from parameter, fallback to stored ref_text in prompt
                prompt_ref_text = voice_clone_prompt.get("ref_text")
                if ref_text is not None:
                    ref_text_for_ids = ref_text
                elif prompt_ref_text is not None:
                    ref_text_for_ids = prompt_ref_text
                else:
                    raise ValueError(
                        "ICL mode is enabled in voice_clone_prompt but ref_text is not available. "
                        "Please provide ref_text when creating voice_clone_prompt or when calling generate_voice_clone."
                    )
            else:
                ref_text_for_ids = None
        
        # Tokenize text
        input_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = _tokenize_texts([input_text], self.processor, self.device)
        
        # Tokenize ref_text if provided (for ICL mode)
        ref_ids = None
        if ref_text_for_ids is not None and ref_text_for_ids != "":
            ref_tok = _tokenize_texts([self._build_ref_text(ref_text_for_ids)], self.processor, self.device)[0]
            ref_ids = [ref_tok]
        
        # Prepare voice_clone_prompt as lists (for compatibility with prepare_inputs which expects batch format)
        voice_clone_prompt_lists = {
            "ref_code": [voice_clone_prompt["ref_code"]],
            "ref_spk_embedding": [voice_clone_prompt["ref_spk_embedding"]],
            "x_vector_only_mode": [voice_clone_prompt["x_vector_only_mode"]],
            "icl_mode": [voice_clone_prompt["icl_mode"]],
        }
        
        # Prepare generate_speaker_prompt_fn and generate_icl_prompt_fn
        def generate_speaker_prompt_fn(prompt):
            return generate_speaker_prompt(prompt, self.device)
        
        def generate_icl_prompt_fn(text_id, ref_id, ref_code, tts_pad_embed, tts_eos_embed, non_streaming_mode):
            return generate_icl_prompt(
                text_id=text_id,
                ref_id=ref_id,
                ref_code=ref_code,
                tts_pad_embed=tts_pad_embed,
                tts_eos_embed=tts_eos_embed,
                non_streaming_mode=non_streaming_mode,
                config=self.model_config,
                text_embedding=self.text_embedding,
                input_embedding=self.input_embedding,
                text_projection=self.text_projection,
                code_predictor_embeddings=self.predictor_input_embeddings,
                device=self.device,
            )
        
        # Prepare inputs
        talker_input_embeds, trailing_text_hiddens, tts_pad_embed, talker_attention_mask = prepare_inputs(
            config=self.model_config,
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=voice_clone_prompt_lists,
            languages=[language],
            non_streaming_mode=non_streaming_mode,
            text_embedding=self.text_embedding,
            input_embedding=self.input_embedding,
            text_projection=self.text_projection,
            device=self.device,
            generate_speaker_prompt_fn=generate_speaker_prompt_fn,
            generate_icl_prompt_fn=generate_icl_prompt_fn,
        )
        
        # Generate and yield codebook_id chunks
        yield from self._generate_caller_driven(
            talker_input_embeds, trailing_text_hiddens, tts_pad_embed,
            str(uuid.uuid4()),
            SamplingParams(temperature=1.0, max_tokens=1),
            SamplingParams(temperature=0.9, max_tokens=17),
        )
    
    def generate_voice_design(
        self,
        text: str,
        instruct: str,
        language: str = None,
        non_streaming_mode: bool = True,
    ):
        """Generate speech with voice design (yields codec chunks).
        
        This is a generator that yields codebook_id chunks. Use SpeechTokenizer to decode.
        
        Args:
            text: Text to synthesize (single string only, no batch support).
            instruct: Instruction describing desired voice/style.
            language: Language for the sample (default: "Auto").
            non_streaming_mode: Using non-streaming text input.
        
        Yields:
            Codebook ID chunks (List[int]). Use SpeechTokenizer.decode() to convert to audio.
            
        Example:
            chunks = list(interface.generate_voice_design(text="Hello", instruct="cheerful voice"))
            wavs, sr = interface.speech_tokenizer.decode([{"audio_codes": chunks}])
        """
        if self.zmq_bridge is not None:
            raise RuntimeError("generate_voice_design does not support ZMQ bridge. Use sync mode.")
        
        if language is None:
            language = "Auto"
        
        # Prepare prompts
        input_ids, instruct_ids, speakers, languages = prepare_custom_voice_prompt(
            text=[text],
            speaker=[""],  # Voice design doesn't use speakers
            language=[language],
            instruct=[instruct],
            processor=self.processor,
            device=self.device,
        )
        
        # Prepare inputs
        talker_input_embeds, trailing_text_hiddens, tts_pad_embed, talker_attention_mask = prepare_inputs(
            config=self.model_config,
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            speakers=speakers,
            non_streaming_mode=non_streaming_mode,
            text_embedding=self.text_embedding,
            input_embedding=self.input_embedding,
            text_projection=self.text_projection,
            device=self.device,
        )
        
        # Generate and yield codebook_id chunks
        yield from self._generate_caller_driven(
            talker_input_embeds, trailing_text_hiddens, tts_pad_embed,
            str(uuid.uuid4()),
            SamplingParams(temperature=1.0, max_tokens=1),
            SamplingParams(temperature=0.9, max_tokens=17),
        )
    
    async def start_zmq_tasks(self) -> None:
        """Start the ZMQ dispatcher (thread + asyncio task). Engine loops run in separate processes. Call once before generate_async when zmq_bridge is set."""
        if self.zmq_bridge is None:
            raise RuntimeError("start_zmq_tasks requires zmq_bridge to be set on the interface")
        if self._zmq_tasks_started:
            return
        self._zmq_tasks_started = True
        from nano_qwen3tts_vllm.zmq.dispatcher import start_dispatcher_thread, run_dispatch_loop
        
        # Start dispatcher threads for both talker and predictor addresses
        # (they publish on separate ports in separate processes)
        if hasattr(self, '_talker_zmq_address'):
            # New architecture: separate processes
            _, inbox_talker = start_dispatcher_thread(self._talker_zmq_address)
            _, inbox_predictor = start_dispatcher_thread(self._predictor_zmq_address)
            
            # Merge both inboxes into the main inbox
            import queue as sync_queue
            self._zmq_inbox = sync_queue.Queue()
            
            def merge_inboxes():
                """Merge messages from both inboxes into one."""
                import threading
                def forwarder(src, dst):
                    while True:
                        try:
                            msg = src.get()
                            if msg is None:
                                break
                            dst.put(msg)
                        except Exception:
                            break
                
                t1 = threading.Thread(target=forwarder, args=(inbox_talker, self._zmq_inbox), daemon=True)
                t2 = threading.Thread(target=forwarder, args=(inbox_predictor, self._zmq_inbox), daemon=True)
                t1.start()
                t2.start()
            
            merge_inboxes()
        else:
            # Old architecture: single process (for backwards compatibility)
            _, self._zmq_inbox = start_dispatcher_thread(self.zmq_bridge.bind_address)
        
        t1 = asyncio.create_task(run_dispatch_loop(self._zmq_inbox, self._request_queues, self._queues_lock))
        self._zmq_tasks.append(t1)
        # Give the recv thread time to connect and enter recv (avoid ZMQ slow-joiner)
        await asyncio.sleep(0.2)

    async def stop_zmq_tasks(self) -> None:
        """Stop ZMQ tasks and engine processes. Puts sentinel in inbox to unblock executor thread."""
        if not self._zmq_tasks:
            return
        
        # Send shutdown signal to engine processes if they exist
        if hasattr(self, '_talker_request_queue') and hasattr(self, '_talker_process'):
            shutdown_request = EngineRequest(action="shutdown", request_id="")
            self._talker_request_queue.put(shutdown_request)
            print(f"[Interface] Sent shutdown signal to talker process")
            
        if hasattr(self, '_predictor_request_queue') and hasattr(self, '_predictor_process'):
            shutdown_request = EngineRequest(action="shutdown", request_id="")
            self._predictor_request_queue.put(shutdown_request)
            print(f"[Interface] Sent shutdown signal to predictor process")
        
        # Unblock the thread blocked on inbox.get() in run_in_executor so shutdown_default_executor() can finish
        if self._zmq_inbox is not None:
            self._zmq_inbox.put(None)
        
        # Cancel asyncio tasks
        for t in self._zmq_tasks:
            t.cancel()
        await asyncio.gather(*self._zmq_tasks, return_exceptions=True)
        self._zmq_tasks.clear()
        self._zmq_inbox = None
        
        # Wait for engine processes to terminate
        if hasattr(self, '_talker_process'):
            self._talker_process.join(timeout=5.0)
            if self._talker_process.is_alive():
                print(f"[Interface] Talker process did not terminate, forcing...")
                self._talker_process.terminate()
                self._talker_process.join(timeout=2.0)
            print(f"[Interface] Talker process terminated")
            
        if hasattr(self, '_predictor_process'):
            self._predictor_process.join(timeout=5.0)
            if self._predictor_process.is_alive():
                print(f"[Interface] Predictor process did not terminate, forcing...")
                self._predictor_process.terminate()
                self._predictor_process.join(timeout=2.0)
            print(f"[Interface] Predictor process terminated")

    def generate_custom_voice(self, text: str, language: str = "English", speaker: str = "Vivian"):
        """Sync generator. Only valid when zmq_bridge is None. For ZMQ use generate_custom_voice_async()."""
        if self.zmq_bridge is not None:
            raise RuntimeError(
                "When using ZMQ bridge, use async API: await interface.start_zmq_tasks(); "
                "async for chunk in interface.generate_custom_voice_async(...)"
            )
        input_ids, instruct_ids, speakers, languages = prepare_custom_voice_prompt(
            text=text, language=language, speaker=speaker,
            processor=self.processor, device=self.device,
        )
        talker_input_embeds, trailing_text_hiddens, tts_pad_embed, talker_attention_mask = prepare_inputs(
            config=self.model_config,
            input_ids=input_ids, instruct_ids=instruct_ids, speakers=speakers, languages=languages,
            non_streaming_mode=True,
            text_embedding=self.text_embedding, input_embedding=self.input_embedding,
            text_projection=self.text_projection, device=self.device,
        )
        yield from self._generate_caller_driven(
            talker_input_embeds, trailing_text_hiddens, tts_pad_embed,
            str(uuid.uuid4()),
            SamplingParams(temperature=1.0, max_tokens=1),
            SamplingParams(temperature=0.9, max_tokens=17),
        )

    async def generate_custom_voice_async(
        self, text: str, language: str = "English", speaker: str = "Vivian"
    ):
        """Async generator of codebook_id chunks. Requires zmq_bridge; call await start_zmq_tasks() first."""
        if self.zmq_bridge is None:
            raise RuntimeError("generate_custom_voice_async requires zmq_bridge")

        def _prep_in_thread() -> tuple:
            """Run prep in executor so event loop can run engine_loop; lock serializes GPU prep."""
            with self._prep_lock:
                input_ids, instruct_ids, speakers, languages = prepare_custom_voice_prompt(
                    text=text, language=language, speaker=speaker,
                    processor=self.processor, device=self.device,
                )
                return prepare_inputs(
                    config=self.model_config,
                    input_ids=input_ids, instruct_ids=instruct_ids, speakers=speakers, languages=languages,
                    non_streaming_mode=True,
                    text_embedding=self.text_embedding, input_embedding=self.input_embedding,
                    text_projection=self.text_projection, device=self.device,
                )

        loop = asyncio.get_event_loop()
        talker_input_embeds, trailing_text_hiddens, tts_pad_embed, talker_attention_mask = await loop.run_in_executor(
            None, _prep_in_thread
        )
        async for chunk in self.generate_async(
            talker_input_embeds, trailing_text_hiddens, tts_pad_embed, talker_attention_mask
        ):
            yield chunk

    def generate(self, inputs_embeds: torch.Tensor, trailing_text_hiddens: torch.Tensor, tts_pad_embed: torch.Tensor, talker_attention_mask: torch.Tensor, request_id: str | None = None):
        """Sync generator. Only valid when zmq_bridge is None."""
        if self.zmq_bridge is not None:
            raise RuntimeError("When using ZMQ bridge use generate_async() after await start_zmq_tasks()")
        request_id = request_id or str(uuid.uuid4())
        talker_sampling_params = SamplingParams(temperature=1.0, max_tokens=1)
        predictor_sampling_params = SamplingParams(temperature=0.9, max_tokens=17)
        yield from self._generate_caller_driven(
            inputs_embeds, trailing_text_hiddens, tts_pad_embed,
            request_id, talker_sampling_params, predictor_sampling_params,
        )

    async def generate_async(
        self,
        inputs_embeds: torch.Tensor,
        trailing_text_hiddens: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        talker_attention_mask: torch.Tensor,
        request_id: str | None = None,
    ):
        """Async generator of codebook_id chunks. ZMQ path; step() runs in separate engine processes. Call await start_zmq_tasks() first."""
        if self.zmq_bridge is None:
            raise RuntimeError("generate_async requires zmq_bridge")
        talker_sampling_params = SamplingParams(temperature=1.0, max_tokens=1)
        predictor_sampling_params = SamplingParams(temperature=0.9, max_tokens=17)
        request_id = request_id or str(uuid.uuid4())
        request_queue: asyncio.Queue = asyncio.Queue()
        async with self._queues_lock:
            self._request_queues[request_id] = request_queue
        try:
            next_talker_embeds = inputs_embeds
            if next_talker_embeds.dim() == 2:
                next_talker_embeds = next_talker_embeds.unsqueeze(0)
            generation_step = 0
            
            # Send add_request to talker engine process via multiprocessing queue
            # Detach tensors so they can be pickled across process boundaries (no requires_grad)
            talker_request = EngineRequest(
                action="add_request",
                request_id=request_id,
                inputs_embeds=next_talker_embeds.detach().clone(),
                sampling_params=talker_sampling_params,
            )
            self._talker_request_queue.put(talker_request)

            while True:
                engine_type, msg_type, payload = await request_queue.get()
                if engine_type == "talker" and msg_type == "done":
                    # Send clear_request to talker engine process
                    clear_request = EngineRequest(
                        action="clear_request",
                        request_id=request_id,
                    )
                    self._talker_request_queue.put(clear_request)
                    break
                if engine_type == "talker" and msg_type == "token":
                    token_ids = payload["token_ids"]
                    hidden_states = payload.get("hidden_states")
                    last_id = token_ids[-1]
                    if last_id == 2150:
                        # Send clear_request to talker engine process
                        clear_request = EngineRequest(
                            action="clear_request",
                            request_id=request_id,
                        )
                        self._talker_request_queue.put(clear_request)
                        break
                    last_id_hidden = self.input_embedding(torch.tensor([last_id], device=self.device)).unsqueeze(0)
                    if hidden_states is not None:
                        h = torch.from_numpy(hidden_states.copy()).to(self.device)
                        if h.dim() == 1:
                            h = h.unsqueeze(0).unsqueeze(0)
                        else:
                            h = h.unsqueeze(0).unsqueeze(0)
                        last_hidden_state = h
                    else:
                        last_hidden_state = last_id_hidden.unsqueeze(0)
                    predictor_inputs_embeds = torch.cat((last_hidden_state, last_id_hidden), dim=1)
                    
                    # Send add_request to predictor engine process via multiprocessing queue
                    # Detach tensors so they can be pickled across process boundaries (no requires_grad)
                    predictor_request = EngineRequest(
                        action="add_request",
                        request_id=request_id,
                        inputs_embeds=predictor_inputs_embeds.detach().clone(),
                        sampling_params=predictor_sampling_params,
                    )
                    self._predictor_request_queue.put(predictor_request)
                    
                    _, _, payload2 = await request_queue.get()
                    pred_token_ids = payload2.get("token_ids", [])
                    codebook_ids = [last_id] + pred_token_ids
                    yield codebook_ids

                    codec_hiddens = torch.cat(
                        [last_id_hidden]
                        + [self.predictor_input_embeddings[i](torch.tensor([pred_token_ids[i]], device=self.device)).unsqueeze(0) for i in range(15)],
                        dim=1,
                    )
                    next_talker_embeds = codec_hiddens.sum(1, keepdim=True)
                    if generation_step < trailing_text_hiddens.shape[1]:
                        next_talker_embeds = next_talker_embeds + trailing_text_hiddens[:, generation_step].unsqueeze(1)
                    else:
                        next_talker_embeds = next_talker_embeds + tts_pad_embed
                    generation_step += 1
                    
                    # Send next add_request to talker engine process
                    talker_request = EngineRequest(
                        action="add_request",
                        request_id=request_id,
                        inputs_embeds=next_talker_embeds.detach().clone(),
                        sampling_params=talker_sampling_params,
                    )
                    self._talker_request_queue.put(talker_request)
        finally:
            async with self._queues_lock:
                self._request_queues.pop(request_id, None)

    def _generate_caller_driven(
        self,
        inputs_embeds: torch.Tensor,
        trailing_text_hiddens: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        request_id: str,
        talker_sampling_params: SamplingParams,
        predictor_sampling_params: SamplingParams,
    ):
        generation_step = 0
        next_talker_embeds = inputs_embeds
        if next_talker_embeds.dim() == 2:
            next_talker_embeds = next_talker_embeds.unsqueeze(0)

        while True:
            self.talker_llm.add_request([next_talker_embeds], talker_sampling_params, request_id=request_id)
            _, _, outputs_all = self.talker_llm.step_with_outputs()
            if not outputs_all:
                self.talker_llm.clear_request(request_id)
                return

            match = next((o for o in outputs_all if o[0] == request_id), None)
            if match is None:
                continue
            _, _, token_ids, hidden_states, is_finished = match
            last_id = token_ids[-1]
            if last_id == 2150:
                self.talker_llm.clear_request(request_id)
                return

            last_id_hidden = self.input_embedding(torch.tensor([last_id], device=self.device)).unsqueeze(0)
            last_hidden_state = hidden_states.unsqueeze(0).unsqueeze(0)
            predictor_inputs_embeds = torch.cat((last_hidden_state, last_id_hidden), dim=1)
            predictor_outputs = self.predictor_llm.generate(
                [predictor_inputs_embeds.unsqueeze(0)],
                predictor_sampling_params,
                use_tqdm=False,
                request_id=request_id,
            )
            pred_token_ids = predictor_outputs[0]["token_ids"]
            codebook_ids = [last_id] + pred_token_ids
            yield codebook_ids

            codec_hiddens = torch.cat(
                [last_id_hidden]
                + [self.predictor_input_embeddings[i](torch.tensor([pred_token_ids[i]], device=self.device)).unsqueeze(0) for i in range(15)],
                dim=1,
            )
            next_talker_embeds = codec_hiddens.sum(1, keepdim=True)
            if generation_step < trailing_text_hiddens.shape[1]:
                next_talker_embeds = next_talker_embeds + trailing_text_hiddens[:, generation_step].unsqueeze(1)
            else:
                next_talker_embeds = next_talker_embeds + tts_pad_embed
            generation_step += 1


if __name__ == "__main__":
    # Example: Use HuggingFace model ID (automatically downloads if needed)
    # interface = Qwen3TTSInterface.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    
    # Or use local path
    interface = Qwen3TTSInterface(model_path="/work/weights/qwen3tts")
    print("Warm up...")
    audio_codes = list(interface.generate_custom_voice(text="Hi there this is a test.", language="English", speaker="Vivian"))

    print("Generate...")
    start = time.time()
    audio_codes = list(interface.generate_custom_voice(text="Hi there, this is tsdocode, hope you are doing well.", language="English", speaker="Vivian"))
    end = time.time()

    
    
    
    