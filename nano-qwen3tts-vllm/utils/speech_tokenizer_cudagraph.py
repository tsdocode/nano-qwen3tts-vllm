"""Speech tokenizer with CUDA graph capture for fast decode (50 graphs T=1..50)."""
import os
import sys
import torch
from typing import Union, Tuple, List

try:
    qwen_tts_path = os.path.expanduser(os.environ.get("QWEN_TTS_PATH", "/home/sang/work/Qwen3-TTS"))
    if os.path.exists(qwen_tts_path) and qwen_tts_path not in sys.path:
        sys.path.insert(0, qwen_tts_path)
    from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer as _Qwen3TTSTokenizer
    HAS_SPEECH_TOKENIZER = True
except ImportError as e:
    HAS_SPEECH_TOKENIZER = False
    _Qwen3TTSTokenizer = None


def _capture_decoder_cudagraphs(decoder, device: str, graph_lengths: List[int]):
    """Capture 50 CUDA graphs for decoder (one per length T=1..50). Patches decoder.forward in-place."""
    decoder.eval()
    # Warmup
    with torch.inference_mode():
        _ = decoder(torch.randint(0, 100, (1, 16, 100), device=device))
        torch.cuda.synchronize()

    decoder.graphs = {}
    decoder.graph_inputs = {}
    decoder.graph_outputs = {}
    graph_pool = None

    with torch.inference_mode():
        for T in reversed(graph_lengths):
            graph = torch.cuda.CUDAGraph()
            input_buf = torch.randint(0, 100, (1, 16, T), device=device, dtype=torch.long)
            _ = decoder(input_buf)
            torch.cuda.synchronize()
            with torch.cuda.graph(graph, graph_pool):
                output_buf = decoder(input_buf)
            if graph_pool is None:
                graph_pool = graph.pool()
            decoder.graphs[T] = graph
            decoder.graph_inputs[T] = input_buf
            decoder.graph_outputs[T] = output_buf
            torch.cuda.synchronize()

    decoder.original_forward = decoder.forward

    def forward_with_graph_replay(codes):
        T = codes.shape[2]
        if T in decoder.graphs:
            decoder.graph_inputs[T].copy_(codes)
            decoder.graphs[T].replay()
            return decoder.graph_outputs[T]
        return decoder.original_forward(codes)

    decoder.forward = forward_with_graph_replay


class SpeechTokenizerCUDAGraph:
    """Qwen3-TTS speech tokenizer with 50 captured CUDA graphs for fast decode.

    Loads the 12Hz tokenizer, captures one graph per decode length T=1..50,
    and patches the decoder to replay graphs when shape matches (like predictor_model_runner).
    """

    def __init__(
        self,
        model_path: str,
        device: str = None,
        dtype: torch.dtype = torch.bfloat16,
        num_graph_lengths: int = 1,
    ):
        """Load tokenizer and capture CUDA graphs for decoder.

        Args:
            model_path: Path to model dir; speech tokenizer loaded from {model_path}/speech_tokenizer.
            device: Device for model (default: cuda:0 if available).
            dtype: Model dtype (default: bfloat16).
            num_graph_lengths: Number of graphs to capture for lengths 1..num_graph_lengths (default: 50).
        """
        if not HAS_SPEECH_TOKENIZER:
            raise ImportError(
                "qwen_tts not found. Install Qwen3-TTS and set QWEN_TTS_PATH or add to path."
            )
        device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        speech_tokenizer_path = model_path

        print(f"Loading speech tokenizer (CUDAGraph) from {speech_tokenizer_path}...")
        self.tokenizer = _Qwen3TTSTokenizer.from_pretrained(
            speech_tokenizer_path,
            device_map=device,
        )
        self.tokenizer.model = self.tokenizer.model.to(dtype)

        if hasattr(self.tokenizer.config, "sample_rate"):
            self.sample_rate = self.tokenizer.config.sample_rate
        elif hasattr(getattr(self.tokenizer, "feature_extractor", None), "sampling_rate"):
            self.sample_rate = self.tokenizer.feature_extractor.sampling_rate
        else:
            self.sample_rate = 12500

        self.device = device
        self.dtype = dtype

        if device.startswith("cuda") and num_graph_lengths > 0:
            graph_lengths = list(range(1, num_graph_lengths + 1))
            print(f"Capturing {len(graph_lengths)} CUDA graphs for decoder (T=1..{num_graph_lengths})...")
            _capture_decoder_cudagraphs(self.tokenizer.model.decoder, device, graph_lengths)
            print("CUDA graph capture done.")
        else:
            print("Skipping CUDA graph capture (CPU or num_graph_lengths=0).")

        print(f"Speech tokenizer (CUDAGraph) loaded: sample_rate={self.sample_rate}Hz, device={self.device}")

    @torch.inference_mode()
    def decode(self, inputs: List[dict]) -> Tuple[List, int]:
        """Decode audio_codes to waveform. Same API as Qwen3TTSTokenizer.decode.

        Args:
            inputs: List of dicts with key 'audio_codes' (tensor [time, 16] or list).

        Returns:
            (wavs, sample_rate).
        """
        return self.tokenizer.decode(inputs)

    @torch.inference_mode()
    def decode_codec_ids(self, codec_ids: torch.Tensor) -> Tuple[List, int]:
        """Decode codec IDs [batch, 16, time] to (audio_list, sample_rate). Drop-in for SpeechTokenizer.decode."""
        batch_size = codec_ids.shape[0]
        inputs = []
        for i in range(batch_size):
            codes = codec_ids[i]
            if codes.dim() == 2:
                codes = codes.transpose(0, 1)
            inputs.append({"audio_codes": codes})
        return self.decode(inputs)
