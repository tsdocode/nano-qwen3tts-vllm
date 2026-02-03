# nano-qwen3tts-vllm

Qwen3-TTS with nano vLLM-style optimizations for fast text-to-speech generation.

## Highlights

- **Continuous Batching** — Batches multiple sequences and schedules prefill/decode across them for higher throughput.
- **Page Attention** — Paged KV cache with block tables and slot mapping for efficient memory use and variable-length sequences.
- **CUDA Graph** — Predictor and speech decoder use captured CUDA graphs (multiple batch sizes / decode lengths) to reduce kernel launch overhead.
- **Streaming Support** — Async generation with ZMQ: stream codec chunks as they are produced; API returns PCM audio stream (e.g. `POST /v1/audio/speech` with `StreamingResponse`).

## Installation

**Requirements**

- Python ≥3.10
- PyTorch ≥2.10 with CUDA
- **Compute capability ≥8.0** (e.g. Ampere/Ada) for Flash Attention
- `qwen-tts`, `transformers`, and other deps below

**Flash Attention (recommended)**

For a fast install without building from source, use pre-built wheels:

```bash
# Example: Python 3.12, CUDA 12.4, PyTorch 2.5
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.0/flash_attn-2.6.3+cu124torch2.5-cp312-cp312-linux_x86_64.whl
```

Pick the wheel that matches your Python, CUDA, and PyTorch from:  
[https://github.com/mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels)

**Project**

```bash
git clone https://github.com/tsdocode/nano-qwen3tts-vllm.git
cd nano-qwen3tts-vllm
uv sync
# or
pip install -e .
```

## Usage

### Basic TTS generation

```python
from nano_qwen3tts_vllm.interface import Qwen3TTSInterface
from nano_qwen3tts_vllm.utils.speech_tokenizer_cudagraph import SpeechTokenizerCUDAGraph
import soundfile as sf

interface = Qwen3TTSInterface(
    model_path="/path/to/qwen3tts",
    enforce_eager=False,   # use CUDA graphs when False
    tensor_parallel_size=1,
)

audio_codes = interface.generate_custom_voice(
    text="Hello, this is a test.",
    language="English",
    speaker="Vivian",
)

tokenizer = SpeechTokenizerCUDAGraph("/path/to/qwen3tts", device="cuda:0")
wavs, sr = tokenizer.decode([{"audio_codes": audio_codes}])
sf.write("output.wav", wavs[0], sr)
```

### Streaming (ZMQ + async)

With `USE_ZMQ=1`, the server uses an async engine loop and streams codec chunks; `POST /v1/audio/speech` returns a streaming PCM response.

```python
# Client: stream PCM and write to file (see examples/client.py)
import requests
r = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={"text": "Hello world.", "language": "English", "speaker": "Vivian"},
    stream=True,
)
# consume r.iter_content() and write to WAV
```

### Run server

```bash
export QWEN3_TTS_MODEL_PATH=/path/to/qwen3tts
export USE_ZMQ=1
python -m uvicorn examples.server:app --host 0.0.0.0 --port 8000
# or
python examples/server.py
```

## Options

| Parameter | Description |
|-----------|--------------|
| `model_path` | Path to Qwen3-TTS model (custom voice) |
| `enforce_eager` | Disable CUDA graphs (for debugging) |
| `tensor_parallel_size` | Number of GPUs (1–8) |
| `USE_ZMQ` | Use ZMQ + async engine for streaming (server) |
| `QWEN3_TTS_MODEL_PATH` | Model directory (server env) |

## Benchmark (L4 GPU, 0.6B custom model, decode wav each 1 chunk)

| Setup | First chunk latency (16 codec codes) | Inner chunk latency | RTF |
|------|--------------------------------------|---------------------|-----|
| **1 CCU** | 160 ms | 50 ms | 0.65 |
| **2 CCUs** | 250 ms | 90 ms | 1.125 |

*(CCU = concurrent request / “concurrent chunk unit” in your setup.)*

## Note

Currently only the **custom voice** model is supported.


# Further Optimization (contribution is welcome)
- Chunk prefill for smaller first chunk latency: Current running prefill stage on eager mode
- Support voice clone, voice design
- Make prefill stage run with CUDAGraph

## Star History
![Star History Chart](https://api.star-history.com/svg?repos=tsdocode/nano-qwen3tts-vllm&type=Date)



