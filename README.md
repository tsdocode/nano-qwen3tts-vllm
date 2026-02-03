# nano-qwen3tts-vllm

Qwen3-TTS with nano vLLM-style optimizations for fast text-to-speech generation.

## Author note
This project was born after I tested the Qwen3-TTS model and realized the RTF (Real-Time Factor) remained stubbornly high, even on an H100. It sparked an idea: could we create a **"nano-vllm" moment** for Qwen3TTS? My goal is to match the efficiency of vLLM implementation within a concise, ~1k line codebase.

I am building this while diving deep into both the [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and [Qwen3TTS](https://github.com/QwenLM/Qwen3-TTS) source code. I anticipate some hurdles along the way and would appreciate any insights or feedback from the community.

Optimization for the Qwen3-TTS model will continue in the [vllm-omni](https://github.com/vllm-project/vllm-omni) repo—stay tuned for updates!

## Highlights

### Performance Optimizations
- **Continuous Batching** — Batches multiple sequences and schedules prefill/decode across them for higher throughput.
- **Page Attention** — Paged KV cache with block tables and slot mapping for efficient memory use and variable-length sequences.
- **CUDA Graph** — Predictor and speech decoder use captured CUDA graphs (multiple batch sizes / decode lengths) to reduce kernel launch overhead.
- **Streaming Support** — Async generation with ZMQ: stream codec chunks as they are produced; API returns PCM audio stream (e.g. `POST /v1/audio/speech` with `StreamingResponse`).

## Benchmark Results

### Performance Comparison (Voice Design Model)

Tested on NVIDIA H100 with `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` with script `examples/quick_benchmark.py`:
| Metric | nano-vllm | Original Qwen3-TTS | Improvement |
|--------|-----------|-------------------|-------------|
| **Avg Generation Time** | 2.612s | 8.487s | **3.25x faster** |
| **Real-Time Factor (RTF)** | 0.399 | 1.467 | **3.68x better** |

**Key Findings:**
- 🚀 **4.86x faster** generation speed
- 📊 **RTF < 0.4** means nano-vllm generates audio **2.8x faster than real-time**
- ⚡ Original implementation has RTF ~2.0 (slower than real-time)
- 💪 Consistent performance across different text lengths

### Streaming Performance (Custom Voice Model)

L4 GPU, 0.6B custom model, decode wav each 1 chunk

| Setup | First chunk latency (16 codec codes) | Inner chunk latency | RTF |
|------|--------------------------------------|---------------------|-----|
| **1 CCU** | 160 ms | 50 ms | 0.65 |
| **2 CCUs** | 250 ms | 90 ms | 1.125 |

*(CCU = concurrent request / “concurrent chunk unit” in the setup.)*

### Feature Completeness
- ✅ **All Model Types Supported** — CustomVoice (pre-defined   speakers), VoiceDesign (text-to-voice), and Base (voice cloning)
- ✅ **Voice Cloning** — ICL mode and x_vector_only mode for reference audio-based voice cloning
- ✅ **Voice Design** — Generate voices from natural language descriptions
- ✅ **Streaming Generation** — Generator-based API for codec chunk streaming
- ✅ **Multi-language** — English, Chinese, and auto-detection support

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

## Supported Models

nano-qwen3tts-vllm supports **all three Qwen3-TTS model types**:

| Model | Features | Language Support | Streaming | Instruction Control |
|---|---|---|---|---|
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | Performs voice design based on user-provided descriptions. | Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian | ✅ | ✅ |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | Provides style control over target timbres via user instructions; supports 9 premium timbres covering various combinations of gender, age, language, and dialect. | Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian | ✅ | ✅ |
| Qwen3-TTS-12Hz-1.7B-Base | Base model capable of 3-second rapid voice clone from user audio input; can be used for fine-tuning (FT) other models. | Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian | ✅ |  |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | Supports 9 premium timbres covering various combinations of gender, age, language, and dialect. | Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian | ✅ |  |
| Qwen3-TTS-12Hz-0.6B-Base | Base model capable of 3-second rapid voice clone from user audio input; can be used for fine-tuning (FT) other models. | Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian | ✅ |  |

All models support both **12Hz** (default, faster) and **25Hz** (higher quality) variants.

> 💡 **For complete, runnable examples**, see the [`examples/`](examples/) directory. Each example includes detailed usage instructions and demonstrates all features.

## API Pattern

**All generation methods return codec chunk generators for streaming support:**

1. **Generate codec chunks** - Call `generate_*()` and wrap with `list()` to collect all chunks
2. **Decode to audio** - Use `interface.speech_tokenizer.decode()` to convert chunks to audio

This design enables:
- ✅ Streaming codec generation for low latency
- ✅ Consistent API across all model types  
- ✅ Separation of generation and decoding steps
- ✅ Easy integration with streaming servers

## Usage

### 1. Custom Voice (Pre-defined Speakers)

Generate speech with built-in speaker voices (e.g., Vivian, Mike, Sarah, etc.):

```python
from nano_qwen3tts_vllm.interface import Qwen3TTSInterface
import soundfile as sf

# Load CustomVoice model
interface = Qwen3TTSInterface.from_pretrained(
    pretrained_model_name_or_path="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    enforce_eager=False,
    tensor_parallel_size=1,
)

# Generate codec chunks
audio_codes = list(interface.generate_custom_voice(
    text="Hello, this is a test.",
    language="English",
    speaker="Vivian",
))

# Decode to audio using interface's built-in speech tokenizer
wavs, sr = interface.speech_tokenizer.decode([{"audio_codes": audio_codes}])
sf.write("output.wav", wavs[0], sr)
```

**Available speakers**: Vivian, Mike, Sarah, Laura, Alex, Ethan, Emma, and more (see model card).

### 2. Voice Design (Text-to-Voice)

Create voices from text descriptions:

```python
from nano_qwen3tts_vllm.interface import Qwen3TTSInterface
import soundfile as sf

# Load VoiceDesign model
interface = Qwen3TTSInterface.from_pretrained(
    pretrained_model_name_or_path="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    enforce_eager=False,
    tensor_parallel_size=1,
)

# Generate with voice design instruction
audio_codes = list(interface.generate_voice_design(
    text="Hi! How are you doing today?",
    language="English",
    instruct="A young woman with a warm, friendly voice and slight excitement",
))

# Decode to audio using interface's built-in speech tokenizer
wavs, sr = interface.speech_tokenizer.decode([{"audio_codes": audio_codes}])
sf.write("output_designed.wav", wavs[0], sr)
```

**See**: [`examples/voice_design_example.py`](examples/voice_design_example.py) for more examples.

### 3. Voice Clone (Reference Audio)

Clone voices from reference audio samples:

```python
from nano_qwen3tts_vllm.interface import Qwen3TTSInterface
import soundfile as sf

# Load Base model
interface = Qwen3TTSInterface.from_pretrained(
    pretrained_model_name_or_path="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    enforce_eager=False,
    tensor_parallel_size=1,
)

# Load reference audio
ref_audio, ref_sr = sf.read("reference.wav")

# Create voice clone prompt (ICL mode - with reference text)
voice_clone_prompt = interface.create_voice_clone_prompt(
    ref_audio=(ref_audio, ref_sr),
    ref_text="This is the reference text that was spoken in the audio.",
    x_vector_only_mode=False,  # ICL mode for better quality
)

# Generate codec chunks with cloned voice
audio_codes = list(interface.generate_voice_clone(
    text="Hello, this is a cloned voice speaking.",
    language="English",
    voice_clone_prompt=voice_clone_prompt,
))

# Decode to audio using interface's built-in speech tokenizer
wavs, sr = interface.speech_tokenizer.decode([{"audio_codes": audio_codes}])
sf.write("output_cloned.wav", wavs[0], sr)
```

**Voice Clone Modes:**
- **ICL mode** (`x_vector_only_mode=False`): Uses both speaker embedding and reference audio codes. Requires `ref_text`. Better quality and more accurate voice matching.
- **x_vector_only mode** (`x_vector_only_mode=True`): Uses only speaker embedding. No `ref_text` needed. Faster but less accurate.

**See**: [`examples/voice_clone_example.py`](examples/voice_clone_example.py) for comprehensive examples with ICL and x_vector modes.

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

## Examples

> 📁 **See [`examples/`](examples/) for complete, runnable code!**

Comprehensive example scripts are provided in the [`examples/`](examples/) directory:

- **[`custom_voice_example.py`](examples/custom_voice_example.py)** - Pre-defined speaker voices
- **[`voice_design_example.py`](examples/voice_design_example.py)** - Voice design from text descriptions
- **[`voice_clone_example.py`](examples/voice_clone_example.py)** - Voice cloning with ICL and x_vector modes
- **[`server.py`](examples/server.py)** - FastAPI server with streaming support
- **[`client.py`](examples/client.py)** - Client for streaming API

### Running Examples

```bash
# Custom Voice
python examples/custom_voice_example.py \
    --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --text "Hello world" \
    --speaker Vivian

# Voice Design
python examples/voice_design_example.py \
    --model-path Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --output-dir output

# Voice Clone
python examples/voice_clone_example.py \
    --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --ref-audio reference.wav \
    --ref-text "Reference transcript" \
    --output-dir output
```

## Advanced Features

### Using Local Models

If you have models downloaded locally:

```python
interface = Qwen3TTSInterface(
    model_path="/path/to/local/model",
    enforce_eager=False,
    tensor_parallel_size=1,
)
```

### Non-Streaming Mode

For better quality in non-real-time scenarios:

```python
audio_codes = list(interface.generate_voice_design(
    text="Hello world",
    language="English",
    instruct="professional voice",
    non_streaming_mode=True,  # Better for offline processing
))
wavs, sr = interface.speech_tokenizer.decode([{"audio_codes": audio_codes}])
```

### Supported Languages

All models support multiple languages:
- **English** - Full support
- **Chinese** (中文) - Full support with dialect variants
- **Auto** - Automatic language detection

Example with Chinese:

```python
audio_codes = list(interface.generate_voice_design(
    text="你好，世界！",
    language="Chinese",
    instruct="温暖的女声",
))
wavs, sr = interface.speech_tokenizer.decode([{"audio_codes": audio_codes}])
```

## Performance Tips

1. **Use CUDA Graphs** (`enforce_eager=False`) for 2-3x speedup
2. **Use 12Hz models** for faster generation (25Hz for higher quality)
3. **Enable streaming mode** (ZMQ) for lowest latency in production
4. **Use generators** — Process codec chunks as they're generated instead of collecting all at once

## Further Optimization (contributions welcome)

- ✅ Support for all model types (CustomVoice, VoiceDesign, Base)
- ✅ Voice clone with ICL and x_vector modes
- ✅ Voice design from text descriptions
- ⏳ Make prefill stage run with CUDA Graph

## Star History
![Star History Chart](https://api.star-history.com/svg?repos=tsdocode/nano-qwen3tts-vllm&type=Date)



