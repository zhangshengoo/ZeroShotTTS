# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start Commands

### Setup and Installation
```bash
# Clone and setup submodules
git clone https://github.com/your-repo/ZeroShotTTS.git
cd ZeroShotTTS
git submodule update --init --recursive

# Install dependencies
pip install -e .
```

### Core Development Commands
```bash
# Run Fish-Speech inference
fishspeech-infer --text "Hello world" --reference-audio ref.wav --reference-text "Hello"

# Run GPT-SoVITS API
python TTS_Model/GPT-SoVITS/api.py -dr ref.wav -dt "reference text" -dl zh

# Run CosyVoice API
python TTS_Model/CosyVoice/api.py --port 50000

# Batch processing with GPU management
python infer.py --input_dir ./texts --output_dir ./output --model fish-speech

# Start Fish-Speech API server
python -m tools.api_server --llama-checkpoint path/to/checkpoint
```

### Testing Commands
```bash
# Test unified API for different models
python -c "from infer_api.fishspeech_api import FishSpeechTTS; tts = FishSpeechTTS(); print('FishSpeech API loaded successfully')"
python -c "from infer_api.gptSoVITS_tts import GPTSoVITSTTS; tts = GPTSoVITSTTS(); print('GPTSoVITS API loaded successfully')"
python -c "from infer_api.cosyvoice_tts import CosyVoiceTTS; tts = CosyVoiceTTS(); print('CosyVoice API loaded successfully')"

# Test speaker registration
curl -X POST http://localhost:8080/register_speaker -F "audio=@ref.wav" -F "text=Hello"

# Test TTS synthesis
curl -X POST http://localhost:8080/tts -H "Content-Type: application/json" -d '{"text": "Hello world", "speaker_id": "speaker_1"}'
```

### Development and Debugging
```bash
# Check GPU status
nvidia-smi

# Test audio file format
soxi ref.wav  # Should show 16kHz, mono, 16-bit

# Run single test file
python -m pytest tests/test_api.py -v

# Format code
black infer_api/
isort infer_api/
```

## Architecture Overview

ZeroShotTTS is a unified text-to-speech system supporting **Fish-Speech**, **GPT-SoVITS**, and **CosyVoice** models:

- **Fish-Speech**: Transformer-based TTS with Llama encoder + VQGAN decoder
- **GPT-SoVITS**: GPT-based prosody modeling + SoVITS audio synthesis
- **CosyVoice**: Multi-language TTS from Alibaba with API-based integration

### Key Components

```
ZeroShotTTS/
├── infer.py                 # Main batch processing entry with GPU management
├── infer_api/               # Unified API layer
│   ├── baseTTS.py          # Abstract TTS interface (BaseTTS class)
│   ├── fishspeech_api.py   # Fish-Speech wrapper implementation
│   ├── gptSoVITS_tts.py    # GPT-SoVITS wrapper implementation
│   └── cosyvoice_tts.py    # CosyVoice wrapper implementation
├── TTS_Model/              # Model implementations (git submodules)
│   ├── fish-speech/        # Transformer-based TTS
│   ├── GPT-SoVITS/         # GPT + SoVITS models
│   └── CosyVoice/          # Multi-language TTS from Alibaba
├── examples/               # Sample files and reference audio
├── scripts/                # Utility scripts for setup
└── docs/                   # Documentation
```

### BaseTTS Interface
All TTS models implement the BaseTTS abstract class with these key methods:
- `register_speaker(prompt_audio, prompt_text)`: Register new speakers
- `tts(text)`: Synthesize with default speaker
- `tts_with_speaker(text, speaker_id)`: Synthesize with specific speaker
- `_process_speaker_embedding()`: Model-specific speaker embedding

## Development Patterns

### Adding New TTS Models
1. Create wrapper in `infer_api/` implementing `BaseTTS` interface
2. Add model to `TTS_Model/` as git submodule
3. Update `infer.py` for batch processing support
4. Add model-specific configuration in wrapper class
5. **Use local GPU inference only** - All model implementations should use direct local model loading and GPU inference, avoid API calls

### Speaker Registration Flow
```python
from infer_api.fishspeech_api import FishSpeechTTS
from infer_api.gptSoVITS_tts import GPTSoVITSTTS
from infer_api.cosyvoice_tts import CosyVoiceTTS

# Initialize models
tts = FishSpeechTTS()  # or GPTSoVITSTTS() or CosyVoiceTTS()

# Register speaker and synthesize
speaker_id = tts.register_speaker(prompt_audio="ref.wav", prompt_text="Reference text")
audio = tts.tts_with_speaker("Hello world", speaker_id)
```

### GPU Management and Resource Allocation
- **Automatic GPU Selection**: Based on memory availability and task count
- **Multi-GPU Load Balancing**: Distributes tasks across available GPUs via `infer.py`
- **Resource Monitoring**: Tracks GPU memory usage and task limits
- **Concurrent Processing**: Semaphore-based locking for parallel synthesis
- **Memory Requirements**: Models need 8GB+ VRAM, check with `nvidia-smi`

### Batch Processing System
The `infer.py` script provides advanced features:
- Concurrent synthesis with configurable worker count
- Automatic GPU assignment based on memory and task load
- Progress tracking and error handling
- Support for multiple input/output formats

## Configuration and Model Management

### Configuration Files
- **Fish-Speech**: `TTS_Model/fish-speech/configs/` (YAML configs)
- **GPT-SoVITS**: `TTS_Model/GPT-SoVITS/config.py`
- **CosyVoice**: `TTS_Model/CosyVoice/cosyvoice.yaml`
- **Unified API**: Environment variables override defaults

### Model Locations
Models are stored in:
- **Fish-Speech**: `TTS_Model/fish-speech/checkpoints/`
- **GPT-SoVITS**: `TTS_Model/GPT-SoVITS/GPT_weights/` and `SoVITS_weights/`
- **CosyVoice**: Downloaded automatically or placed in `TTS_Model/CosyVoice/pretrained_models/`

### Supported Languages
- Chinese (Mandarin)
- English
- Japanese
- Korean
- Cantonese

## Common Issues and Solutions

### Setup Issues
- **Submodule problems**: Run `git submodule update --init --recursive`
- **Missing dependencies**: Check `requirements.txt` and install with `pip install -e .`
- **CUDA compatibility**: Ensure PyTorch CUDA version matches system

### Runtime Issues
- **GPU OOM**: Reduce batch size, check `nvidia-smi`, models need 8GB+ VRAM
- **Audio format errors**: Use 16kHz mono WAV for best results (`soxi ref.wav` to check)
- **Speaker audio quality**: 3-10 seconds recommended for voice cloning
- **Model loading failures**: Check model paths and file permissions

### Performance Optimization
- **Multi-GPU scaling**: Use `infer.py` for automatic load balancing
- **Batch size tuning**: Adjust based on GPU memory and text length
- **Audio preprocessing**: Normalize audio levels and remove silence
- **Concurrent processing**: Configure worker count based on GPU count

## Code Implementation Guidelines

### Model Implementation Standards
All TTS model wrappers in `infer_api/` must follow these standards:

#### 1. Local GPU Inference Only
- **No API calls**: All models must use direct local GPU inference
- **Direct model loading**: Load models locally using the model's native Python interface
- **GPU optimization**: Implement proper CUDA device management and memory optimization
- **Example**: CosyVoiceTTS uses direct `CosyVoice2()` initialization instead of HTTP API calls

#### 2. Import Conventions
- **Direct imports**: Import model dependencies directly without try-except blocks
- **Path setup**: Add model paths to sys.path before imports
- **Clean imports**: Avoid conditional imports that complicate debugging

```python
# ✅ Good - Direct import
sys.path.insert(0, str(cosyvoice_path))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2

# ❌ Avoid - Conditional import
try:
    from cosyvoice.cli.cosyvoice import CosyVoice
except ImportError:
    CosyVoice = None
```

#### 3. Model Initialization Pattern
```python
def __init__(self, model_dir: str = None, device: str = None):
    # Set default paths
    if model_dir is None:
        model_dir = self._get_default_model_path()

    # Handle model download if needed
    if not os.path.exists(model_dir):
        from modelscope import snapshot_download
        model_dir = snapshot_download(model_id, local_dir=model_dir)

    # Initialize model with GPU support
    self.model = ModelClass(model_dir, fp16=torch.cuda.is_available())
    self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
```

#### 4. Speaker Management
- Use model's native speaker registration when available
- Implement speaker caching to avoid repeated loading
- Support both file paths and numpy arrays for audio input

#### 5. Inference Implementation
- Use model's native inference methods
- Convert output to numpy array format consistently
- **Sample rate normalization**: All models must output 16kHz audio for consistency
- Handle batch processing efficiently
- Implement proper error handling and logging

#### 6. Sample Rate Standardization
All TTS models must output audio at **16kHz sample rate** for consistency:
```python
# Model may output different sample rates, resample to 16kHz
if self.model_sample_rate != 16000:
    audio = librosa.resample(
        audio,
        orig_sr=self.model_sample_rate,
        target_sr=16000
    )
```

#### 7. BaseTTS Interface Compliance
All models must implement:
- `register_speaker()`: Speaker registration with audio/text pairs
- `tts()`: Basic synthesis with default speaker
- `tts_with_speaker()`: Synthesis with specific speaker
- `is_available()`: Model availability check
- `get_model_info()`: Model metadata and capabilities

## Testing and Validation

### Unit Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific model tests
python -m pytest tests/test_fishspeech.py -v
python -m pytest tests/test_gptsovits.py -v
python -m pytest tests/test_cosyvoice.py -v
```

### Integration Tests
```bash
# Test API endpoints
curl -X POST http://localhost:8080/register_speaker -F "audio=@examples/ref.wav" -F "text=Hello"
curl -X POST http://localhost:8080/tts -H "Content-Type: application/json" -d '{"text": "Test synthesis", "speaker_id": "test_speaker"}'

# Test batch processing
python infer.py --input_dir examples/texts --output_dir test_output --model fish-speech --max_workers 2
```