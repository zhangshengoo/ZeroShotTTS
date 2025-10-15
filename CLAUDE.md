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

# Download Index-TTS models (optional, if using Index-TTS)
cd TTS_Model/index-tts
mkdir -p checkpoints
cd checkpoints
# Download from HuggingFace
huggingface-cli download IndexTeam/IndexTTS-2 --local-dir=.
# Or download manually from: https://huggingface.co/IndexTeam/IndexTTS-2
```

### Core Development Commands
```bash
# Run Fish-Speech inference
fishspeech-infer --text "Hello world" --reference-audio ref.wav --reference-text "Hello"

# Run GPT-SoVITS API
python TTS_Model/GPT-SoVITS/api.py -dr ref.wav -dt "reference text" -dl zh

# Run CosyVoice API
python TTS_Model/CosyVoice/api.py --port 50000

# Run Index-TTS inference
python -c "from infer_api.index_tts import IndexTTSTTS; tts = IndexTTSTTS(); audio = tts.tts_with_speaker('Hello world', 'speaker_1')"

# Run MeloTTS inference (multi-lingual)
python -c "from infer_api.melo_tts import MeloTTSTTS; tts = MeloTTSTTS(language='EN'); audio = tts.tts_with_speaker('Hello world', 'speaker_1')"
python -c "from infer_api.melo_tts import MeloTTSTTS; tts = MeloTTSTTS(language='ZH'); audio = tts.tts('你好世界')"

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
python -c "from infer_api.index_tts import IndexTTSTTS; tts = IndexTTSTTS(); print('Index-TTS API loaded successfully')"
python -c "from infer_api.melo_tts import MeloTTSTTS; tts = MeloTTSTTS(); print('MeloTTS API loaded successfully')"

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
- **CosyVoice**: Multi-language TTS from Alibaba with local GPU inference
- **Index-TTS**: Industrial-level controllable zero-shot TTS from IndexTeam
- **MeloTTS**: High-quality multi-lingual TTS from MyShell AI with built-in speakers

### Key Components

```
ZeroShotTTS/
├── infer.py                 # Main batch processing entry with GPU management
├── infer_api/               # Unified API layer
│   ├── baseTTS.py          # Abstract TTS interface (BaseTTS class)
│   ├── fishspeech_api.py   # Fish-Speech wrapper implementation
│   ├── gptSoVITS_tts.py    # GPT-SoVITS wrapper implementation
│   ├── cosyvoice_tts.py    # CosyVoice wrapper implementation
│   ├── index_tts.py        # Index-TTS wrapper implementation
│   └── melo_tts.py         # MeloTTS wrapper implementation
├── TTS_Model/              # Model implementations (git submodules)
│   ├── fish-speech/        # Transformer-based TTS
│   ├── GPT-SoVITS/         # GPT + SoVITS models
│   ├── CosyVoice/          # Multi-language TTS from Alibaba
│   ├── index-tts/          # Industrial-level zero-shot TTS
│   └── melo-tts/           # High-quality multi-lingual TTS
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
from infer_api.index_tts import IndexTTSTTS
from infer_api.melo_tts import MeloTTSTTS

# Initialize models
tts = FishSpeechTTS()  # or GPTSoVITSTTS() or CosyVoiceTTS() or IndexTTSTTS() or MeloTTSTTS()

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
- **Index-TTS**: `TTS_Model/index-tts/checkpoints/config.yaml`
- **MeloTTS**: Uses HuggingFace models, auto-downloaded to cache
- **Unified API**: Environment variables override defaults

### Model Locations
Models are stored in:
- **Fish-Speech**: `TTS_Model/fish-speech/checkpoints/`
- **GPT-SoVITS**: `TTS_Model/GPT-SoVITS/GPT_weights/` and `SoVITS_weights/`
- **CosyVoice**: Downloaded automatically or placed in `TTS_Model/CosyVoice/pretrained_models/`
- **Index-TTS**: Download from HuggingFace: `IndexTeam/IndexTTS-2` to `TTS_Model/index-tts/checkpoints/`
- **MeloTTS**: Auto-downloaded from HuggingFace: `myshell-ai/MeloTTS-*` to cache

### Supported Languages
- **Fish-Speech**: Chinese, English, Japanese, Korean, Cantonese
- **GPT-SoVITS**: Chinese, English, Japanese, Korean, Cantonese
- **CosyVoice**: Chinese, English, Japanese, Korean, Cantonese
- **Index-TTS**: Chinese, English (primary support)
- **MeloTTS**: English (US/UK/India/Australia), Chinese, Spanish, French, Japanese, Korean (built-in speakers)

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
- **Examples**:
  - CosyVoiceTTS uses direct `CosyVoice2()` initialization
  - IndexTTSTTS uses direct `IndexTTS2()` initialization

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

#### 7. Model-Specific Implementation Rules

**Universal Mode Standards:**
All TTS models should implement these standard modes:
- **zero_shot**: Use registered speaker for voice cloning
- **default**: Use model's default voice or first registered speaker
- **sft** (if supported): Use pre-trained speakers from the model

**MeloTTS Specific Rules:**
- **Built-in Speaker System**: Uses pre-defined speakers per language (EN-US, EN-BR, ZH, etc.)
- **Zero-shot Mode**: Stores reference audio for speaker adaptation guidance
- **Default Mode**: Uses language-specific built-in speakers
- **Low Memory**: Only 4GB+ GPU memory required
- **CPU Real-time**: Optimized for CPU inference
- **Multi-accent**: English supports US, UK, India, Australia accents

**Index-TTS Specific Rules:**
- **Simplified Modes**: Only support `zero_shot` and `default` modes (as per design requirement)
  - `zero_shot`: Use registered speaker for voice cloning
  - `default`: Use default reference audio or first registered speaker
- **Speaker Registration**: Store reference audio paths, use IndexTTS2.infer() directly
- **Audio Format**: Index-TTS expects 16kHz input, outputs 24kHz (resample to 16kHz)
- **Error Handling**: Validate speaker existence before inference

**Future Model Implementation Rule:**
When adding new TTS models, follow the Index-TTS pattern of simplified modes:
- Start with `zero_shot` and `default` modes only
- Add additional modes only when specific features require them
- Keep the interface simple and consistent with BaseTTS

```python
def tts_with_speaker(self, text, speaker_id=None, mode='zero_shot'):
    if mode == 'zero_shot':
        # Use registered speaker audio
        if speaker_id and speaker_id in self.speakers:
            spk_audio = self.speakers[speaker_id]['prompt_audio']
            result = self.index_tts.infer(spk_audio_prompt=spk_audio, text=text)
        else:
            raise ValueError("Speaker not found")
    elif mode == 'default':
        # Use default reference audio or first registered speaker
        default_audio = kwargs.get('default_spk_audio') or self._get_first_speaker()
        result = self.index_tts.infer(spk_audio_prompt=default_audio, text=text)

    # Process result and resample to 16kHz
    audio = self._process_inference_result(result)
    return audio
```

#### 8. BaseTTS Interface Compliance
All models must implement:
- `register_speaker()`: Speaker registration with audio/text pairs
- `tts()`: Basic synthesis with default speaker
- `tts_with_speaker()`: Synthesis with specific speaker
- `is_available()`: Model availability check
- `get_model_info()`: Model metadata and capabilities

### Code Implementation Standards Summary

| Model | Modes | Sample Rate | GPU Memory | Languages | Special Notes |
|-------|-------|-------------|------------|-----------|---------------|
| FishSpeech | zero_shot, sft | 16kHz output | 8GB+ | CN, EN, JP, KR, Yue | Transformer + VQGAN |
| GPT-SoVITS | zero_shot, sft | 16kHz output | 8GB+ | CN, EN, JP, KR, Yue | GPT + SoVITS |
| CosyVoice | zero_shot, cross_lingual, instruct, sft | 16kHz output | 6GB+ | CN, EN, JP, KR, Yue | Multi-language support |
| Index-TTS | **zero_shot, default only** | 16kHz output | 6GB+ | CN, EN | **Simplified modes, industrial-grade** |
| MeloTTS | **zero_shot, default only** | 16kHz output | 4GB+ | EN, ZH, ES, FR, JP, KR | **Built-in speakers, CPU real-time** |

## Unified Model Interface Architecture

### Design Philosophy

The ZeroShotTTS system implements a **unified interface architecture** that separates model-specific logic from common processing logic. This design follows several key principles:

#### 1. **DRY Principle (Don't Repeat Yourself)**
All common processing logic is centralized in `infer.py`, eliminating code duplication across model implementations.

#### 2. **Single Responsibility Principle**
- `infer.py`: Handles common processing pipeline (file I/O, batch processing, GPU management)
- Model wrappers: Handle model-specific logic (initialization, inference, speaker management)
- BaseTTS interface: Defines the contract all models must implement

#### 3. **Open/Closed Principle**
The system is open for extension (new models) but closed for modification (common logic).

### Architecture Benefits

#### Before Optimization (❌ Anti-pattern)
```python
# Repetitive if-elif structure - hard to maintain
if model_name == 'model1':
    # 40+ lines of duplicate logic
    model = Model1()
    speaker_id = model.register_speaker(audio, text)
    # ... same file processing logic
elif model_name == 'model2':
    # 40+ lines of duplicate logic
    model = Model2()
    speaker_id = model.register_speaker(audio, text)
    # ... same file processing logic
# ... repeated for each model
```

#### After Optimization (✅ Best Practice)
```python
# Unified interface - clean and maintainable
def run_tts_with_list_file(input_file, output_dir, model_name, speaker_audio=None, speaker_text=None):
    tts_model = create_tts_model(model_name)  # Factory pattern
    process_text_file_with_tts(input_file, output_dir, tts_model, speaker_audio, speaker_text)
```

### Model Mode Standards

#### Core Mode Requirements
All TTS models must implement these standard modes:

1. **zero_shot** (Required)
   - Uses registered speaker audio for voice cloning
   - Implementation: Check if speaker_id exists in registered speakers
   - Fallback: Use default speaker if no specific speaker provided

2. **default** (Required)
   - Uses model's built-in default voice or first registered speaker
   - Implementation: Use model's native default or first speaker in registry
   - Fallback: Use basic TTS without speaker reference

3. **sft** (Optional, if model supports)
   - Uses model's pre-trained speakers
   - Implementation: Use model's built-in speaker voices
   - Only implement if model has native SFT support

#### Mode Implementation Pattern
```python
def tts_with_speaker(self, text, speaker_id=None, mode='zero_shot'):
    if mode == 'zero_shot':
        if speaker_id and speaker_id in self.speakers:
            # Use registered speaker
            return self._synthesize_with_speaker(text, speaker_id)
        else:
            raise ValueError("Speaker not found for zero_shot mode")

    elif mode == 'default':
        if self.speakers:
            # Use first registered speaker as default
            first_speaker = next(iter(self.speakers.keys()))
            return self._synthesize_with_speaker(text, first_speaker)
        else:
            # Use model's built-in default
            return self._synthesize_default(text)

    elif mode == 'sft' and hasattr(self, 'list_available_spks'):
        # Use model's pre-trained speakers
        available_spks = self.list_available_spks()
        if available_spks:
            return self._synthesize_sft(text, available_spks[0])
        else:
            raise ValueError("No pre-trained speakers available")

    else:
        raise ValueError(f"Unsupported mode: {mode}")
```

### Adding New Models

To add a new TTS model, follow this checklist:

#### 1. **Repository Setup**
```bash
# Add as git submodule
git submodule add <repository-url> TTS_Model/new-model

# Update submodules
git submodule update --init --recursive
```

#### 2. **Wrapper Implementation**
Create `infer_api/new_model.py`:
```python
from .baseTTS import BaseTTS

class NewModelTTS(BaseTTS):
    def __init__(self, model_dir=None, device=None):
        super().__init__()
        # Model-specific initialization

    def register_speaker(self, prompt_audio, prompt_text, speaker_name=None):
        # Model-specific speaker registration
        return speaker_id

    def tts(self, text, **kwargs):
        # Model-specific inference without speaker
        return audio_array

    def tts_with_speaker(self, text, speaker_id, mode='zero_shot', **kwargs):
        # Model-specific inference with speaker
        if mode == 'zero_shot':
            # Use registered speaker
        elif mode == 'default':
            # Use default/first speaker
        return audio_array
```

#### 3. **Integration**
Update `infer.py`:
```python
def create_tts_model(model_name: str):
    model_map = {
        'existing-models': '...',
        'new-model': 'infer_api.new_model:NewModelTTS',  # Add this line
    }
    # ... rest of factory logic
```

**MeloTTS Integration Example:**
```python
def create_tts_model(model_name: str):
    model_map = {
        'fish-speech': 'infer_api.fishspeech_api:FishSpeechTTS',
        'gpt-sovits': 'infer_api.gptSoVITS_tts:GPTSoVITSTTS',
        'cosyvoice': 'infer_api.cosyvoice_tts:CosyVoiceTTS',
        'index-tts': 'infer_api.index_tts:IndexTTSTTS',
        'melo-tts': 'infer_api.melo_tts:MeloTTSTTS',  # Add this line
    }
    # ... rest of factory logic
```

**Update argument parser:**
```python
parser.add_argument('--model', choices=['fish-speech', 'gpt-sovits', 'cosyvoice', 'index-tts', 'melo-tts'],
                   default='fish-speech', help='使用的TTS模型')
```

#### 4. **Testing**
```bash
# Test model loading
python -c "from infer_api.new_model import NewModelTTS; m = NewModelTTS(); print('OK')"

# Test with infer.py
python infer.py --model new-model --input_dir ./texts --output_dir ./output
```

### Quality Assurance Checklist

When implementing a new TTS model wrapper:

- [ ] Implement `BaseTTS` interface completely
- [ ] Support `zero_shot` and `default` modes at minimum
- [ ] Output 16kHz audio consistently
- [ ] Use local GPU inference (no API calls)
- [ ] Handle errors gracefully with informative messages
- [ ] Follow direct import pattern (no try-except blocks)
- [ ] Add model to factory map in `create_tts_model()`
- [ ] Update CLAUDE.md with model-specific notes
- [ ] Test with both speaker cloning and default modes
- [ ] Verify batch processing works correctly

#### 1. Factory Pattern Implementation
All models use a unified factory pattern for instantiation:

```python
def create_tts_model(model_name: str):
    model_map = {
        'fish-speech': 'infer_api.fishspeech_api:FishSpeechTTS',
        'gpt-sovits': 'infer_api.gptSoVITS_tts:GPTSoVITSTTS',
        'cosyvoice': 'infer_api.cosyvoice_tts:CosyVoiceTTS',
        'index-tts': 'infer_api.index_tts:IndexTTSTTS',
        'melo-tts': 'infer_api.melo_tts:MeloTTSTTS',
    }
    module_path, class_name = model_map[model_name].split(':')
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)()
```

#### 2. Unified Processing Pipeline
All models follow the same processing pattern in `infer.py`:

```python
def process_text_file_with_tts(input_file, output_dir, tts_model, speaker_audio=None, speaker_text=None):
    # 1. Register speaker (unified logic)
    speaker_id = None
    if speaker_audio and speaker_text:
        speaker_id = tts_model.register_speaker(speaker_audio, speaker_text)

    # 2. Process text file (unified logic)
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                print(f"Processing line {i+1}: {line[:50]}...")

                try:
                    # 3. Unified audio synthesis logic
                    if speaker_id:
                        audio = tts_model.tts_with_speaker(line, speaker_id)
                    else:
                        audio = tts_model.tts(line)

                    # 4. Unified audio saving logic
                    output_file = os.path.join(output_dir, f"audio_{i+1:04d}.wav")
                    wavfile.write(output_file, 16000, (audio * 32767).astype('int16'))
                    print(f"Saved: {output_file}")

                except Exception as e:
                    print(f"Error processing line {i+1}: {e}")
```

#### 3. Model-Specific Logic Isolation
All model-specific logic is isolated in individual wrapper classes:

- **Model Initialization**: Each model handles its own initialization in `__init__()`
- **Speaker Processing**: Each model handles speaker registration in `register_speaker()`
- **Audio Synthesis**: Each model handles synthesis in `tts()` and `tts_with_speaker()`
- **Resource Management**: Each model handles its own GPU memory and model loading

#### 4. Adding New Models
To add a new TTS model:

1. **Create wrapper class** in `infer_api/` implementing `BaseTTS` interface
2. **Add to factory map** in `create_tts_model()` function
3. **Implement model-specific logic** in the wrapper class
4. **Follow unified standards** (16kHz output, local GPU inference, etc.)

**Example**:
```python
def create_tts_model(model_name: str):
    model_map = {
        'existing-models': '...',
        'new-model': 'infer_api.new_model:NewModelTTS',  # Add here
    }
    # ... rest of factory logic
```

#### 5. Benefits of Unified Architecture

- **DRY Principle**: No code duplication across model implementations
- **Maintainability**: Changes to common logic only need to be made once
- **Extensibility**: New models can be added with minimal code changes
- **Consistency**: All models follow the same interface and behavior patterns
- **Testability**: Unified testing approach for all models
- **Debugging**: Centralized error handling and logging

## Testing and Validation

### Unit Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific model tests
python -m pytest tests/test_fishspeech.py -v
python -m pytest tests/test_gptsovits.py -v
python -m pytest tests/test_cosyvoice.py -v
python -m pytest tests/test_index_tts.py -v
python -m pytest tests/test_melo_tts.py -v
```

### Integration Tests
```bash
# Test API endpoints
curl -X POST http://localhost:8080/register_speaker -F "audio=@examples/ref.wav" -F "text=Hello"
curl -X POST http://localhost:8080/tts -H "Content-Type: application/json" -d '{"text": "Test synthesis", "speaker_id": "test_speaker"}'

# Test batch processing
python infer.py --input_dir examples/texts --output_dir test_output --model fish-speech --max_workers 2
python infer.py --input_dir examples/texts --output_dir test_output --model index-tts --max_workers 2
python infer.py --input_dir examples/texts --output_dir test_output --model melo-tts --max_workers 2
```