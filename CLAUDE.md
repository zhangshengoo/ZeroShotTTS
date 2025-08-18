# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start Commands

### Core Development Commands
```bash
# Install dependencies
pip install -e .

# Run Fish-Speech inference
fishspeech-infer --text "Hello world" --reference-audio ref.wav --reference-text "Hello"

# Run GPT-SoVITS API
python TTS_Model/GPT-SoVITS/api.py -dr ref.wav -dt "reference text" -dl zh

# Batch processing
python infer.py --input_dir ./texts --output_dir ./output --model fish-speech

# Start Fish-Speech API server
python -m tools.api_server --llama-checkpoint path/to/checkpoint
```

### Testing Commands
```bash
# Test unified API
python -c "from infer_api.fishspeech_api import FishSpeechTTS; tts = FishSpeechTTS(); print('API loaded successfully')"

# Test speaker registration
curl -X POST http://localhost:8080/register_speaker -F "audio=@ref.wav" -F "text=Hello"

# Test TTS synthesis
curl -X POST http://localhost:8080/tts -H "Content-Type: application/json" -d '{"text": "Hello world", "speaker_id": "speaker_1"}'
```

## Architecture Overview

ZeroShotTTS is a unified text-to-speech system supporting **Fish-Speech** and **GPT-SoVITS** models:

- **Fish-Speech**: Transformer-based TTS with Llama encoder + VQGAN decoder
- **GPT-SoVITS**: GPT-based prosody modeling + SoVITS audio synthesis

### Key Components

```
ZeroShotTTS/
├── infer.py                 # Main batch processing entry
├── infer_api/               # Unified API layer
│   ├── baseTTS.py          # Abstract TTS interface
│   ├── fishspeech_api.py   # Fish-Speech wrapper
│   └── gptSoVITS_tts.py    # GPT-SoVITS wrapper
└── TTS_Model/              # Model implementations (git submodules)
    ├── fish-speech/        # Transformer-based TTS
    └── GPT-SoVITS/         # GPT + SoVITS models
```

## Development Patterns

### Adding New TTS Models
1. Create wrapper in `infer_api/` implementing `BaseTTS` interface
2. Add model to `TTS_Model/` as submodule
3. Update `infer.py` for batch processing support

### Speaker Registration Flow
```python
from infer_api.fishspeech_api import FishSpeechTTS
tts = FishSpeechTTS()
speaker_id = tts.register_speaker(prompt_audio, prompt_text)
audio = tts.tts_with_speaker("Hello", speaker_id)
```

### GPU Management
- Automatic GPU selection based on memory availability
- Multi-GPU batch processing via `infer.py`
- GPU memory monitoring for model loading

## Configuration Files

- **Fish-Speech**: `TTS_Model/fish-speech/configs/` (YAML configs)
- **GPT-SoVITS**: `TTS_Model/GPT-SoVITS/config.py`
- **Unified API**: Environment variables override defaults

## Model Locations

Models are stored in:
- Fish-Speech: `TTS_Model/fish-speech/checkpoints/`
- GPT-SoVITS: `TTS_Model/GPT-SoVITS/GPT_weights/` and `SoVITS_weights/`

## Common Issues

- **GPU OOM**: Check `nvidia-smi`, models need 8GB+ VRAM
- **Submodule issues**: Run `git submodule update --init --recursive`
- **Audio format**: Use 16kHz mono WAV for best results
- **Speaker audio**: 3-10 seconds recommended for voice cloning