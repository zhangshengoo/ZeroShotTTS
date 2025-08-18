# ZeroShotTTS - 开源零样本语音合成模型对比平台

简体中文 | [English](#english-version)

本项目集合了当前主流的开源零样本语音合成(Zero-Shot TTS)模型，提供统一的推理接口和对比工具，帮助用户快速评估和选择适合的TTS解决方案。

## 🎯 项目特色

- **统一接口**: 统一的API设计，简化不同模型的使用
- **零样本克隆**: 仅需3-10秒音频即可克隆任意声音
- **多模型支持**: 集成Fish-Speech和GPT-SoVITS两大主流模型
- **批量处理**: 支持大规模文本批量语音合成
- **多语言**: 支持中文、英文、日语、韩语、粤语
- **GPU优化**: 自动GPU管理和负载均衡


## 📦 快速开始

### 环境准备

```bash
# 克隆项目
git clone https://github.com/your-repo/ZeroShotTTS.git
cd ZeroShotTTS

# 初始化子模块
git submodule update --init --recursive

# 安装依赖
pip install -e .
```

### 基础使用
```bash
# 使用Fish-Speech批量处理
python infer.py \
  --input_dir ./test_texts \
  --output_dir ./output/fish-speech \
  --model fish-speech \
  --speaker_audio ./examples/ref.wav \
  --speaker_text "参考文本"

# 使用GPT-SoVITS批量处理  
python infer.py \
  --input_dir ./test_texts \
  --output_dir ./output/gpt-sovits \
  --model gpt-sovits \
  --speaker_audio ./examples/ref.wav \
  --speaker_text "参考文本"
```

## 🔧 统一API使用

### Python API

```python
from infer_api.fishspeech_api import FishSpeechTTS
from infer_api.gptSoVITS_tts import GPTSoVITSTTS

# 初始化模型
fish_tts = FishSpeechTTS()
gpt_tts = GPTSoVITSTTS()

# 注册说话人
speaker_id = fish_tts.register_speaker(
    prompt_audio="./examples/ref.wav",
    prompt_text="这是参考文本"
)

# 语音合成
audio = fish_tts.tts_with_speaker(
    text="需要合成的文本",
    speaker_id=speaker_id
)
```


## 📁 项目结构

```
ZeroShotTTS/
├── infer.py                 # 批量推理主入口
├── infer_api/              # 统一API接口
│   ├── baseTTS.py         # 抽象基类
│   ├── fishspeech_api.py  # Fish-Speech封装
│   └── gptSoVITS_tts.py   # GPT-SoVITS封装
├── TTS_Model/             # 模型实现(子模块)
│   ├── fish-speech/       # Fish-Speech模型
│   └── GPT-SoVITS/        # GPT-SoVITS模型
├── examples/              # 示例文件
├── benchmark/            # 性能测试
├── scripts/              # 实用脚本
└── docs/                 # 文档资料
```

## 🔧 常见问题

### 音频格式问题
- 推荐使用 16kHz 单声道 WAV 格式
- 参考音频时长: 3-10秒
- 避免背景噪音

### 子模块更新
```bash
git submodule update --remote --merge
```

## 📋 持续更新的TTS模型集成

### ✅ 当前已集成
- **[Fish-Speech](https://github.com/fishaudio/fish-speech)** (Transformer + VQGAN) - 由FishAudio团队开发的多语言零样本TTS
- **[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)** (GPT + SoVITS) - 由RVC-Boss开发的强大的零样本语音克隆工具

### 🚧 计划中集成
- **[CosyVoice](https://github.com/FunAudioLLM/CosyVoice)** - 阿里巴巴FunAudioLLM团队开发的多语言零样本TTS
- **[MegaTTS3](https://github.com/bytedance/MegaTTS3)** - 字节跳动最新开源的零样本语音合成模型

---

## English Version

# ZeroShotTTS - Open Source Zero-Shot TTS Model Comparison Platform

This project integrates mainstream open-source zero-shot text-to-speech models, providing unified inference interfaces and comparison tools to help users quickly evaluate and select appropriate TTS solutions.

## 🎯 Features

- **Unified Interface**: Consistent API design across different models
- **Zero-shot Cloning**: Clone any voice with just 3-10 seconds of audio
- **Multi-model Support**: Integrated Fish-Speech and GPT-SoVITS models
- **Batch Processing**: Large-scale text-to-speech synthesis
- **Multi-language**: Chinese, English, Japanese, Korean, Cantonese
- **GPU Optimization**: Automatic GPU management and load balancing

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/your-repo/ZeroShotTTS.git
cd ZeroShotTTS
git submodule update --init --recursive
pip install -e .
```

### Basic Usage
```bash
# Fish-Speech inference
fishspeech-infer --text "Hello world" --reference-audio ref.wav --reference-text "Hello"

# GPT-SoVITS API
python TTS_Model/GPT-SoVITS/api.py -dr ref.wav -dt "reference text" -dl zh

# Batch comparison
python infer.py --input_dir ./texts --output_dir ./output --model fish-speech
```