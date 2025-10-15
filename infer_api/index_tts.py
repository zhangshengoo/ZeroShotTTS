from typing import Optional, Dict, Any, Union
import os
import sys
import json
import numpy as np
from pathlib import Path
import tempfile
import wave
import torch
import torchaudio
import librosa
import scipy.io.wavfile as wavfile

from .baseTTS import BaseTTS

# 添加Index-TTS路径到系统路径
indextts_path = Path(__file__).parent.parent / "TTS_Model" / "index-tts"
sys.path.insert(0, str(indextts_path))

from indextts.infer_v2 import IndexTTS2


class IndexTTSTTS(BaseTTS):
    """Index-TTS TTS封装类 - 本地GPU推理版本

    Index-TTS是IndexTeam开发的工业级可控零样本文本到语音合成系统
    支持零样本语音克隆、情感控制、高效推理等多种功能
    本版本使用本地GPU推理，无需API服务
    """

    def __init__(self,
                 model_dir: str = None,
                 model_name: str = "index-tts",
                 device: str = None,
                 use_fp16: bool = False,
                 use_cuda_kernel: bool = False,
                 use_deepspeed: bool = False):
        super().__init__()

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.speakers: Dict[str, Dict[str, Any]] = {}
        self.index_tts = None
        self.model_sample_rate = None  # 模型的实际采样率
        self.target_sample_rate = 16000  # 统一输出采样率

        # Index-TTS支持的语音模式 - 简化为两种模式
        self.modes = {
            'zero_shot': '零样本语音克隆',
            'default': '默认参考音频'
        }

        # 模型配置
        self.use_fp16 = use_fp16
        self.use_cuda_kernel = use_cuda_kernel
        self.use_deepspeed = use_deepspeed

        # 初始化模型
        self._initialize_model(model_dir)

    def _initialize_model(self, model_dir: str = None):
        """初始化Index-TTS模型"""
        try:
            # 设置默认模型路径
            if model_dir is None:
                model_dir = self._get_default_model_path()

            # 检查模型是否存在
            if not os.path.exists(model_dir):
                logging.warning(f"Model directory {model_dir} not found. Please download models first.")
                logging.info("Please download Index-TTS models from HuggingFace: IndexTeam/IndexTTS-2")

            # 设置配置文件和模型目录
            cfg_path = os.path.join(model_dir, "config.yaml")

            if not os.path.exists(cfg_path):
                raise FileNotFoundError(f"Config file not found: {cfg_path}")

            # 初始化模型
            self.index_tts = IndexTTS2(
                cfg_path=cfg_path,
                model_dir=model_dir,
                use_fp16=self.use_fp16,
                use_cuda_kernel=self.use_cuda_kernel,
                use_deepspeed=self.use_deepspeed,
                device=self.device
            )

            # 获取模型采样率（Index-TTS通常是24kHz）
            self.model_sample_rate = 24000  # Index-TTS默认采样率
            logging.info(f"Index-TTS model loaded successfully from {model_dir}, sample rate: {self.model_sample_rate}Hz")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Index-TTS model: {str(e)}")

    def _get_default_model_path(self) -> str:
        """获取默认模型路径"""
        base_path = Path(__file__).parent.parent / "TTS_Model" / "index-tts"
        return str(base_path / "checkpoints")

    def is_available(self) -> bool:
        """检查Index-TTS是否可用"""
        return self.index_tts is not None

    def register_speaker(self,
                        prompt_audio: Union[str, np.ndarray],
                        prompt_text: str,
                        speaker_name: Optional[str] = None) -> str:
        """注册说话人 - Index-TTS使用参考音频方式

        Args:
            prompt_audio: 参考音频文件路径或音频数据
            prompt_text: 参考文本内容（Index-TTS中可选）
            speaker_name: 说话人名称，自动生成如果不提供

        Returns:
            speaker_id: 说话人唯一标识符
        """
        if speaker_name is None:
            speaker_name = f"speaker_{len(self.speakers) + 1}"

        speaker_id = f"indextts_{speaker_name}"

        try:
            # 处理音频数据
            if isinstance(prompt_audio, str):
                # 文件路径
                audio_path = prompt_audio
                # 验证文件存在
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
            else:
                # numpy数组，保存为临时文件
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    self._save_audio_to_file(prompt_audio, f.name, 16000)
                    audio_path = f.name

            # Index-TTS使用参考音频文件路径进行语音克隆
            # 保存说话人信息供后续使用
            self.speakers[speaker_id] = {
                'prompt_audio': audio_path,
                'prompt_text': prompt_text,
                'speaker_name': speaker_name
            }

            logging.info(f"Speaker {speaker_name} registered successfully with ID: {speaker_id}")
            return speaker_id

        except Exception as e:
            raise Exception(f"Index-TTS registration error: {str(e)}")
        finally:
            # 清理临时文件
            if isinstance(prompt_audio, np.ndarray) and os.path.exists(audio_path) and audio_path != prompt_audio:
                os.unlink(audio_path)

    def tts(self, text: str, **kwargs) -> np.ndarray:
        """基础TTS合成 - 使用默认声音"""
        return self.tts_with_speaker(text, speaker_id=None, **kwargs)

    def tts_with_speaker(self,
                        text: str,
                        speaker_id: Optional[str] = None,
                        mode: str = 'zero_shot',
                        speed: float = 1.0,
                        **kwargs) -> np.ndarray:
        """使用指定说话人进行语音合成 - 简化为两种模式

        Args:
            text: 要合成的文本
            speaker_id: 说话人ID，None表示使用默认声音
            mode: 合成模式，可选: zero_shot, default
                - zero_shot: 使用已注册的说话人进行语音克隆
                - default: 使用默认参考音频
            speed: 语速控制（Index-TTS通过其他参数控制）

        Returns:
            合成的音频数据 (numpy数组)
        """
        try:
            # 设置推理参数
            inference_kwargs = {
                'verbose': kwargs.get('verbose', False),
                'interval_silence': kwargs.get('interval_silence', 200),
                'max_text_tokens_per_segment': kwargs.get('max_text_tokens_per_segment', 120)
            }

            if mode == 'zero_shot':
                # 零样本语音克隆 - 使用已注册的说话人
                if speaker_id and speaker_id in self.speakers:
                    speaker_info = self.speakers[speaker_id]
                    spk_audio_prompt = speaker_info['prompt_audio']

                    # 使用Index-TTS的infer方法进行语音合成
                    result = self.index_tts.infer(
                        spk_audio_prompt=spk_audio_prompt,
                        text=text,
                        output_path=None,  # 不保存到文件，返回音频数据
                        **inference_kwargs
                    )

                    if result is None:
                        raise Exception("Index-TTS inference returned None")

                    # 提取音频数据
                    if isinstance(result, list) and len(result) > 0:
                        audio_data = result[0]  # 获取第一个结果
                    else:
                        audio_data = result

                    # 处理音频数据格式
                    if isinstance(audio_data, torch.Tensor):
                        audio_np = audio_data.squeeze().cpu().numpy()
                    elif isinstance(audio_data, np.ndarray):
                        audio_np = audio_data.squeeze()
                    else:
                        raise Exception(f"Unsupported audio data type: {type(audio_data)}")

                    # 重采样到16kHz（如果需要）
                    if self.model_sample_rate != self.target_sample_rate:
                        audio_np = librosa.resample(
                            audio_np,
                            orig_sr=self.model_sample_rate,
                            target_sr=self.target_sample_rate
                        )

                    return audio_np.astype(np.float32)

                else:
                    raise ValueError(f"Speaker {speaker_id} not found. Please register speaker first.")

            elif mode == 'default':
                # 默认参考音频模式 - 使用默认的参考音频文件
                default_spk_audio = kwargs.get('default_spk_audio', None)

                if default_spk_audio and os.path.exists(default_spk_audio):
                    # 使用提供的默认参考音频
                    result = self.index_tts.infer(
                        spk_audio_prompt=default_spk_audio,
                        text=text,
                        output_path=None,
                        **inference_kwargs
                    )
                elif speaker_id and speaker_id in self.speakers:
                    # 如果没有提供默认音频，但有已注册的说话人，使用第一个说话人
                    speaker_info = self.speakers[speaker_id]
                    spk_audio_prompt = speaker_info['prompt_audio']

                    result = self.index_tts.infer(
                        spk_audio_prompt=spk_audio_prompt,
                        text=text,
                        output_path=None,
                        **inference_kwargs
                    )
                else:
                    raise ValueError("Default mode requires either: 1) default_spk_audio parameter with valid audio file path, or 2) registered speaker")

                if result is None:
                    raise Exception("Index-TTS inference returned None")

                # 处理音频数据（与zero_shot模式相同）
                if isinstance(result, list) and len(result) > 0:
                    audio_data = result[0]
                else:
                    audio_data = result

                if isinstance(audio_data, torch.Tensor):
                    audio_np = audio_data.squeeze().cpu().numpy()
                elif isinstance(audio_data, np.ndarray):
                    audio_np = audio_data.squeeze()
                else:
                    raise Exception(f"Unsupported audio data type: {type(audio_data)}")

                # 重采样到16kHz（如果需要）
                if self.model_sample_rate != self.target_sample_rate:
                    audio_np = librosa.resample(
                        audio_np,
                        orig_sr=self.model_sample_rate,
                        target_sr=self.target_sample_rate
                    )

                return audio_np.astype(np.float32)

            else:
                raise ValueError(f"Unsupported mode: {mode}. Use 'zero_shot' or 'default'")

        except Exception as e:
            raise Exception(f"Index-TTS TTS error: {str(e)}")

    def _save_audio_to_file(self, audio_data: np.ndarray, file_path: str, sample_rate: int = 16000):
        """将音频数据保存为WAV文件"""
        audio_int16 = (audio_data * 32767).astype(np.int16)

        with wave.open(file_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16位
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'name': 'Index-TTS',
            'version': '2.0',
            'description': 'IndexTeam开发的工业级可控零样本文本到语音合成系统（本地GPU推理版）',
            'supported_languages': ['zh', 'en'],  # 主要支持中文和英文
            'supported_modes': list(self.modes.keys()),
            'device': self.device,
            'model_sample_rate': self.model_sample_rate,
            'target_sample_rate': self.target_sample_rate,
            'requirements': {
                'gpu_memory': '6GB+',
                'python': '>=3.10',
                'dependencies': ['torch', 'torchaudio', 'librosa', 'transformers']
            }
        }

    def list_speakers(self) -> Dict[str, Dict[str, Any]]:
        """列出已注册的说话人"""
        return self.speakers.copy()

    def list_available_spks(self) -> list:
        """列出模型内置的可用说话人"""
        # Index-TTS主要通过参考音频进行语音克隆，没有固定的说话人列表
        return []

    def remove_speaker(self, speaker_id: str) -> bool:
        """移除说话人"""
        if speaker_id in self.speakers:
            del self.speakers[speaker_id]
            return True
        return False