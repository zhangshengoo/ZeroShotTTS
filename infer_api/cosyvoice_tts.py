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
import scipy.io.wavfile as wavfile
import librosa

from .baseTTS import BaseTTS

# 添加CosyVoice路径到系统路径
cosyvoice_path = Path(__file__).parent.parent / "TTS_Model" / "CosyVoice"
sys.path.insert(0, str(cosyvoice_path))
third_party_path = cosyvoice_path / "third_party" / "Matcha-TTS"
sys.path.insert(0, str(third_party_path))

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging


class CosyVoiceTTS(BaseTTS):
    """CosyVoice TTS封装类 - 本地GPU推理版本

    CosyVoice是阿里巴巴FunAudioLLM团队开发的多语言零样本TTS模型
    支持零样本、跨语言、情感控制等多种语音合成模式
    本版本使用本地GPU推理，无需API服务
    """

    def __init__(self,
                 model_dir: str = None,
                 model_version: str = "2.0",  # "1.0" or "2.0"
                 model_name: str = "cosyvoice",
                 device: str = None,
                 load_jit: bool = False,
                 load_trt: bool = False,
                 load_vllm: bool = False,
                 fp16: bool = False):
        super().__init__()

        self.model_name = model_name
        self.model_version = model_version
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.speakers: Dict[str, Dict[str, Any]] = {}
        self.cosyvoice = None
        self.model_sample_rate = None  # 模型的实际采样率
        self.target_sample_rate = 16000  # 统一输出采样率

        # CosyVoice支持的语音模式
        self.modes = {
            'zero_shot': '零样本语音克隆',
            'instruct': '指令控制语音合成',
            'sft': '有监督微调语音合成'
        }

        # 初始化模型
        self._initialize_model(model_dir)

    def _initialize_model(self, model_dir: str = None):
        """初始化CosyVoice模型"""
        try:
            # 设置默认模型路径
            if model_dir is None:
                model_dir = self._get_default_model_path()

            # 检查模型是否存在
            if not os.path.exists(model_dir):
                # 尝试从ModelScope下载
                from modelscope import snapshot_download
                model_id = self._get_modelscope_id()
                model_dir = snapshot_download(model_id, local_dir=model_dir)

            # 初始化模型
            if self.model_version == "2.0":
                self.cosyvoice = CosyVoice2(
                    model_dir,
                    load_jit=False,
                    load_trt=False,
                    load_vllm=False,
                    fp16=False
                )
            else:
                self.cosyvoice = CosyVoice(
                    model_dir,
                    load_jit=False,
                    load_trt=False,
                    fp16=False
                )

            self.model_sample_rate = self.cosyvoice.sample_rate
            logging.info(f"CosyVoice {self.model_version} model loaded successfully from {model_dir}, sample rate: {self.model_sample_rate}Hz")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize CosyVoice model: {str(e)}")

    def _get_default_model_path(self) -> str:
        """获取默认模型路径"""
        base_path = Path(__file__).parent.parent / "TTS_Model" / "CosyVoice"
        if self.model_version == "2.0":
            return str(base_path / "pretrained_models" / "CosyVoice2-0.5B")
        else:
            return str(base_path / "pretrained_models" / "CosyVoice-300M")

    def _get_modelscope_id(self) -> str:
        """获取ModelScope模型ID"""
        if self.model_version == "2.0":
            return "iic/CosyVoice2-0.5B"
        else:
            return "iic/CosyVoice-300M"

    def is_available(self) -> bool:
        """检查CosyVoice是否可用"""
        return self.cosyvoice is not None

    def register_speaker(self,
                        prompt_audio: Union[str, np.ndarray],
                        prompt_text: str,
                        speaker_name: Optional[str] = None) -> str:
        """注册说话人 - 使用CosyVoice的save zero_shot spk方式

        Args:
            prompt_audio: 参考音频文件路径或音频数据
            prompt_text: 参考文本内容
            speaker_name: 说话人名称，自动生成如果不提供

        Returns:
            speaker_id: 说话人唯一标识符
        """
        if speaker_name is None:
            speaker_name = f"speaker_{len(self.speakers) + 1}"

        speaker_id = f"cosyvoice_{speaker_name}"

        try:
            # 处理音频数据
            if isinstance(prompt_audio, str):
                # 文件路径
                prompt_speech = load_wav(prompt_audio, 16000)
                audio_path = prompt_audio
            else:
                # numpy数组，保存为临时文件
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    self._save_audio_to_file(prompt_audio, f.name, 16000)
                    prompt_speech = load_wav(f.name, 16000)
                    audio_path = f.name

            # 使用CosyVoice的save zero_shot spk方式注册说话人
            # add_zero_shot_spk 方法用于注册说话人
            success = self.cosyvoice.add_zero_shot_spk(
                prompt_text=prompt_text,
                prompt_speech_16k=prompt_speech,
                zero_shot_spk_id=speaker_id
            )

            if success:
                # 保存说话人信息到模型（save_spkinfo）
                self.cosyvoice.save_spkinfo()

                self.speakers[speaker_id] = {
                    'prompt_audio': audio_path,
                    'prompt_text': prompt_text,
                    'speaker_name': speaker_name,
                    'prompt_speech': prompt_speech  # 保存音频数据避免重复加载
                }
                logging.info(f"Speaker {speaker_name} registered successfully with ID: {speaker_id}")
                return speaker_id
            else:
                raise Exception("Failed to register speaker in CosyVoice")

        except Exception as e:
            raise Exception(f"CosyVoice registration error: {str(e)}")
        finally:
            # 清理临时文件
            if isinstance(prompt_audio, np.ndarray) and os.path.exists(audio_path) and audio_path != prompt_audio:
                os.unlink(audio_path)

    def tts(self, text: str, **kwargs) -> np.ndarray:
        """基础TTS合成 - 使用默认说话人或SFT模式"""
        return self.tts_with_speaker(text, speaker_id=None, **kwargs)

    def tts_with_speaker(self,
                        text: str,
                        speaker_id: Optional[str] = None,
                        mode: str = 'zero_shot',
                        speed: float = 1.0,
                        **kwargs) -> np.ndarray:
        """使用指定说话人进行语音合成 - 使用save zero_shot spk方式

        Args:
            text: 要合成的文本
            speaker_id: 说话人ID，None表示使用默认声音
            mode: 合成模式，可选: zero_shot, cross_lingual, instruct, sft
            speed: 语速，默认1.0

        Returns:
            合成的音频数据 (numpy数组)
        """
        try:
            # 设置推理参数
            inference_kwargs = {
                'stream': False,
                'speed': speed,
                'text_frontend': kwargs.get('text_frontend', True)
            }

            if mode == 'sft':
                # 使用预训练说话人
                if speaker_id and speaker_id in self.cosyvoice.list_available_spks():
                    spk_id = speaker_id
                else:
                    # 使用默认说话人
                    available_spks = self.cosyvoice.list_available_spks()
                    spk_id = available_spks[0] if available_spks else "中文女"

                outputs = self.cosyvoice.inference_sft(
                    tts_text=text,
                    spk_id=spk_id,
                    **inference_kwargs
                )

            elif mode == 'zero_shot':
                # 零样本语音克隆 - 使用save zero_shot spk方式
                if speaker_id and speaker_id in self.speakers:
                    # 使用已注册的说话人，通过zero_shot_spk_id参数指定
                    outputs = self.cosyvoice.inference_zero_shot(
                        tts_text=text,
                        zero_shot_spk_id=speaker_id,  # 使用已注册的说话人ID
                        **inference_kwargs
                    )
                else:
                    raise ValueError(f"Speaker {speaker_id} not found. Please register speaker first.")

            else:
                raise ValueError(f"Unsupported mode: {mode}")

            # 提取音频数据
            audio_chunks = []
            for output in outputs:
                audio_tensor = output['tts_speech']
                # 转换为numpy数组
                if isinstance(audio_tensor, torch.Tensor):
                    audio_np = audio_tensor.squeeze().cpu().numpy()
                else:
                    audio_np = audio_tensor.squeeze()
                audio_chunks.append(audio_np)

            # 合并音频片段
            if audio_chunks:
                final_audio = np.concatenate(audio_chunks)

                # 重采样到16kHz（如果需要）
                if self.model_sample_rate != self.target_sample_rate:
                    final_audio = librosa.resample(
                        final_audio,
                        orig_sr=self.model_sample_rate,
                        target_sr=self.target_sample_rate
                    )

                return final_audio.astype(np.float32)
            else:
                raise Exception("No audio data generated")

        except Exception as e:
            raise Exception(f"CosyVoice TTS error: {str(e)}")

    def _process_speaker_embedding(self, prompt_audio: np.ndarray, prompt_text: str) -> Dict[str, Any]:
        """处理参考音频和文本以生成说话人嵌入 - CosyVoice简化实现

        直接返回原始的参考音频和文本数据，让CosyVoice的add_zero_shot_spk方法
        自行处理特征提取，避免复杂的嵌入计算逻辑。

        Args:
            prompt_audio: 参考音频数据 (numpy数组, 16kHz)
            prompt_text: 参考文本内容

        Returns:
            简化的说话人特征字典，包含原始音频和文本
        """
        try:
            # 直接返回原始数据，让CosyVoice自行处理特征提取
            speaker_embedding = {
                'prompt_audio': prompt_audio,      # 原始参考音频
                'prompt_text': prompt_text,        # 原始参考文本
                'sample_rate': 16000,              # 采样率
                'model_version': self.model_version,
                'language': self.language if hasattr(self, 'language') else 'auto'
            }

            logging.info(f"CosyVoice speaker embedding simplified: returning raw audio and text data")
            return speaker_embedding

        except Exception as e:
            logging.error(f"Failed to process CosyVoice speaker embedding: {str(e)}")
            raise Exception(f"CosyVoice speaker embedding error: {str(e)}")

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
            'name': 'CosyVoice',
            'version': self.model_version,
            'description': '阿里巴巴FunAudioLLM团队开发的多语言零样本TTS（本地GPU推理版）',
            'supported_languages': ['zh', 'en', 'ja', 'ko'],
            'supported_modes': list(self.modes.keys()),
            'device': self.device,
            'model_sample_rate': self.model_sample_rate,
            'target_sample_rate': self.target_sample_rate,
            'requirements': {
                'gpu_memory': '6GB+',
                'python': '>=3.8',
                'dependencies': ['torch', 'torchaudio', 'transformers']
            }
        }

    def list_speakers(self) -> Dict[str, Dict[str, Any]]:
        """列出已注册的说话人"""
        return self.speakers.copy()

    def list_available_spks(self) -> list:
        """列出模型内置的可用说话人"""
        if self.cosyvoice:
            return self.cosyvoice.list_available_spks()
        return []

    def remove_speaker(self, speaker_id: str) -> bool:
        """移除说话人"""
        if speaker_id in self.speakers:
            del self.speakers[speaker_id]
            return True
        return False