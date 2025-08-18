from typing import Optional, Dict, Any, Union
import os
import json
import requests
import numpy as np
from pathlib import Path
import tempfile
import wave

from .baseTTS import BaseTTS


class CosyVoiceTTS(BaseTTS):
    """CosyVoice TTS封装类
    
    CosyVoice是阿里巴巴FunAudioLLM团队开发的多语言零样本TTS模型
    支持零样本、跨语言、情感控制等多种语音合成模式
    """
    
    def __init__(self, 
                 api_url: str = "http://localhost:50000",
                 model_name: str = "cosyvoice"):
        super().__init__()
        self.api_url = api_url.rstrip('/')
        self.model_name = model_name
        self.speakers: Dict[str, Dict[str, Any]] = {}
        
        # CosyVoice支持的语音模式
        self.modes = {
            'zero_shot': '零样本语音克隆',
            'cross_lingual': '跨语言语音克隆',
            'instruct': '指令控制语音合成',
            'sft': '有监督微调语音合成'
        }
        
    def is_available(self) -> bool:
        """检查CosyVoice服务是否可用"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def register_speaker(self, 
                        prompt_audio: Union[str, np.ndarray], 
                        prompt_text: str,
                        speaker_name: Optional[str] = None) -> str:
        """注册说话人
        
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
        
        # 处理音频数据
        if isinstance(prompt_audio, str):
            audio_path = prompt_audio
        else:
            # 保存临时音频文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                self._save_audio_to_file(prompt_audio, f.name)
                audio_path = f.name
        
        # 上传到CosyVoice服务
        try:
            with open(audio_path, 'rb') as f:
                files = {'audio': f}
                data = {'text': prompt_text}
                
                response = requests.post(
                    f"{self.api_url}/upload_speaker",
                    files=files,
                    data=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    speaker_token = result.get('speaker_token', speaker_id)
                    
                    self.speakers[speaker_id] = {
                        'speaker_token': speaker_token,
                        'prompt_audio': audio_path,
                        'prompt_text': prompt_text,
                        'speaker_name': speaker_name
                    }
                    
                    return speaker_id
                else:
                    raise Exception(f"Failed to register speaker: {response.text}")
                    
        except Exception as e:
            raise Exception(f"CosyVoice registration error: {str(e)}")
        finally:
            # 清理临时文件
            if isinstance(prompt_audio, np.ndarray) and os.path.exists(audio_path):
                os.unlink(audio_path)
    
    def tts(self, text: str, **kwargs) -> np.ndarray:
        """基础TTS合成"""
        return self.tts_with_speaker(text, speaker_id=None, **kwargs)
    
    def tts_with_speaker(self, 
                        text: str, 
                        speaker_id: Optional[str] = None,
                        mode: str = 'zero_shot',
                        **kwargs) -> np.ndarray:
        """使用指定说话人进行语音合成
        
        Args:
            text: 要合成的文本
            speaker_id: 说话人ID，None表示使用默认声音
            mode: 合成模式，可选: zero_shot, cross_lingual, instruct, sft
            
        Returns:
            合成的音频数据 (numpy数组)
        """
        
        # 构建请求数据
        data = {
            'text': text,
            'mode': mode,
            'speed': kwargs.get('speed', 1.0),
            'volume': kwargs.get('volume', 1.0),
            'pitch': kwargs.get('pitch', 0.0),
        }
        
        # 添加说话人信息
        if speaker_id and speaker_id in self.speakers:
            data['speaker_token'] = self.speakers[speaker_id]['speaker_token']
        elif speaker_id:
            data['speaker_token'] = speaker_id
        
        # 指令模式特殊参数
        if mode == 'instruct' and 'instruct_text' in kwargs:
            data['instruct_text'] = kwargs['instruct_text']
        
        try:
            response = requests.post(
                f"{self.api_url}/inference",
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # 处理返回的音频数据
                if 'audio' in result:
                    # 假设返回的是base64编码的音频
                    import base64
                    audio_data = base64.b64decode(result['audio'])
                    
                    # 转换为numpy数组
                    import io
                    with wave.open(io.BytesIO(audio_data), 'rb') as wav_file:
                        audio_np = np.frombuffer(wav_file.readframes(wav_file.getnframes()), 
                                               dtype=np.int16)
                        return audio_np.astype(np.float32) / 32768.0
                        
                elif 'audio_url' in result:
                    # 下载音频文件
                    audio_response = requests.get(result['audio_url'])
                    if audio_response.status_code == 200:
                        import io
                        with wave.open(io.BytesIO(audio_response.content), 'rb') as wav_file:
                            audio_np = np.frombuffer(wav_file.readframes(wav_file.getnframes()), 
                                                   dtype=np.int16)
                            return audio_np.astype(np.float32) / 32768.0
                            
                raise Exception("No audio data in response")
                
            else:
                raise Exception(f"TTS request failed: {response.text}")
                
        except Exception as e:
            raise Exception(f"CosyVoice TTS error: {str(e)}")
    
    def _save_audio_to_file(self, audio_data: np.ndarray, file_path: str, sample_rate: int = 22050):
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
            'version': '1.0',
            'description': '阿里巴巴FunAudioLLM团队开发的多语言零样本TTS',
            'supported_languages': ['zh', 'en', 'ja', 'ko', 'yue'],
            'supported_modes': list(self.modes.keys()),
            'api_url': self.api_url,
            'requirements': {
                'gpu_memory': '8GB+',
                'python': '>=3.8',
                'dependencies': ['torch', 'transformers', 'librosa']
            }
        }
    
    def list_speakers(self) -> Dict[str, Dict[str, Any]]:
        """列出已注册的说话人"""
        return self.speakers.copy()
    
    def remove_speaker(self, speaker_id: str) -> bool:
        """移除说话人"""
        if speaker_id in self.speakers:
            del self.speakers[speaker_id]
            return True
        return False