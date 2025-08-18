import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import soundfile as sf
from loguru import logger

from .baseTTS import BaseTTS
from .fishspeech.infer_text2semantic import text2semantic
from .fishspeech.infer_vqgan import vqgan_encoder, vqgan_decoder


class FishSpeechTTS(BaseTTS):
    def __init__(
        self,
        llama_checkpoint_path: Union[str, Path] = "checkpoints/fish-speech-1.5",
        decoder_checkpoint_path: Union[str, Path] = "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
        decoder_config_name: str = "firefly_gan_vq",
        device: str = "cuda",
        compile: bool = False,
        half: bool = False,
        max_new_tokens: int = 0,
        top_p: float = 0.7,
        repetition_penalty: float = 1.2,
        temperature: float = 0.7,
        chunk_length: int = 100,
        iterative_prompt: bool = True,
    ):
        """
        初始化 FishSpeech TTS 系统
        
        Args:
            llama_checkpoint_path: Text2Semantic 模型检查点路径
            decoder_checkpoint_path: VQGAN 解码器检查点路径
            decoder_config_name: VQGAN 配置名称
            device: 运行设备
            compile: 是否编译模型
            half: 是否使用半精度
            max_new_tokens: 最大生成token数
            top_p: top-p采样参数
            repetition_penalty: 重复惩罚参数
            temperature: 温度参数
            chunk_length: 文本分块长度
            iterative_prompt: 是否使用迭代prompt
        """
        super().__init__()
        
        self.llama_checkpoint_path = Path(llama_checkpoint_path)
        self.decoder_checkpoint_path = Path(decoder_checkpoint_path)
        self.decoder_config_name = decoder_config_name
        self.device = device
        self.compile = compile
        self.half = half
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.chunk_length = chunk_length
        self.iterative_prompt = iterative_prompt
        
        # 验证检查点路径
        if not self.llama_checkpoint_path.exists():
            raise ValueError(f"Text2Semantic checkpoint not found: {self.llama_checkpoint_path}")
        if not self.decoder_checkpoint_path.exists():
            raise ValueError(f"VQGAN decoder checkpoint not found: {self.decoder_checkpoint_path}")

    def _process_speaker_embedding(self, prompt_audio: np.ndarray, prompt_text: str) -> torch.Tensor:
        """
        处理说话人音频和文本，生成说话人嵌入向量
        
        Args:
            prompt_audio: 说话人音频数据
            prompt_text: 说话人音频对应的文本
            
        Returns:
            说话人嵌入向量
        """
        # 保存临时音频文件
        temp_audio_path = Path("temp_prompt.wav")
        sf.write(temp_audio_path, prompt_audio, samplerate=16000)  # 假设采样率为16kHz
        
        try:
            # 使用VQGAN编码器获取说话人嵌入
            embedding, _ = vqgan_encoder(
                input_path=temp_audio_path,
                output_path=None,
                config_name=self.decoder_config_name,
                checkpoint_path=self.decoder_checkpoint_path,
                device=self.device
            )
        finally:
            # 清理临时文件
            if temp_audio_path.exists():
                os.remove(temp_audio_path)
        
        return embedding

    def tts(self, input_text: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        使用默认说话人合成语音
        
        Args:
            input_text: 输入文本
            output_path: 可选的输出音频文件路径
            
        Returns:
            合成的音频数据
        """
        if self.default_speaker_id is None:
            raise ValueError("No default speaker registered. Please register a speaker first.")
        
        return self.tts_with_speaker(input_text, self.default_speaker_id, output_path)

    def tts_with_speaker(self, input_text: str, speaker_id: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        使用指定说话人合成语音
        
        Args:
            input_text: 输入文本
            speaker_id: 说话人ID
            output_path: 可选的输出音频文件路径
            
        Returns:
            合成的音频数据
        """
        if speaker_id not in self.speaker_embeddings:
            raise ValueError(f"Speaker ID {speaker_id} not found. Please register the speaker first.")
        
        # 获取说话人嵌入
        prompt_tokens = self.speaker_embeddings[speaker_id]
        
        # 生成语义tokens
        semantic_tokens = text2semantic(
            text=input_text,
            prompt_text=input_text[:100],  # 使用输入文本的前100个字符作为prompt
            prompt_tokens=prompt_tokens,
            checkpoint_path=self.llama_checkpoint_path,
            device=self.device,
            compile=self.compile,
            half=self.half,
            max_new_tokens=self.max_new_tokens,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
            iterative_prompt=self.iterative_prompt,
            chunk_length=self.chunk_length,
            num_samples=1
        )[0]  # 只取第一个样本
        
        # 解码为音频
        audio_data, sample_rate = vqgan_decoder(
            indices=semantic_tokens,
            output_path=output_path,
            config_name=self.decoder_config_name,
            checkpoint_path=self.decoder_checkpoint_path,
            device=self.device
        )
        
        return audio_data
