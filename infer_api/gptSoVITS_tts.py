from typing import Optional, Union, Dict
import numpy as np
import soundfile as sf
import os
from pathlib import Path

from .baseTTS import BaseTTS
from GPT_SoVITS.inference_webui import (
    change_gpt_weights, 
    change_sovits_weights, 
    get_tts_wav
)
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()

class GPTSoVITSTTS(BaseTTS):
    def __init__(
        self,
        gpt_model_path: Union[str, Path],
        sovits_model_path: Union[str, Path],
        default_language: str = "中文",
        top_p: float = 0.6,
        temperature: float = 0.7,
    ):
        """
        Initialize GPT-SoVITS TTS system
        
        Args:
            gpt_model_path: Path to GPT model checkpoint (.ckpt file)
            sovits_model_path: Path to SoVITS model checkpoint (.pth file)
            default_language: Default language for synthesis, options: ["中文", "英文", "日文", "中英混合", "日英混合", "多语种混合"]
            top_p: Top-p sampling parameter (default: 0.6)
            temperature: Temperature parameter for sampling (default: 0.7)
        """
        super().__init__()
        self.gpt_model_path = str(gpt_model_path)
        self.sovits_model_path = str(sovits_model_path)
        self.default_language = default_language
        self.top_p = top_p
        self.temperature = temperature
        
        # Initialize models
        change_gpt_weights(gpt_path=self.gpt_model_path)
        try:
            next(change_sovits_weights(sovits_path=self.sovits_model_path))
        except StopIteration:
            pass
        
        # Store speaker embeddings as (audio_path, text) pairs
        self.speaker_embeddings: Dict[str, tuple] = {}
    
    def _process_speaker_embedding(self, prompt_audio: np.ndarray, prompt_text: str) -> tuple:
        """
        Process prompt audio and text for speaker embedding
        
        For GPT-SoVITS, we store the reference audio path and text directly,
        as the model processes them during inference
        
        Args:
            prompt_audio: Audio data as numpy array
            prompt_text: Text corresponding to the prompt audio
            
        Returns:
            Tuple of (temp_audio_path, prompt_text)
        """
        # Save audio to temporary file
        temp_dir = Path("temp_speakers")
        temp_dir.mkdir(exist_ok=True)
        
        temp_audio_path = temp_dir / f"ref_audio_{len(self.speaker_embeddings)}.wav"
        sf.write(str(temp_audio_path), prompt_audio, samplerate=16000)
        
        return (str(temp_audio_path), prompt_text)
    
    def tts(self, input_text: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        Synthesize speech using default speaker
        
        Args:
            input_text: Text to synthesize
            output_path: Optional path to save audio file
            
        Returns:
            Synthesized audio as numpy array
        """
        if self.default_speaker_id is None:
            raise ValueError("No default speaker registered. Please register a speaker first.")
        
        return self.tts_with_speaker(input_text, self.default_speaker_id, output_path)
    
    def tts_with_speaker(self, input_text: str, speaker_id: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        Synthesize speech using specified speaker
        
        Args:
            input_text: Text to synthesize
            speaker_id: ID of speaker to use
            output_path: Optional path to save audio file
            
        Returns:
            Synthesized audio as numpy array
        """
        if speaker_id not in self.speaker_embeddings:
            raise ValueError(f"Speaker ID {speaker_id} not found. Please register the speaker first.")
        
        # Get reference audio and text
        ref_audio_path, ref_text = self.speaker_embeddings[speaker_id]
        
        # Run inference
        synthesis_result = get_tts_wav(
            ref_wav_path=ref_audio_path,
            prompt_text=ref_text,
            prompt_language=i18n(self.default_language),
            text=input_text,
            text_language=i18n(self.default_language),
            top_p=self.top_p,
            temperature=self.temperature,
        )
        
        # Get audio data
        result_list = list(synthesis_result)
        if not result_list:
            raise RuntimeError("Failed to synthesize audio")
            
        sample_rate, audio_data = result_list[-1]
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, audio_data, sample_rate)
            
        return audio_data
    
    def __del__(self):
        """Cleanup temporary files on deletion"""
        temp_dir = Path("temp_speakers")
        if temp_dir.exists():
            for file in temp_dir.glob("*.wav"):
                file.unlink()
            temp_dir.rmdir() 