from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Union, Dict
import soundfile as sf

class BaseTTS(ABC):
    def __init__(self):
        self.speaker_embeddings: Dict[str, any] = {}
        self.default_speaker_id: Optional[str] = None
    
    def register_speaker(self, prompt_audio: Union[str, np.ndarray], prompt_text: str, speaker_id: Optional[str] = None) -> str:
        """
        Register a new speaker with prompt audio and text
        
        Args:
            prompt_audio: Path to audio file or numpy array of audio data
            prompt_text: Text corresponding to the prompt audio
            speaker_id: Optional speaker ID. If None, will generate one
            
        Returns:
            speaker_id: The ID of the registered speaker
        """
        if speaker_id is None:
            speaker_id = f"speaker_{len(self.speaker_embeddings)}"
            
        # Load audio if path is provided
        if isinstance(prompt_audio, str):
            prompt_audio, _ = sf.read(prompt_audio)
            
        # Process and store speaker embedding
        embedding = self._process_speaker_embedding(prompt_audio, prompt_text)
        self.speaker_embeddings[speaker_id] = embedding
        
        # Set as default speaker if first one
        if self.default_speaker_id is None:
            self.default_speaker_id = speaker_id
            
        return speaker_id
    
    @abstractmethod
    def _process_speaker_embedding(self, prompt_audio: np.ndarray, prompt_text: str) -> any:
        """
        Process prompt audio and text to generate speaker embedding
        
        Args:
            prompt_audio: Audio data as numpy array
            prompt_text: Text corresponding to the prompt audio
            
        Returns:
            Speaker embedding in format required by specific TTS model
        """
        pass
    
    @abstractmethod
    def tts(self, input_text: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        Synthesize speech from input text using default speaker
        
        Args:
            input_text: Text to synthesize
            output_path: Optional path to save audio file
            
        Returns:
            Synthesized audio as numpy array
        """
        pass
    
    @abstractmethod
    def tts_with_speaker(self, input_text: str, speaker_id: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        Synthesize speech from input text using specified speaker
        
        Args:
            input_text: Text to synthesize
            speaker_id: ID of speaker to use
            output_path: Optional path to save audio file
            
        Returns:
            Synthesized audio as numpy array
        """
        pass
