import base64
import numpy as np
from typing import Tuple, Optional
import audioop
import logging

logger = logging.getLogger(__name__)

class AudioConverter:
    """
    Handles audio format conversion between Twilio's μ-law format and Whisper's expected format.
    """
    
    @staticmethod
    def base64_to_pcm(audio_base64: str) -> bytes:
        """
        Convert base64 encoded audio to PCM bytes.
        
        Args:
            audio_base64: Base64 encoded audio data
            
        Returns:
            bytes: Decoded PCM audio data
        """
        try:
            return base64.b64decode(audio_base64)
        except Exception as e:
            logger.error(f"Error decoding base64 audio: {e}")
            raise
    
    @staticmethod
    def ulaw_to_linear(audio_ulaw: bytes) -> bytes:
        """
        Convert μ-law encoded audio to 16-bit linear PCM.
        
        Args:
            audio_ulaw: μ-law encoded audio data
            
        Returns:
            bytes: 16-bit linear PCM audio data
        """
        try:
            return audioop.ulaw2lin(audio_ulaw, 2)  # 2 = 16-bit samples
        except Exception as e:
            logger.error(f"Error converting μ-law to linear PCM: {e}")
            raise
    
    @staticmethod
    def resample_audio(audio_pcm: bytes, in_rate: int = 8000, out_rate: int = 16000) -> bytes:
        """
        Resample audio from in_rate to out_rate.
        
        Args:
            audio_pcm: PCM audio data
            in_rate: Input sample rate in Hz
            out_rate: Output sample rate in Hz
            
        Returns:
            bytes: Resampled audio data
        """
        try:
            if in_rate == out_rate:
                return audio_pcm
                
            # Convert bytes to numpy array for resampling
            audio_array = np.frombuffer(audio_pcm, dtype=np.int16)
            
            # Calculate new length after resampling
            new_length = int(len(audio_array) * out_rate / in_rate)
            
            # Simple linear resampling (for production, consider more sophisticated resampling)
            resampled = np.interp(
                np.linspace(0, len(audio_array), new_length, endpoint=False),
                np.arange(len(audio_array)),
                audio_array
            ).astype(np.int16)
            
            return resampled.tobytes()
            
        except Exception as e:
            logger.error(f"Error resampling audio: {e}")
            raise
    
    @classmethod
    def process_twilio_audio(cls, audio_base64: str) -> bytes:
        """
        Process Twilio audio from base64 μ-law to 16kHz 16-bit PCM.
        
        Args:
            audio_base64: Base64 encoded μ-law audio from Twilio
            
        Returns:
            bytes: Processed 16kHz 16-bit PCM audio
        """
        try:
            # 1. Decode base64
            ulaw_audio = cls.base64_to_pcm(audio_base64)
            
            # 2. Convert μ-law to linear PCM (16-bit)
            pcm_audio = cls.ulaw_to_linear(ulaw_audio)
            
            # 3. Resample from 8kHz to 16kHz
            resampled_audio = cls.resample_audio(pcm_audio, in_rate=8000, out_rate=16000)
            
            return resampled_audio
            
        except Exception as e:
            logger.error(f"Error processing Twilio audio: {e}")
            raise
