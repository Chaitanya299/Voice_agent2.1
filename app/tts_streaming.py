import re
import numpy as np
from typing import List, Optional

from TTS.api import TTS

class StreamingTTS:
    """
    XTTS-v2 based sentence-level TTS.
    - In-memory audio only
    - CPU-first
    - Load model once, reuse
    - Future-safe for Twilio / WebRTC
    """

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        language: str = "en",
        device: str = "cpu",
        debug_save_wav: bool = False,
    ):
        self.language = language
        self.debug_save_wav = debug_save_wav

        # Load once
        self.tts = TTS(
            model_name=model_name,
            progress_bar=False,
            gpu=(device != "cpu"),
        )

    # -------------------------
    # Sentence handling
    # -------------------------

    def split_sentences(self, text: str) -> List[str]:
        """
        Agent already splits, this is a safety net.
        """
        text = text.strip()
        if not text:
            return []

        # Simple punctuation-based split
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    # -------------------------
    # Synthesis
    # -------------------------

    def synthesize_sentence(
        self,
        sentence: str,
        sample_rate: int = 24000,
    ) -> Optional[np.ndarray]:
        """
        Generate audio for a single sentence.
        Returns float32 PCM in range [-1, 1]
        """

        if not sentence:
            return None

        wav = self.tts.tts(
            text=sentence,
            language=self.language,
            speaker_wav=None,  # no voice cloning for now
        )

        audio = np.array(wav, dtype=np.float32)

        if self.debug_save_wav:
            from scipy.io.wavfile import write
            write("debug_xtts.wav", sample_rate, audio)

        return audio

    def synthesize(
        self,
        text: str,
        sample_rate: int = 24000,
    ) -> List[np.ndarray]:
        """
        Sentence-by-sentence synthesis.
        Returns list of float32 audio chunks.
        """

        sentences = self.split_sentences(text)
        audio_chunks: List[np.ndarray] = []

        for sentence in sentences:
            audio = self.synthesize_sentence(sentence, sample_rate)
            if audio is not None:
                audio_chunks.append(audio)

        return audio_chunks
