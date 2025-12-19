

import numpy as np
import logging
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class StreamingSTT:
    """
    Streaming-style STT engine that:
    - Initializes Whisper ONCE
    - Accepts incremental audio chunks
    - Produces FINAL text only on demand

    Twilio / WebSocket support is temporarily disabled.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language

        self.model: WhisperModel | None = None
        self.audio_buffer: list[np.ndarray] = []

    # =========================
    # Lifecycle
    # =========================

    def initialize(self):
        """
        Load Whisper model once and reuse across turns.
        """
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("Whisper model loaded")

    def reset(self):
        """
        Clear audio buffer between utterances.
        """
        self.audio_buffer.clear()

    # =========================
    # Audio ingestion
    # =========================

    def feed_audio_chunk(self, audio_chunk: np.ndarray):
        """
        Feed a chunk of audio into the buffer.

        Expected format:
        - mono
        - float32
        - range [-1.0, 1.0]
        - sample rate 16 kHz
        """
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.squeeze()

        self.audio_buffer.append(audio_chunk)

    # =========================
    # Final transcription
    # =========================

    def finalize(self) -> str:
        """
        Transcribe all buffered audio and return FINAL text.
        """
        if not self.audio_buffer:
            return ""

        self.initialize()

        audio = np.concatenate(self.audio_buffer).astype(np.float32)

        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=5,
            vad_filter=True,
            without_timestamps=True,
        )

        text_parts = [seg.text.strip() for seg in segments if seg.text.strip()]
        final_text = " ".join(text_parts)

        self.reset()
        return final_text


# ============================================================
# ⚠️ Twilio / WebSocket support (TEMPORARILY DISABLED)
# ============================================================
#
# The following logic will be re-enabled later.
# Keeping it commented ensures ZERO breaking changes now.
#
# class StreamingSTTWebSocket:
#     ...
#
# ============================================================