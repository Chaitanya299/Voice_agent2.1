import json
import logging
import asyncio
from typing import Dict, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from .audio_converter import AudioConverter

logger = logging.getLogger(__name__)

class StreamingSTT:
    """
    Handles real-time speech-to-text using WebSockets and Whisper.
    """
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize the StreamingSTT with a Whisper model.
        
        Args:
            model_size: Size of the Whisper model (tiny, base, small, medium, large)
        """
        self.model_size = model_size
        self.model = None
        self.audio_buffer = bytearray()
        self.active_connections: Dict[str, WebSocket] = {}
        
    async def initialize(self):
        """Initialize the Whisper model asynchronously."""
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = WhisperModel(
                self.model_size,
                device="cpu",  # Use "cuda" if available
                compute_type="int8"  # Use "float16" for better quality if you have GPU
            )
            logger.info("Whisper model loaded successfully")
    
    async def process_audio(self, audio_data: bytes) -> str:
        """
        Process audio data and return the transcribed text.
        
        Args:
            audio_data: PCM audio data at 16kHz
            
        Returns:
            str: Transcribed text
        """
        try:
            if not self.model:
                await self.initialize()
                
            # Convert bytes to numpy array for Whisper
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Transcribe the audio
            segments, _ = self.model.transcribe(
                audio_array,
                language="en",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                without_timestamps=True
            )
            
            # Get the transcribed text
            text = ""
            for segment in segments:
                text += segment.text
                
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return ""
    
    async def handle_websocket(self, websocket: WebSocket, call_sid: str):
        """
        Handle a WebSocket connection for real-time audio streaming.
        
        Args:
            websocket: The WebSocket connection
            call_sid: The Twilio Call SID for this connection
        """
        await websocket.accept()
        self.active_connections[call_sid] = websocket
        logger.info(f"New WebSocket connection: {call_sid}")
        
        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                event = message.get("event")
                
                if event == "media":
                    # Process the audio data
                    media_payload = message.get("media", {})
                    chunk = media_payload.get("payload")
                    
                    if chunk:
                        # Process the audio chunk
                        pcm_audio = AudioConverter.process_twilio_audio(chunk)
                        
                        # Add to buffer and process if we have enough data
                        self.audio_buffer.extend(pcm_audio)
                        
                        # Process in chunks of 1 second of audio (32000 bytes for 16kHz, 16-bit mono)
                        while len(self.audio_buffer) >= 32000:
                            chunk_to_process = self.audio_buffer[:32000]
                            self.audio_buffer = self.audio_buffer[32000:]
                            
                            # Transcribe the audio chunk
                            text = await self.process_audio(chunk_to_process)
                            
                            if text:
                                logger.info(f"Partial transcript: {text}")
                                # Send the partial transcript back to the client
                                await websocket.send_json({
                                    "event": "transcript",
                                    "text": text,
                                    "is_final": False
                                })
                
                elif event == "stop":
                    logger.info(f"Stop event received for call: {call_sid}")
                    break
                    
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {call_sid}")
        except Exception as e:
            logger.error(f"Error in WebSocket handler: {e}")
        finally:
            # Clean up
            if call_sid in self.active_connections:
                del self.active_connections[call_sid]
                
            # Process any remaining audio in the buffer
            if self.audio_buffer:
                try:
                    text = await self.process_audio(self.audio_buffer)
                    if text:
                        logger.info(f"Final transcript: {text}")
                        await websocket.send_json({
                            "event": "transcript",
                            "text": text,
                            "is_final": True
                        })
                except Exception as e:
                    logger.error(f"Error processing final audio chunk: {e}")
            
            logger.info(f"WebSocket connection closed: {call_sid}")
            
    async def close(self):
        """Clean up resources."""
        # Close all active WebSocket connections
        for websocket in self.active_connections.values():
            try:
                await websocket.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
        
        self.active_connections.clear()
        self.audio_buffer = bytearray()
        self.model = None
