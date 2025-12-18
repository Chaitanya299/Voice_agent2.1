import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import tempfile
import os
import sys
import re
from app.stt import transcribe_audio
from app.agent import get_rag_response
from app.tts import synthesize_speech

# =========================
# Audio configuration
# =========================
SAMPLE_RATE = 16000        # 16 kHz (Whisper best practice)
CHANNELS = 1              # mono
DTYPE = "float32"         # IMPORTANT: float32 for macOS
BLOCKSIZE = 1024
MIN_SECONDS = 1.0         # prevent Whisper guessing

def clean_for_tts(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # bold
    text = re.sub(r"[-•]\s*", "", text)           # bullets
    text = re.sub(r"\n+", " ", text)
    return text.strip()


def record_push_to_talk():
    """
    Push-to-talk recording:
    - ENTER to start
    - ENTER to stop
    - Ensures minimum duration
    """
    input("\n🎤 Press ENTER to start recording...")
    print("🎙️ Recording... Press ENTER to stop.")

    frames = []

    def callback(indata, frames_count, time, status):
        if status:
            print(f"⚠️ Audio status: {status}", file=sys.stderr)
        frames.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=BLOCKSIZE,
        callback=callback,
    ):
        input()

    if not frames:
        raise RuntimeError("No audio captured")

    # Concatenate audio
    audio = np.concatenate(frames, axis=0)
    duration = len(audio) / SAMPLE_RATE
    print(f"🧪 Recorded duration: {duration:.2f}s")

    if duration < MIN_SECONDS:
        raise ValueError("Recording too short. Please speak clearly.")

    # =========================
    # Normalize + convert
    # =========================
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    audio = (audio * 32767).astype(np.int16)

    # 🔈 DEBUG: RMS loudness
    rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
    print(f"🔈 Audio RMS: {rms:.2f}")

    # Save temp WAV
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(tmp.name, SAMPLE_RATE, audio)

    return tmp.name


def clean_for_voice(text: str) -> str:
    """
    Remove markdown, separators, and prompt artifacts
    so TTS gets clean spoken language.
    """
    cleaned = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("---", "*", "#", "Answer:", "Context:")):
            continue
        cleaned.append(line)
    return " ".join(cleaned)


def play_audio_macos(path: str):
    os.system(f"afplay '{path}'")


def run_agent_loop():
    print("\n🟢 Local Voice Agent started (Ctrl+C to exit)\n")

    while True:
        try:
            # =========================
            # 1. Record
            # =========================
            audio_path = record_push_to_talk()

            # =========================
            # 2. STT
            # =========================
            text = transcribe_audio(audio_path)
            print("\n📝 STT OUTPUT repr():", repr(text))
            print(f"📝 You said: {text}")

            if not text.strip():
                print("⚠️ Empty transcription, try again.")
                os.remove(audio_path)
                continue

            # =========================
            # 3. LLM + RAG
            # =========================
            reply = get_rag_response(text)
            print("\n🤖 LLM RAW OUTPUT repr():", repr(reply))

            # =========================
            # 4. Clean for voice
            # =========================
            tts_input = clean_for_voice(reply)
            print("\n🔊 TTS INPUT repr():", repr(tts_input))

            if not tts_input:
                print("⚠️ Nothing to speak.")
                os.remove(audio_path)
                continue

            # =========================
            # 5. TTS
            # =========================
            output_audio = synthesize_speech(tts_input, "local_reply.wav")

            # =========================
            # 6. Playback
            # =========================
            play_audio_macos(output_audio)

            # Cleanup
            os.remove(audio_path)

        except KeyboardInterrupt:
            print("\n👋 Exiting voice agent.")
            break

        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    run_agent_loop()