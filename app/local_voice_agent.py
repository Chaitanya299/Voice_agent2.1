import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import tempfile
import os
import sys
import re
import time

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
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)   # bold
    text = re.sub(r"[-•]\s*", "", text)            # bullets
    text = re.sub(r"\n+", " ", text)               # newlines
    text = text.replace("–", "-")                  # en-dash
    return text.strip()


def clean_for_voice(text: str) -> str:
    cleaned = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("---", "*", "#", "Answer:", "Context:")):
            continue
        cleaned.append(line)
    return " ".join(cleaned)


def record_push_to_talk():
    input("\n🎤 Press ENTER to start recording...")
    print("🎙️ Recording... Press ENTER to stop.")

    frames = []

    def callback(indata, frames_count, time_info, status):
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

    audio = np.concatenate(frames, axis=0)
    duration = len(audio) / SAMPLE_RATE
    print(f"🧪 Recorded duration: {duration:.2f}s")

    if duration < MIN_SECONDS:
        raise ValueError("Recording too short. Please speak clearly.")

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    audio = (audio * 32767).astype(np.int16)

    # RMS debug
    rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
    print(f"🔈 Audio RMS: {rms:.2f}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(tmp.name, SAMPLE_RATE, audio)

    return tmp.name


def play_audio_macos(path: str):
    os.system(f"afplay '{path}'")


def run_agent_loop():
    print("\n🟢 Local Voice Agent started (Ctrl+C to exit)\n")

    while True:
        try:
            # =========================
            # 1. Record (NOT timed)
            # =========================
            audio_path = record_push_to_talk()

            # =========================
            # START COMPUTE TIMING
            # =========================
            compute_start = time.perf_counter()

            # =========================
            # 2. STT
            # =========================
            stt_start = time.perf_counter()
            text = transcribe_audio(audio_path)
            stt_time = time.perf_counter() - stt_start

            print("\n📝 STT OUTPUT repr():", repr(text))
            print(f"📝 You said: {text}")

            if not text.strip():
                print("⚠️ Empty transcription, try again.")
                os.remove(audio_path)
                continue

            # =========================
            # 3. LLM + RAG
            # =========================
            llm_start = time.perf_counter()
            reply = get_rag_response(text)
            llm_time = time.perf_counter() - llm_start

            print("\n🤖 LLM RAW OUTPUT repr():", repr(reply))

            # =========================
            # 4. Clean for TTS
            # =========================
            clean_reply = clean_for_tts(clean_for_voice(reply))
            print("\n🔊 TTS INPUT repr():", repr(clean_reply))

            if not clean_reply:
                print("⚠️ Nothing to speak.")
                os.remove(audio_path)
                continue

            # =========================
            # 5. TTS
            # =========================
            tts_start = time.perf_counter()
            output_audio = synthesize_speech(clean_reply, "local_reply.wav")
            tts_time = time.perf_counter() - tts_start

            # =========================
            # END COMPUTE TIMING
            # =========================
            compute_time = time.perf_counter() - compute_start

            # =========================
            # 6. Playback (NOT timed)
            # =========================
            play_audio_macos(output_audio)

            # =========================
            # Timing report
            # =========================
            print("\n⏱ COMPUTE TIMING BREAKDOWN")
            print(f"⏱ STT time:   {stt_time:.2f}s")
            print(f"⏱ LLM time:   {llm_time:.2f}s")
            print(f"⏱ TTS time:   {tts_time:.2f}s")
            print(f"⏱ TOTAL (compute only): {compute_time:.2f}s\n")

            os.remove(audio_path)

        except KeyboardInterrupt:
            print("\n👋 Exiting voice agent.")
            break

        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    run_agent_loop()