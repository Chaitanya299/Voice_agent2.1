from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, Response
import os
import shutil
import uuid
import httpx
from twilio.twiml.voice_response import VoiceResponse, Play
from loguru import logger

from app.stt import transcribe_audio
from app.agent import get_rag_response # Changed from app.llm import generate_reply
from app.tts import synthesize_speech
from app.config import AUDIO_UPLOAD_DIR, AUDIO_OUTPUT_DIR, BASE_DIR, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN

# Configure Loguru logger
LOG_FILE_PATH = os.path.join(BASE_DIR, "logs", "agent.log")
logger.add(LOG_FILE_PATH, rotation="500 MB", compression="zip", level="INFO")

app = FastAPI()

# Global dict to track if first reply was given for each call
first_reply_given = {}

@app.post("/process_audio/")
async def process_audio(audio_file: UploadFile = File(...)):
    logger.info(f"Received audio processing request for file: {audio_file.filename}")
    if not audio_file.filename.endswith(('.wav', '.mp3', '.ogg')):
        logger.error(f"Invalid file format received: {audio_file.filename}")
        raise HTTPException(status_code=400, detail="Invalid file format. Only WAV, MP3, OGG are supported.")

    # Save the uploaded audio file temporarily
    unique_filename = f"{uuid.uuid4()}_{audio_file.filename}"
    audio_path = os.path.join(AUDIO_UPLOAD_DIR, unique_filename)
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)
    logger.info(f"Audio file saved temporarily to: {audio_path}")

    # 1. Transcribe audio
    transcribed_text = transcribe_audio(audio_path)
    if "Error" in transcribed_text:
        logger.error(f"STT Error for {audio_file.filename}: {transcribed_text}")
        raise HTTPException(status_code=500, detail=f"STT Error: {transcribed_text}")
    logger.info(f"Transcribed text: {transcribed_text}")

    # 2. Generate LLM reply using RAG
    llm_reply = get_rag_response(transcribed_text) # Changed from generate_reply
    if "Error" in llm_reply:
        logger.error(f"RAG Error for \"{transcribed_text}\": {llm_reply}")
        raise HTTPException(status_code=500, detail=f"RAG Error: {llm_reply}")
    logger.info(f"LLM Reply (from RAG): {llm_reply}")

    # 3. Synthesize speech from LLM reply
    output_audio_filename = f"reply_{uuid.uuid4()}.wav"
    synthesized_audio_path = synthesize_speech(llm_reply, output_audio_filename)
    if "Error" in synthesized_audio_path:
        logger.error(f"TTS Error for \"{llm_reply}\": {synthesized_audio_path}")
        raise HTTPException(status_code=500, detail=f"TTS Error: {synthesized_audio_path}")
    logger.info(f"Synthesized audio saved to: {synthesized_audio_path}")

    # Clean up the uploaded audio file
    os.remove(audio_path)
    logger.info(f"Cleaned up temporary audio file: {audio_path}")

    return {"transcribed_text": transcribed_text, "llm_reply": llm_reply, "reply_audio_path": synthesized_audio_path}

@app.post("/twilio_voice")
async def twilio_voice(request: Request):
    logger.info("Received Twilio voice webhook request.")
    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    recording_url = form_data.get("RecordingUrl") # URL of the recorded speech from Twilio

    response = VoiceResponse()

    if recording_url:
        logger.info(f"Twilio recording URL received: {recording_url} for CallSid: {call_sid}")
        try:
            # Create auth tuple
            auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            
            # First, get the recording details to verify it exists
            recording_sid = recording_url.split('/')[-1]
            details_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Recordings/{recording_sid}.json"
            
            async with httpx.AsyncClient(auth=auth) as client:
                # Verify recording exists
                details_response = await client.get(details_url)
                details_response.raise_for_status()
                
                # Download the recording with .wav extension
                media_url = f"{recording_url}.wav"
                audio_response = await client.get(media_url)
                audio_response.raise_for_status()
                
                # Save the downloaded audio
                unique_filename = f"{call_sid}_recorded.wav"
                recorded_audio_path = os.path.join(AUDIO_UPLOAD_DIR, unique_filename)
                with open(recorded_audio_path, "wb") as f:
                    f.write(audio_response.content)
                
                logger.info(f"Recorded audio saved temporarily to: {recorded_audio_path}")

                # 1. Transcribe audio
            transcribed_text = transcribe_audio(recorded_audio_path)
            os.remove(recorded_audio_path) # Clean up recorded audio
            logger.info(f"Transcribed text from Twilio call {call_sid}: {transcribed_text}")

            if "Error" in transcribed_text:
                logger.error(f"STT Error for Twilio call {call_sid}: {transcribed_text}")
                response.say("I apologize, but I encountered an error transcribing your speech.")
                return Response(content=str(response), media_type="application/xml")

            # 2. Generate LLM reply using RAG
            llm_reply = get_rag_response(transcribed_text) # Changed from generate_reply
            if "Error" in llm_reply:
                logger.error(f"RAG Error for Twilio call {call_sid} (prompt: \"{transcribed_text}\"): {llm_reply}")
                response.say("I apologize, but I encountered an error generating a reply.")
                return Response(content=str(response), media_type="application/xml")
            logger.info(f"LLM Reply (from RAG) for Twilio call {call_sid}: {llm_reply}")

            # 3. Synthesize speech from LLM reply
            if not first_reply_given.get(call_sid):
                max_tts_length = 200  
                first_reply_given[call_sid] = True
            else:
                max_tts_length = 100  
            short_reply = llm_reply[:max_tts_length]
            output_audio_filename = f"reply_{call_sid}.wav"
            synthesized_audio_path = synthesize_speech(short_reply, output_audio_filename)

            if "Error" in synthesized_audio_path:
                logger.error(f"TTS Error for Twilio call {call_sid} (reply: \"{llm_reply}\"): {synthesized_audio_path}")
                response.say("I apologize, but I encountered an error synthesizing my response.")
                return Response(content=str(response), media_type="application/xml")
            logger.info(f"Synthesized audio for Twilio call {call_sid} saved to: {synthesized_audio_path}")

            # Construct the URL for the synthesized audio using the provided public URL
            audio_url = f"https://preexistent-multiaxial-kelsie.ngrok-free.dev/audio/{output_audio_filename}"
            logger.info(f"Twilio audio URL for playback: {audio_url}")

            response.say("Here is my response:")
            response.play(audio_url)
            response.say("Is there anything else I can assist you with?")

        except httpx.RequestError as e:
            logger.error(f"HTTPX Request Error for Twilio call {call_sid}: {e}")
            response.say(f"I am sorry, I could not retrieve the audio recording. Error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred for Twilio call {call_sid}: {e}")
            response.say(f"An unexpected error occurred: {e}")
    else:
        logger.info(f"No recording URL received for Twilio call {call_sid}. Initiating recording.")
        response.say("I did not receive any audio. Please try speaking after the tone.")
        response.record(action="/twilio_voice", maxLength="10", timeout="5", transcribe=True) # Record user's speech

    return Response(content=str(response), media_type="application/xml")

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """
    Serves synthesized audio files.
    """
    file_path = os.path.join(AUDIO_OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return FileResponse(file_path, media_type="audio/wav")

if __name__ == "__main__":
    import uvicorn
    print(f"FastAPI app created. To run, use: uvicorn app.main:app --reload --port 8000")
    # For direct execution (e.g., during development/testing):
    # uvicorn.run(app, host="0.0.0.0", port=8000)