import os
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "data", "knowledge_base.md")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "embeddings", "chroma_db")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PHI2_MODEL_PATH = os.path.join(MODEL_DIR, "phi-2.Q4_K_M.gguf")
LLAMA3B_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
)

ACTIVE_MODEL_PATH = LLAMA3B_MODEL_PATH

AUDIO_UPLOAD_DIR = os.path.join(BASE_DIR, "audio_uploads")
AUDIO_OUTPUT_DIR = os.path.join(BASE_DIR, "audio_output")

# Twilio Credentials
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

# Ensure directories exist
for d in [AUDIO_UPLOAD_DIR, AUDIO_OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)
