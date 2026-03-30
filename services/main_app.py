# services/main_app.py
import os
import time
import json
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import whisper

# Import your existing service classes
from emotion_service import EmotionClassifier
from rag_service import RAG_LLM_System

# --- Global dictionary to hold models and services ---
ml_services = {}

# --- NEW: Configuration for a dedicated uploads folder ---
AUDIO_UPLOADS_DIR = "../pipeline_io/audio_uploads"

# --- FFmpeg Check ---
def is_ffmpeg_installed():
    """Check if ffmpeg is installed and accessible in the system's PATH."""
    return shutil.which("ffmpeg") is not None

# --- Application Lifespan (for loading models) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all ML models and services on startup.
    """
    print("--- 🚀 Initializing All Services ---")
    
    # Check for FFmpeg before doing anything else.
    print("🔍 Checking for FFmpeg installation...")
    if not is_ffmpeg_installed():
        print("❌ CRITICAL ERROR: FFmpeg is not installed or not found in your system's PATH.")
        print("   Whisper requires FFmpeg to process audio.")
        print("   Please install FFmpeg from https://ffmpeg.org/download.html and ensure it's added to your system's PATH.")
        raise RuntimeError("FFmpeg not found. Please see console for installation instructions.")
    print("✅ FFmpeg found.")

    # Create the dedicated directory for audio uploads
    os.makedirs(AUDIO_UPLOADS_DIR, exist_ok=True)
    print(f"✅ Ensured audio upload directory exists at: {os.path.abspath(AUDIO_UPLOADS_DIR)}")

    ml_services["whisper"] = whisper.load_model("base")
    ml_services["emotion_classifier"] = EmotionClassifier()
    ml_services["rag_llm_system"] = RAG_LLM_System()
    print("--- ✅ All Services Ready ---")
    yield
    # This part runs on application shutdown
    print("--- 🧹 Clearing Services ---")
    ml_services.clear()

# --- FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---

@app.post("/start_session")
async def start_session(file: UploadFile = File(...)):
    """
    1. Receives audio file and saves it to a dedicated project folder.
    2. Transcribes the audio.
    3. Cleans up the audio file.
    4. Continues with the pipeline.
    """
    # --- FIX APPLIED HERE ---
    # We now save to a predictable folder within our project.
    file_path = None
    try:
        # Create a unique filename to prevent conflicts
        timestamp = int(time.time() * 1000)
        file_path = os.path.join(AUDIO_UPLOADS_DIR, f"upload_{timestamp}.webm")

        # Save the uploaded file to our dedicated directory
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"🎤 Audio saved to disk: {file_path}")
        print("🤫 Handing off to Whisper for transcription...")
        transcription_result = ml_services["whisper"].transcribe(file_path, fp16=False)
        original_text = transcription_result["text"]
        print(f"📄 Transcription: '{original_text}'")

    except Exception as e:
        print(f"❌ An error occurred during transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Error during transcription: {e}")
    finally:
        # This block always runs, ensuring we clean up the audio file
        if file_path and os.path.exists(file_path):
            print(f"🗑️ Deleting audio file: {file_path}")
            os.remove(file_path)
    # --- END OF FIX ---

    if not original_text.strip():
        raise HTTPException(status_code=400, detail="Transcription resulted in empty text. Please try speaking again.")

    # 2. Classify Emotion
    print("🤔 Classifying emotion...")
    emotion_classifier = ml_services["emotion_classifier"]
    emotion_data = emotion_classifier.predict(original_text)
    print(f"❤️ Detected Emotion: {emotion_data.get('emotion')}")

    # 3. Generate Clarifying Questions
    print("❓ Generating clarifying questions...")
    rag_system = ml_services["rag_llm_system"]
    questions = rag_system.generate_clarifying_questions(original_text, emotion_data)

    return {
        "original_text": original_text,
        "emotion_data": emotion_data,
        "questions": questions
    }

@app.post("/get_summary")
async def get_summary(payload: dict = Body(...)):
    """
    1. Receives the conversation history (original text + questions + answers).
    2. Generates the final summary response.
    3. Returns the summary to the frontend.
    """
    print("📝 Generating final summary...")
    
    original_text = payload.get("original_text")
    answers = payload.get("answers", []) # List of dicts: [{"question": q, "answer": a}]

    if not original_text or not answers:
        raise HTTPException(status_code=400, detail="Missing original text or answers for summary.")

    # Build the conversation history string for the LLM
    history_lines = [f"User's initial statement: \"{original_text}\""]
    for item in answers:
        history_lines.append(f"Assistant's Question: \"{item['question']}\"")
        history_lines.append(f"User's Answer (1-10): {item['answer']}")
    
    conversation_history = "\n".join(history_lines)
    
    rag_system = ml_services["rag_llm_system"]
    final_response = rag_system.generate_summary_response(conversation_history)
    
    print("✅ Summary generated.")
    return {"summary": final_response}
