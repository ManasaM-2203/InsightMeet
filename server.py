from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import stable_whisper
import torch
import uuid
import os
import shutil
import ffmpeg
from transformers import pipeline
import traceback
import logging
import uvicorn

# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="InsightMeet API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# FOLDERS
# =========================
UPLOAD_FOLDER = "uploads"
MEDIA_FOLDER = "media"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MEDIA_FOLDER, exist_ok=True)

# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# =========================
# LOAD MODELS ONCE
# =========================
try:
    logger.info("Loading Whisper model...")
    whisper_model = stable_whisper.load_model("base")
    logger.info("Whisper model loaded successfully.")

    logger.info("Loading summarizer...")
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=0 if device == "cuda" else -1,
    )
    logger.info("Summarizer loaded successfully.")

except Exception as e:
    logger.error(f"Model loading failed: {e}")
    raise e

# =========================
# HELPERS
# =========================
def extract_audio(input_path: str, output_path: str):
    """
    Extract mono 16kHz WAV audio using ffmpeg.
    """
    try:
        (
            ffmpeg
            .input(input_path)
            .output(
                output_path,
                format="wav",
                ar=16000,
                ac=1
            )
            .run(overwrite_output=True, quiet=True)
        )
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        raise Exception("Failed to extract audio")


def safe_summarize(text: str, max_len: int = 150, min_len: int = 40):
    """
    Safely summarize text.
    If text is too short, return original text.
    """
    try:
        words = text.split()

        if len(words) < 30:
            return text

        result = summarizer(
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )

        return result[0]["summary_text"]

    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return text


# =========================
# ROOT ROUTE
# =========================
@app.get("/")
def root():
    return {
        "message": "InsightMeet API — powered by stable-ts + FastAPI"
    }


# =========================
# PROCESS FILE ROUTE
# =========================
@app.post("/process-file/")
async def process_file(file: UploadFile = File(...)):

    file_id = str(uuid.uuid4())

    ext = os.path.splitext(file.filename)[1].lower()

    raw_path = os.path.join(
        UPLOAD_FOLDER,
        f"{file_id}{ext}"
    )

    audio_path = os.path.join(
        UPLOAD_FOLDER,
        f"{file_id}.wav"
    )

    try:
        # =========================
        # SAVE UPLOADED FILE
        # =========================
        logger.info("Saving uploaded file...")

        with open(raw_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # =========================
        # EXTRACT AUDIO
        # =========================
        logger.info("Extracting audio...")
        extract_audio(raw_path, audio_path)

        # =========================
        # SAVE MEDIA FOR STREAMING
        # =========================
        media_output = os.path.join(
            MEDIA_FOLDER,
            f"{file_id}{ext}"
        )

        shutil.copy(raw_path, media_output)

        # =========================
        # TRANSCRIPTION
        # =========================
        logger.info("Starting transcription...")

        result = whisper_model.transcribe(
            audio_path,
            regroup=True
        )

        # =========================
        # BUILD SEGMENTS
        # =========================
        segments = []

        for seg in result.segments:
            segments.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "speaker": "Speaker",
                "text": seg.text.strip()
            })

        # =========================
        # FULL TRANSCRIPT
        # =========================
        full_text = " ".join(
            segment["text"]
            for segment in segments
        )

        # =========================
        # SUMMARIZATION
        # =========================
        logger.info("Generating summary...")

        summary = safe_summarize(full_text)

        # =========================
        # KEY POINTS
        # =========================
        key_points = [
            segment["text"]
            for segment in segments
            if len(segment["text"].split()) > 4
        ][:6]

        # =========================
        # DURATION
        # =========================
        duration_sec = segments[-1]["end"] if segments else 0

        mins = int(duration_sec // 60)
        secs = int(duration_sec % 60)

        logger.info("Processing completed successfully.")

        return {
            "success": True,
            "file_id": file_id,
            "file_ext": ext,
            "duration": f"{mins}:{secs:02d}",
            "segments": segments,
            "summary": summary,
            "key_points": key_points,
            "participants": ["Speaker"]
        }

    except Exception as e:
        traceback.print_exc()

        logger.error(f"Processing failed: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        # =========================
        # CLEAN TEMP FILES
        # =========================
        for path in [raw_path, audio_path]:
            if os.path.exists(path):
                os.remove(path)


# =========================
# MEDIA STREAM ROUTE
# =========================
@app.get("/media/{file_id}")
async def get_media(file_id: str):

    supported_extensions = [
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".mp3",
        ".wav",
        ".m4a"
    ]

    for ext in supported_extensions:

        path = os.path.join(
            MEDIA_FOLDER,
            f"{file_id}{ext}"
        )

        if os.path.exists(path):
            return FileResponse(path)

    raise HTTPException(
        status_code=404,
        detail="Media file not found"
    )


# =========================
# HEALTH CHECK
# =========================
@app.get("/health")
def health_check():
    return {
        "status": "running",
        "device": device
    }


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )