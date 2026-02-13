from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
import ffmpeg
import whisper
import torch
import json
from transformers import pipeline
from speechbrain.pretrained import SpeakerRecognition
import cv2
import pytesseract
from collections import defaultdict
from huggingface_hub import login
from pyannote.audio import Pipeline
import traceback

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# MODELS
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

whisper_model = whisper.load_model("tiny")

speaker_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmp_model"
)

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-6-6",
    device=0 if torch.cuda.is_available() else -1
)

# OCR path
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# =========================
# HUGGINGFACE TOKEN (SAFE)
# =========================
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("⚠️ HF_TOKEN not set in environment")

# =========================
# DIARIZATION INIT
# =========================
diarization_pipeline = None

try:
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN
    )
    print("✅ Diarization pipeline loaded successfully")
except Exception:
    print("❌ Diarization init failed:")
    traceback.print_exc()

# =========================
# AUDIO EXTRACTION
# =========================
def extract_audio(video_path, audio_path):
    ffmpeg.input(video_path).output(
        audio_path,
        format="wav",
        ar=16000,
        ac=1
    ).run(overwrite_output=True)

# =========================
# DIARIZATION
# =========================
def perform_speaker_diarization(audio_path):
    if diarization_pipeline is None:
        raise Exception("Diarization pipeline not initialized")

    diarization = diarization_pipeline(audio_path)

    segments = []
    for segment, speaker in diarization.speaker_diarization:
        segments.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "speaker": speaker
        })

    return segments

# =========================
# OCR NAME EXTRACTION
# =========================
def extract_names_from_frames(video_path):
    import re

    cap = cv2.VideoCapture(video_path)
    names = []

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    interval = frame_rate * 2
    timestamp_pattern = re.compile(r"\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}")

    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if idx % interval == 0:
            h, w = frame.shape[:2]
            roi = frame[h//2-50:h//2+50, w//2-200:w//2+200]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

            text = pytesseract.image_to_string(binary).strip()

            for line in text.split("\n"):
                line = line.strip()
                if not line or len(line) < 2:
                    continue
                if timestamp_pattern.search(line):
                    continue
                if any(c.isdigit() for c in line):
                    continue

                time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                print(f"🧾 OCR Name {time_sec:.2f}s: {line}")
                names.append((time_sec, line))

        idx += 1

    cap.release()
    return names

# =========================
# NAME MAPPING
# =========================
def map_names_to_diarization(speaker_segments, name_timestamps):
    mapping = {}

    for seg in speaker_segments:
        spk = seg["speaker"]
        start = seg["start"]
        end = seg["end"]

        if spk in mapping:
            continue

        best_name = None
        best_dist = 5.0

        for ts, name in name_timestamps:
            if start <= ts <= end:
                best_name = name
                break

            dist = min(abs(ts - start), abs(ts - end))
            if dist < best_dist:
                best_dist = dist
                best_name = name

        if best_name:
            mapping[spk] = best_name
            print(f"🎯 {spk} → {best_name}")

    return mapping

# =========================
# SEGMENT → SPEAKER
# =========================
def map_speakers_to_segments(trans_segments, speaker_segments):
    mapped = defaultdict(list)

    for seg in trans_segments:
        s = seg["start"]
        e = seg["end"]
        text = seg["text"]

        best_spk = "Unknown Speaker"
        best_overlap = 0

        for sp in speaker_segments:
            ov = max(0, min(e, sp["end"]) - max(s, sp["start"]))
            if ov > best_overlap:
                best_overlap = ov
                best_spk = sp["speaker"]

        mapped[best_spk].append(text)

    return mapped

# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    return jsonify({"message": "AI Meeting Summarizer API"})

@app.route("/upload", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    video = request.files["file"]
    filename = secure_filename(video.filename)
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    video.save(video_path)

    try:
        audio_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.wav")
        extract_audio(video_path, audio_path)

        result = whisper_model.transcribe(audio_path)
        transcription = result["text"]
        segments = result["segments"]

        speaker_segments = perform_speaker_diarization(audio_path)

        name_ts = extract_names_from_frames(video_path)
        mapping = map_names_to_diarization(speaker_segments, name_ts)
        named = map_speakers_to_segments(segments, speaker_segments)

        final = {}
        for spk, texts in named.items():
            name = mapping.get(spk, spk)
            final[name] = texts

        full_summary = summarizer(transcription, max_length=150, min_length=50)[0]["summary_text"]

        speaker_summaries = {}
        for spk, texts in final.items():
            txt = " ".join(texts).strip()
            if txt:
                speaker_summaries[spk] = summarizer(txt, max_length=100, min_length=30)[0]["summary_text"]

        participants = [p for p in speaker_summaries if p != "Unknown Speaker"]

        meeting_data = {
            "transcription": transcription,
            "summary": full_summary,
            "speaker_summaries": speaker_summaries,
            "participants": participants,
            "speaker_segments": speaker_segments
        }

        with open("meeting_data.json", "w") as f:
            json.dump(meeting_data, f, indent=2)

        return jsonify({"success": True, "participants": participants})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    q = data.get("question", "").lower()

    try:
        with open("meeting_data.json") as f:
            m = json.load(f)
    except:
        return jsonify({"response": "Upload meeting first"}), 400

    participants = m["participants"]
    summaries = m["speaker_summaries"]

    if "who" in q:
        return jsonify({"response": ", ".join(participants)})

    for name, text in summaries.items():
        if name.lower() in q:
            return jsonify({"response": f"{name}: {text}"})

    if "summary" in q:
        return jsonify({"response": m["summary"]})

    return jsonify({"response": "Ask about participants or summary"})

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True, port=5002)
