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

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Configure CORS properly

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
whisper_model = whisper.load_model("tiny")
speaker_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", device=0 if torch.cuda.is_available() else -1)

# Tesseract OCR path (update this path as needed for your system)
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

def extract_audio(video_path, audio_path):
    ffmpeg.input(video_path).output(audio_path, format='wav').run(overwrite_output=True)

def extract_names_from_frames(video_path):
    import re
    cap = cv2.VideoCapture(video_path)
    name_timestamps = []
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = frame_rate * 2  # Every 2 seconds
    
    # Timestamp pattern to exclude
    timestamp_pattern = re.compile(r'\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}')
    
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if idx % frame_interval == 0:
            h, w = frame.shape[:2]
            # Extract only the center region where the name appears
            # Based on your screenshot, the name appears in the center in a black box
            center_y = h // 2
            center_x = w // 2
            
            # Create a region of interest around the center where the name is likely to be
            # Adjust these values based on your specific video layout
            name_roi = frame[center_y - 50:center_y + 50, center_x - 200:center_x + 200]
            
            # Convert to grayscale for OCR
            name_roi_gray = cv2.cvtColor(name_roi, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to make text more visible
            _, binary = cv2.threshold(name_roi_gray, 200, 255, cv2.THRESH_BINARY)
            
            # Get text from center region
            text = pytesseract.image_to_string(binary).strip()
            
            # Filter the text to remove empty lines and timestamps
            for line in text.split('\n'):
                clean_line = line.strip()
                # Skip empty lines, timestamps, or very short text
                if not clean_line or len(clean_line) < 2 or timestamp_pattern.search(clean_line):
                    continue
                    
                # Check if this looks like a name (reasonable length, no digits)
                if len(clean_line) < 30 and clean_line.isprintable() and not any(c.isdigit() for c in clean_line):
                    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    print(f"🧾 OCR Detected Name at {current_time:.2f}s: {clean_line}")
                    # Add to our list of detected names with timestamps
                    name_timestamps.append((current_time, clean_line))
        
        idx += 1
        
    cap.release()
    return name_timestamps

def map_names_to_segments(transcription_segments, name_timestamps):
    mapped = defaultdict(list)
    
    # If no names were detected, use "Unknown Speaker" for everything
    if not name_timestamps:
        for segment in transcription_segments:
            mapped["Unknown Speaker"].append(segment['text'])
        return mapped
    
    # Sort name_timestamps by time for efficient processing
    name_timestamps.sort(key=lambda x: x[0])
    
    # For each segment, find the most recent speaker name that appeared before it
    for segment in transcription_segments:
        segment_start = segment['start']
        segment_text = segment['text']
        
        # Default speaker name if we can't find a better match
        current_speaker = "Unknown Speaker"
        
        # Find the most recent name that appeared before this segment started
        for ts, name in name_timestamps:
            # Only consider names that appeared before or right at the segment start
            # (with a small buffer for synchronization issues)
            if ts <= segment_start + 0.5:
                current_speaker = name
            else:
                # Since name_timestamps is sorted, once we find a timestamp
                # after our segment, we can stop looking
                break
                
        # Add this segment's text to the appropriate speaker's collection
        mapped[current_speaker].append(segment_text)
        
    return mapped

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Welcome to the AI Meeting Summarizer API. Use /upload to POST a video file."}), 200

@app.route("/status", methods=["GET"])
def status():
    # Check if meeting data exists and return video processing status
    try:
        with open("meeting_data.json", "r") as f:
            meeting_data = json.load(f)
        return jsonify({
            "status": "Server is running",
            "videoProcessed": True,
            "participants": meeting_data.get("participants", [])
        }), 200
    except FileNotFoundError:
        return jsonify({
            "status": "Server is running",
            "videoProcessed": False
        }), 200

@app.route("/upload", methods=["POST"])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
        
    video = request.files['file']
    if video.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    filename = secure_filename(video.filename)
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    video.save(video_path)
    
    try:
        audio_filename = f"{uuid.uuid4()}.wav"
        audio_path = os.path.join(UPLOAD_FOLDER, audio_filename)
        extract_audio(video_path, audio_path)
        print("✅ Audio extracted:", audio_path)
        
        result = whisper_model.transcribe(audio_path, verbose=False, word_timestamps=False)
        transcription = result.get("text", "")
        segments = result.get("segments", [])
        print("✅ Transcription completed. Segments:", len(segments))
        
        if not transcription.strip():
            return jsonify({"error": "Transcription failed or resulted in empty output."}), 500
            
        name_timestamps = extract_names_from_frames(video_path)
        print("✅ Names extracted from frames:", name_timestamps)
        
        named_segments = map_names_to_segments(segments, name_timestamps)
        print("✅ Named segments mapped")
        
        full_summary = summarizer(transcription, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        print("✅ Full summary generated")
        
        speaker_summaries = {}
        for name, texts in named_segments.items():
            combined_text = " ".join(texts).strip()
            if not combined_text:
                speaker_summaries[name] = "No speech content found."
                continue
                
            try:
                summary = summarizer(combined_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
                speaker_summaries[name] = summary
                print(f"✅ Summary for {name}")
            except Exception as e:
                speaker_summaries[name] = "Failed to summarize."
                print(f"⚠️ Failed to summarize for {name}: {e}")
                
        # Get a list of unique participants (excluding "Unknown Speaker")
        participants = [name for name in speaker_summaries.keys() if name != "Unknown Speaker"]
        
        meeting_data = {
            "transcription": transcription,
            "summary": full_summary,
            "speaker_summaries": speaker_summaries,
            "participants": participants
        }
        
        with open("meeting_data.json", "w") as f:
            json.dump(meeting_data, f, indent=4)
        print("✅ meeting_data.json saved")
        
        # Return success message with participants list instead of the summary
        return jsonify({
            "success": True,
            "message": "Video analysis complete. I'm ready to answer your questions about this meeting.",
            "participants": participants
        })
        
    except Exception as e:
        print("🔥 Error during processing:", str(e))
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/ask", methods=["POST", "OPTIONS"])
def ask_about_meeting():
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        return "", 200
        
    data = request.get_json()
    question = data.get("question", "").lower()
    
    try:
        with open("meeting_data.json", "r") as f:
            meeting_data = json.load(f)
    except FileNotFoundError:
        return jsonify({"response": "Meeting data not found. Please upload a video first."}), 400
        
    participants = meeting_data.get("participants", [])
    
    if "who" in question and ("participated" in question or "in the meeting" in question):
        if participants:
            participant_list = ", ".join(participants)
            return jsonify({"response": f"The participants in this meeting were: {participant_list}"})
        else:
            return jsonify({"response": "I couldn't identify any named participants in this meeting."})
    elif "whole summary" in question or "meeting summary" in question:
        return jsonify({"response": meeting_data.get("summary", "No summary found.")})
        
    # Check if question mentions any participant name
    mentioned_name = None
    for name in meeting_data.get("speaker_summaries", {}):
        if name.lower() in question and name != "Unknown Speaker":
            mentioned_name = name
            break
            
    if mentioned_name:
        return jsonify({"response": meeting_data["speaker_summaries"].get(mentioned_name, "No summary found for this participant.")})
    else:
        return jsonify({
            "response": "What would you like to know about this meeting? You can ask for the whole summary, or about specific participants like: " +
            ", ".join(participants[:3]) + (", etc." if len(participants) > 3 else ".")
        })

if __name__ == "__main__":
    app.run(debug=True, port=5002)