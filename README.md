<h1>Project: InsightMeet - AI-Powered Meeting Summarizer</h1>

<h3>Missed the meeting? Don’t worry! Just upload the recording and InsightMeet will handle the rest.</h3>
InsightMeet is your smart assistant for turning meeting recordings into clear, actionable insights. Just upload a video or audio file—InsightMeet transcribes, identifies speakers, summarizes discussions, and lets you ask follow-up questions through a chatbot powered by a Large Language Model (LLM).

🚀 Features
🎥 Upload meeting videos or audio
🧠 Transcription using Whisper
🗣️ Speaker detection + name mapping (via OCR)
🧾 Summaries per speaker & full meeting
💬 Chatbot for smart Q&A using LLM + vector DB
📁 Clean structured output in JSON

🛠️ Tech Stack
Frontend: React.js
Backend: Python (Flask)
AI: Whisper, Transformers, Tesseract OCR
Search: FAISS or ChromaDB
LLM: Mistral, LLaMA, GPT (yet to be added)

nsightMeet — AI Meeting Summarizer
Upload your Zoom, Google Meet, or any meeting recording (video or audio) and get instant AI-powered insights: transcription, speaker identification, summary, action items, and more.

🛠️ Prerequisites
Make sure you have these installed on your system before starting:

Python 3.11
Node.js & npm
ffmpeg (brew install ffmpeg)
Tesseract OCR (brew install tesseract)


🔑 Environment Variables
Create a .env file in the root of the project (same folder as server.py) and add your API keys:
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
GEMINI_API_KEY=AIzaxxxxxxxxxxxxxxxx
How to get each key:

HF_TOKEN → huggingface.co/settings/tokens

Also accept model licenses at:

huggingface.co/pyannote/speaker-diarization-3.1
huggingface.co/pyannote/segmentation-3.0

GROQ_API_KEY → console.groq.com
GEMINI_API_KEY → aistudio.google.com


🚀 Running the Project
Backend
bash# Step 1 — Navigate to the project folder
cd InsightMeet

Step 2 — Activate the virtual environment
source venv/bin/activate

Step 3 — Install dependencies
pip install -r requirements.txt

Step 4 — Start the server
python3 server.py
The backend will start at: http://127.0.0.1:5003
You should see:
✅ Diarization loaded
* Running on http://127.0.0.1:5003

Frontend
Open a new terminal tab/window and run:
bash
Step 1 — Navigate to the frontend folder
cd ai-meeting-app

Step 2 — Install dependencies (first time only)
npm install
Step 3 — Start the frontend
npm run dev
