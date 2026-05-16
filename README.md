# InsightMeet 🧠

InsightMeet is an intelligent meeting summarization application that takes your meeting recordings (audio or video) and automatically generates transcripts, summaries, key points, and participant lists.

## ✨ Features
- **Easy Upload:** Drag-and-drop or click to upload meeting recordings (MP4, MP3, WAV, etc.).
- **Smart Transcripts:** Generates timestamped, speaker-diarized transcripts directly tied to your media file.
- **Intelligent Summaries:** Automatically extracts an overview, key discussion points, and participant lists.
- **Modern UI:** Built with React, Vite, and styled beautifully using Tailwind CSS.
- **Fast Backend:** Powered by FastAPI for quick, reliable processing.

## 🛠️ Tech Stack
**Frontend:**
- React
- Vite
- Tailwind CSS v3

**Backend:**
- FastAPI
- Python
- Uvicorn

## 🚀 Setup Instructions

### 1. Backend Setup
1. Navigate to the project root directory:
   ```bash
   cd InsightMeet
   ```
2. Activate the Python virtual environment:
   ```bash
   source venv/bin/activate
   ```
3. Install the required backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the FastAPI development server:
   ```bash
   uvicorn server:app --host 0.0.0.0 --port 8000 --reload
   ```

### 2. Frontend Setup
1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd InsightMeet/ai-meeting-app
   ```
2. Install the necessary node modules:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
4. Open your browser and navigate to **[http://localhost:5173/](http://localhost:5173/)**.

## 📂 Project Structure
```text
InsightMeet/
├── server.py              # FastAPI backend entry point
├── requirements.txt       # Backend dependencies
├── uploads/               # Temporary storage for uploaded media
├── media/                 # Processed media storage for playback
├── venv/                  # Python virtual environment
└── ai-meeting-app/        # React + Vite frontend source code
    ├── src/
    │   ├── App.jsx        # Main application state and layout
    │   ├── index.css      # Tailwind directives
    │   ├── main.jsx       # React entry point
    │   └── components/
    │       ├── UploadMedia.jsx    # Media upload dropzone
    │       └── TranscriptView.jsx # Transcript and summary display tabs
    ├── index.html
    ├── package.json
    ├── tailwind.config.js # Tailwind CSS configuration
    └── vite.config.js     # Vite configuration
```

## 🎯 How to Use
1. Ensure both the backend and frontend servers are running.
2. Open the frontend application in your web browser.
3. Drag and drop (or click to upload) an audio or video file of a meeting.
4. Wait for the processing to complete (you'll see a pulsing loading indicator).
5. Explore the results! Switch between the **Transcript** tab to read along with the audio/video, or the **Summary** tab to get a quick overview and key points.
