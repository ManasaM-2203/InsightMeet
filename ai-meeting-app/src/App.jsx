import { useState, useEffect } from "react";
import UploadMedia from "./components/ UploadMedia.jsx";
import Chatbot from "./components/ChatBot.jsx";
import axios from "axios";
import "./App.css";
import "./index.css";

export default function App() {
  const [serverStatus, setServerStatus] = useState("Checking...");
  const [videoProcessed, setVideoProcessed] = useState(false);
  const [participants, setParticipants] = useState([]);
  const [uploadStatus, setUploadStatus] = useState("idle"); // idle, uploading, success, error

  useEffect(() => {
    const checkServer = async () => {
      try {
        const response = await axios.get("http://127.0.0.1:5002/status");
        setServerStatus("Connected");
        
        // If the server reports video processing status, update it here
        if (response.data && response.data.videoProcessed !== undefined) {
          setVideoProcessed(response.data.videoProcessed);
        }
        
        // If the server provides participants list, update it here
        if (response.data && response.data.participants) {
          setParticipants(response.data.participants);
        }
      } catch (error) {
        setServerStatus("Disconnected - Check if server is running");
        console.error("Server connection error:", error);
      }
    };

    checkServer();
    // Set up periodic checking every 30 seconds
    const intervalId = setInterval(checkServer, 30000);
    // Clean up interval on component unmount
    return () => clearInterval(intervalId);
  }, []);

  // Handler for upload completion
  const handleUploadComplete = (result) => {
    if (result.success) {
      setUploadStatus("success");
      setVideoProcessed(true);
      if (result.participants) {
        setParticipants(result.participants);
      }
    } else {
      setUploadStatus("error");
    }
  };

  // Handler for upload progress
  const handleUploadProgress = (status) => {
    setUploadStatus(status);
  };

  return (
    <div className="app-container">
      {/* Main content area */}
      <div className="content-container">
        {/* Header */}
        <header className="main-header">
          <h1 className="title">AI Meeting Summarizer</h1>
          <p className="subtitle">
            Upload your meeting recording and get instant insights
          </p>
        </header>

        {/* Server Status Indicator */}
        <div className="status-indicator">
          <div
            className={`status-dot ${
              serverStatus.includes("Connected") ? "connected" : "disconnected"
            }`}
          />
          <span className="status-text">
            Server status: {serverStatus}
          </span>
        </div>

        {/* Upload Section */}
        <div className="panel">
          <h2 className="panel-title">Upload Meeting Recording</h2>
          <UploadMedia 
            onUploadComplete={handleUploadComplete} 
            onUploadProgress={handleUploadProgress}
          />
          {uploadStatus === "uploading" && (
            <p className="upload-status">Processing your video...</p>
          )}
          {uploadStatus === "success" && (
            <p className="upload-status success">Video processed successfully!</p>
          )}
          {uploadStatus === "error" && (
            <p className="upload-status error">Error processing video. Please try again.</p>
          )}
        </div>

        {/* Chatbot Section */}
        <div className="panel">
          <h2 className="panel-title">Ask About the Meeting</h2>
          <Chatbot 
            videoProcessed={videoProcessed} 
            participants={participants}
          />
        </div>
      </div>

      {/* Footer */}
      <footer className="footer">
        <div id="footer">
          AI Meeting Summarizer © 2025
        </div>
      </footer>
    </div>
  );
}