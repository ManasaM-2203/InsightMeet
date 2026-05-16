import React, { useState } from 'react';

export default function UploadMedia({ onResult }) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState('');

  const handleUpload = async (file) => {
    if (!file) return;
    setIsUploading(true);
    setError('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/process-file/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed. Please try again.');
      }

      const data = await response.json();
      onResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsUploading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    handleUpload(file);
  };

  const handleChange = (e) => {
    const file = e.target.files[0];
    handleUpload(file);
  };

  return (
    <div className="flex flex-col items-center justify-center py-20 px-4">
      <div className="text-center mb-10">
        <div className="text-6xl mb-4 animate-bounce">🧠</div>
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-gray-900 mb-2">
          InsightMeet
        </h1>
        <p className="text-lg text-gray-500">
          Upload your meeting recording to get transcripts and intelligent summaries.
        </p>
      </div>

      <div 
        className={`w-full max-w-xl p-8 rounded-3xl border-4 border-dashed transition-all duration-300 flex flex-col items-center justify-center text-center cursor-pointer bg-white shadow-xl hover:shadow-2xl hover:border-cyan-400 group ${
          isDragging ? 'border-cyan-500 bg-cyan-50' : 'border-gray-200'
        }`}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => document.getElementById('file-upload').click()}
      >
        <input 
          id="file-upload" 
          type="file" 
          className="hidden" 
          onChange={handleChange}
          accept="audio/*,video/*"
        />
        
        {isUploading ? (
          <div className="flex flex-col items-center py-10">
            <svg className="animate-spin h-12 w-12 text-cyan-500 mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <p className="text-xl font-medium text-gray-700 animate-pulse">Processing your media...</p>
            <p className="text-sm text-gray-500 mt-2">This might take a moment.</p>
          </div>
        ) : (
          <div className="py-12">
            <svg className="w-16 h-16 text-gray-400 mx-auto mb-4 group-hover:text-cyan-500 transition-colors duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
            <p className="text-xl font-semibold text-gray-700 mb-1">
              Click to upload or drag and drop
            </p>
            <p className="text-gray-500">
              Audio or Video files (MP4, MP3, WAV, etc.)
            </p>
          </div>
        )}
      </div>

      {error && (
        <div className="mt-6 p-4 bg-red-50 text-red-700 rounded-xl max-w-xl w-full text-center border border-red-200">
          <p className="font-medium">Error</p>
          <p>{error}</p>
        </div>
      )}
    </div>
  );
}