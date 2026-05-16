import React, { useState } from 'react';

export default function TranscriptView({ data, onReset }) {
  const [activeTab, setActiveTab] = useState('transcript');
  
  const { 
    file_id, 
    file_ext, 
    duration, 
    segments, 
    summary, 
    key_points, 
    participants 
  } = data;

  const mediaUrl = `http://localhost:8000/media/${file_id}${file_ext}`;
  const isVideo = file_ext === '.mp4' || file_ext === '.mov' || file_ext === '.webm';

  const formatTime = (seconds) => {
    const min = Math.floor(seconds / 60);
    const sec = Math.floor(seconds % 60);
    return `${min}:${sec.toString().padStart(2, '0')}`;
  };

  return (
    <div className="animate-fade-in pb-10">
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-3">
          <span className="text-3xl">🧠</span>
          <h1 className="text-2xl font-bold text-gray-900">InsightMeet</h1>
        </div>
        <button 
          onClick={onReset}
          className="px-4 py-2 bg-white text-gray-700 font-medium rounded-full shadow-sm border border-gray-200 hover:bg-gray-50 hover:text-cyan-600 transition-colors focus:ring-2 focus:ring-cyan-500 focus:outline-none"
        >
          New Upload
        </button>
      </div>

      <div className="bg-white rounded-3xl shadow-xl overflow-hidden border border-gray-100 mb-8">
        <div className="p-4 bg-black flex justify-center">
          {isVideo ? (
            <video 
              controls 
              src={mediaUrl} 
              className="w-full max-h-[400px] rounded-xl object-contain"
            />
          ) : (
            <audio 
              controls 
              src={mediaUrl} 
              className="w-full max-w-3xl my-6"
            />
          )}
        </div>

        <div className="flex border-b border-gray-200">
          <button
            className={`flex-1 py-4 text-center font-semibold text-lg transition-colors ${
              activeTab === 'transcript' 
                ? 'text-cyan-600 border-b-2 border-cyan-600 bg-cyan-50/30' 
                : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
            }`}
            onClick={() => setActiveTab('transcript')}
          >
            Transcript
          </button>
          <button
            className={`flex-1 py-4 text-center font-semibold text-lg transition-colors ${
              activeTab === 'summary' 
                ? 'text-cyan-600 border-b-2 border-cyan-600 bg-cyan-50/30' 
                : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
            }`}
            onClick={() => setActiveTab('summary')}
          >
            Summary
          </button>
        </div>

        <div className="p-6 md:p-8 bg-gray-50/50">
          {activeTab === 'transcript' && (
            <div className="space-y-6 max-w-4xl mx-auto">
              {segments && segments.length > 0 ? (
                segments.map((seg, idx) => (
                  <div key={idx} className="bg-white p-5 rounded-2xl shadow-sm border border-gray-100 flex gap-4 hover:shadow-md transition-shadow">
                    <div className="flex-shrink-0 pt-1">
                      <span className="inline-block bg-cyan-100 text-cyan-800 text-xs px-2 py-1 rounded-md font-mono font-medium">
                        {formatTime(seg.start)}
                      </span>
                    </div>
                    <div>
                      <h4 className="font-bold text-gray-900 mb-1">{seg.speaker}</h4>
                      <p className="text-gray-700 leading-relaxed">{seg.text}</p>
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-center text-gray-500 py-10">No transcript segments available.</p>
              )}
            </div>
          )}

          {activeTab === 'summary' && (
            <div className="max-w-4xl mx-auto space-y-8">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="md:col-span-2 bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
                  <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                    <svg className="w-6 h-6 text-cyan-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                    Overview
                  </h3>
                  <p className="text-gray-700 leading-relaxed text-lg">{summary}</p>
                </div>
                
                <div className="space-y-6">
                  <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
                    <h3 className="text-lg font-bold text-gray-900 mb-3 text-cyan-600">Duration</h3>
                    <p className="text-2xl font-mono text-gray-700">{duration}</p>
                  </div>
                  
                  <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
                    <h3 className="text-lg font-bold text-gray-900 mb-3 text-cyan-600">Participants</h3>
                    <div className="flex flex-wrap gap-2">
                      {participants && participants.map((p, i) => (
                        <span key={i} className="bg-gray-100 text-gray-800 px-3 py-1 rounded-full text-sm font-medium">
                          {p}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white p-6 md:p-8 rounded-2xl shadow-sm border border-gray-100">
                <h3 className="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
                  <svg className="w-6 h-6 text-cyan-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"></path></svg>
                  Key Points
                </h3>
                <ul className="space-y-4">
                  {key_points && key_points.map((point, i) => (
                    <li key={i} className="flex gap-4">
                      <span className="flex-shrink-0 w-8 h-8 flex items-center justify-center bg-cyan-100 text-cyan-700 font-bold rounded-full">
                        {i + 1}
                      </span>
                      <p className="text-gray-700 text-lg pt-1">{point}</p>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}