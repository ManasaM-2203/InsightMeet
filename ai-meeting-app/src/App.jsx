import React, { useState } from 'react';
import UploadMedia from './components/UploadMedia';
import TranscriptView from './components/TranscriptView';

function App() {
  const [result, setResult] = useState(null);

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 font-sans selection:bg-cyan-200 selection:text-cyan-900">
      <main className="max-w-5xl mx-auto p-4 sm:p-6 lg:p-8">
        {!result ? (
          <UploadMedia onResult={setResult} />
        ) : (
          <TranscriptView data={result} onReset={() => setResult(null)} />
        )}
      </main>
    </div>
  );
}

export default App;