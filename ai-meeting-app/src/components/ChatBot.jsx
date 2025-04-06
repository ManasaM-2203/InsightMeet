import { useState, useEffect } from "react";
import axios from "axios";

export default function Chatbot({ videoProcessed, participants }) {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [participantsList, setParticipantsList] = useState(participants || []);

  // Update participants if props change
  useEffect(() => {
    if (participants) {
      setParticipantsList(participants);
    }
  }, [participants]);

  const askQuestion = async () => {
    if (!question.trim()) return;
    setLoading(true);
    try {
      const res = await axios.post("http://127.0.0.1:5002/ask", {
        question,
      });
      setResponse(res.data.response);
    } catch (err) {
      console.error("Error asking question:", err);
      setResponse("There was an error processing your request.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chatbot-container">
      {videoProcessed && (
        <div className="analysis-complete">
          <h3>Video analysis complete!</h3>
          <p>I'm ready to answer your questions about this meeting.</p>
          
          {participantsList.length > 0 && (
            <div className="participants-list">
              <h4>Meeting Participants:</h4>
              <ul>
                {participantsList.map((name, index) => (
                  <li key={index}>{name}</li>
                ))}
              </ul>
              <p>You can ask about the whole meeting or specific participants.</p>
            </div>
          )}
        </div>
      )}

      <textarea
        style={{
          width: "100%",
          padding: "10px",
          borderRadius: "5px",
          border: "1px solid #ccc",
          marginBottom: "10px",
          marginTop: "10px",
        }}
        placeholder="Ask a question about the meeting..."
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        rows={2}
        className="chat-input"
      />
      <button onClick={askQuestion} 
      style={{
        backgroundColor: "#333",
        border: "none",
        padding: "5px 15px",
        borderRadius: "3px",
        cursor: "pointer",
        color: "white"
      }}
      disabled={loading} className="ask-button">
        {loading ? "Asking..." : "Ask"}
      </button>
      
      {response && (
        <div className="chat-response">
          <strong>Response:</strong>
          <p>{response}</p>
        </div>
      )}
    </div>
  );
}