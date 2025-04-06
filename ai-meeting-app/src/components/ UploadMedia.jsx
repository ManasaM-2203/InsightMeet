import { useState } from "react";

const UploadMedia = () => {
  const [file, setFile] = useState(null);
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file");
      return;
    }

    const formData = new FormData();
    formData.append("file", file); // ✅ Correct key now

    try {
      setLoading(true);
      const res = await fetch("http://127.0.0.1:5002/upload", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setResponse(data.summary || "Uploaded and processed successfully!");
    } catch (error) {
      console.error("Error uploading file:", error);
      setResponse("Failed to upload or process");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2 style={{ fontSize: "32px", fontWeight: "bold", margin: "20px 0" }}>Upload Meeting Video</h2>
      <input 
        type="file" 
        accept="video/*" 
        onChange={handleFileChange} 
        style={{ marginRight: "10px" }} 
      />
      <button 
        onClick={handleUpload}
        style={{
          backgroundColor: "#333",
          border: "none",
          padding: "5px 15px",
          borderRadius: "3px",
          cursor: "pointer",
          color: "white"
        }}
        disabled={loading}
      >
        {loading ? "Uploading..." : "Upload"}
      </button>
      <p>{response}</p>
    </div>
  );
};

export default UploadMedia;
