import json
import os
import re  # Import regex for extracting speaker number

def load_meeting_data():
    """Load the processed meeting data from a JSON file."""
    data_file = "meeting_data.json"
    
    if not os.path.exists(data_file):
        print("Error: Meeting data file not found. Please run `full_pipeline.py` first.")
        return None
    
    try:
        with open(data_file, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON data. Ensure `meeting_data.json` is correctly formatted.")
        return None

def chatbot():
    """Interactive chatbot interface."""
    meeting_data = load_meeting_data()
    
    if meeting_data is None:
        return
    
    print("\nChatbot is ready! Type your question below (or type 'exit' to quit).")
    
    # DEBUG: Print the speaker summaries keys
    print("\nDEBUG: Speaker Summaries Keys:", meeting_data.get("speaker_summaries", {}).keys())

    while True:
        user_input = input("\nYou: ").strip().lower()

        if user_input in ["exit", "quit", "bye"]:
            print("\nChatbot: Goodbye!")
            break
        
        # Handle full meeting summary request
        elif "whole summary" in user_input:
            print("\nChatbot: Here is the summary of the entire meeting:\n")
            print(meeting_data.get("summary", "No summary available."))
        
        # Handle specific speaker summary request
        elif "speaker" in user_input:
            match = re.search(r"speaker\s*(\d+)", user_input)  # Extracts only the number
            if match:
                speaker_id = match.group(1)
                speaker_key = f"Speaker {speaker_id}"
                
                # DEBUG: Print key being searched
                print(f"DEBUG: Searching for {speaker_key} in JSON")
                
                speaker_summary = meeting_data.get("speaker_summaries", {}).get(speaker_key, "Speaker not found.")
                print(f"\nChatbot: {speaker_summary}")
            else:
                print("\nChatbot: Please specify a speaker number (e.g., 'What did Speaker 1 say?').")
        
        # Default response
        else:
            print("\nChatbot: Sorry, I didn't understand. Try asking for 'whole summary' or about a specific speaker (e.g., 'What did Speaker 1 say?').")

if __name__ == "__main__":
    chatbot()
