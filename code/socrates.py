
import requests
import json

# Load API URL
try:
    with open("api.txt", "r") as file:
        API_URL = file.read().strip()
except FileNotFoundError:
    print("❌ Error: `api.txt` not found. Please create the file and add the API URL.")
    exit()

# Load System Prompt
try:
    with open("prompt.txt", "r", encoding="utf-8") as file:
        SYSTEM_PROMPT = file.read().strip()
except FileNotFoundError:
    print("❌ Error: `prompt.txt` not found. Please create the file and add the system prompt.")
    exit()

# Load User Prompt
try:
    with open("user_prompt.txt", "r", encoding="utf-8") as file:
        USER_PROMPT = file.read().strip()
except FileNotFoundError:
    print("❌ Error: `user_prompt.txt` not found. Please create the file and add a user query.")
    exit()

HEADERS = {"Content-Type": "application/json"}

def socratic_agent(prompt):
    """Send a Socratic-style message to the LLM and return its response."""
    payload = {
        "user": "single_socratic_agent",
        "model": "gpt4o",
        "system": SYSTEM_PROMPT,
        "prompt": [prompt],
        "stop": [],
        "temperature": 0.3,
        "top_p": 1.0,
        "max_tokens": 2000,
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        return response.json().get("response", "No response received")
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return "Error in processing request"

# Run the agent once
if __name__ == "__main__":
    print("🚀 Sending prompt to Socratic Agent...")
    response = socratic_agent(USER_PROMPT)
    print(f"\n🤖 Agent Response:\n{response}")

    with open("output.txt", "w", encoding="utf-8") as file:
        file.write(response)
    
    print("\n✅ Response saved to `output.txt`")

