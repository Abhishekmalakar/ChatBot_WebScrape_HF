import pickle
import requests
import os
import time
from dotenv import load_dotenv
load_dotenv()

# === Setup HF API ===
# API_URL = "https://api-inference.huggingface.co/models/consciousAI/question-answering-roberta-base-s-v2"
API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-distilled-squad"

# Get API key
api_key = os.getenv('HUGGINGFACE_API_KEY')  # Directly fetch from .env

if not api_key:
    api_key = str(input('Enter HF API KEY:'))  # Fallback if not in the .env

if not api_key:
    raise ValueError("API_KEY environment variable is not set")

headers = {"Authorization": f"Bearer {api_key}"}

# === Define functions ===
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def chat_with_gpt(question, context, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": {"question": question, "context": context}}
        )

        if response.status_code != 200:
            return {"answer": f"[HTTP Error] {response.status_code}"}

        output = response.json()
        print("🔍 RAW API RESPONSE:", output)  # Debugging line

        if isinstance(output, dict):
            if 'answer' in output:
                return output
            elif 'error' in output:
                return {"answer": f"[ERROR] {output['error']}"}
            else:
                return {"answer": "[Unexpected format - No 'answer' key]"}

        elif isinstance(output, list) and 'answer' in output[0]:
            return output[0]  # Assume it's a list of answers
        else:
            return {"answer": "[Invalid response format]"}
    except requests.exceptions.RequestException as e:
        print("Request exception:", e)
        return {"answer": f"[Request Exception] {str(e)}"}
    except Exception as e:
        print("An error occurred:", e)
        return {"answer": f"[Exception] {str(e)}"}


# === Auto-create data.pkl if missing ===
data_path = "Web-Chatbot/data.pkl"
os.makedirs("Web-Chatbot", exist_ok=True)

if not os.path.exists(data_path):
    print("data.pkl not found. Creating default data file...")
    default_data = {
        "context": "BotPenguin is a chatbot building platform that helps businesses automate conversations with customers."
    }
    with open(data_path, "wb") as f:
        pickle.dump(default_data, f)
    print("Created data.pkl with default context.")

# === Load context ===
with open(data_path, "rb") as f:
    loaded_data = pickle.load(f)

context = loaded_data.get('context', '')

# === Run chatbot ===
question = "what is BotPenguin?"
output = chat_with_gpt(question, context, api_key)  # Fixed: Passing the api_key

try:
    print(question, '\n', "Generating answer: ", output["answer"])
except Exception as e:
    print("An error occurred while generating the answer:", e)

# === Interactive Chat ===
while True:
    question = input("Ask a question or type 'exit': ")
    if question.lower() == 'exit':
        break
    try:
        output = chat_with_gpt(question, context, api_key)  # Fixed: Passing the api_key
        if output:
            print(f"Generating answer: {output['answer']}\n")
        else:
            print("No output received from the model.")
    except Exception as e:
        print("An error occurred:", e)
