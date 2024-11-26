import time
import requests

API_URL = "https://api-inference.huggingface.co/models/armaanahuja7777/thaparChatbot"
headers = {
    "Authorization": "Bearer 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",  # Replace with your Hugging Face API key
}

data = {
    "inputs": "Hello, how are you?"
}

# Retry logic
retry_count = 5
for attempt in range(retry_count):
    response = requests.post(API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        print(response.json())  # If successful, print the response
        break
    elif response.status_code == 503:  # Model still loading
        print("Model is still loading, retrying...")
        time.sleep(20)  # Wait for 20 seconds before retrying
    else:
        print(f"Error: {response.status_code}, {response.text}")
        break
