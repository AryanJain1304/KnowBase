import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

def query_ollama_stream(prompt: str, model: str = MODEL_NAME):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        data = line.decode('utf-8')
                        json_data = json.loads(data)
                        yield json_data.get("response", "")
                    except Exception as e:
                        yield f"\n[Stream Parse Error: {e}]\n"
    except Exception as e:
        yield f"\n[Request Error: {e}]\n"