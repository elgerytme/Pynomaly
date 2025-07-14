import requests

API_KEY = "your-api-key-here"
BASE_URL = "https://api.pynomaly.com"

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# Health check
response = requests.get(f"{BASE_URL}/health", headers=headers)
print(response.json())

# List detectors
response = requests.get(f"{BASE_URL}/api/v1/detectors", headers=headers)
print(response.json())
