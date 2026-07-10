import requests
import json

response = requests.post(
    "https://pfjtvbl6ck.execute-api.us-east-1.amazonaws.com/prod/query",
    headers={
        "x-api-key": "PASTE_REAL_KEY_HERE",
        "Content-Type": "application/json"
    },
    json={"question": "What was Microsoft total revenue for fiscal years 2022, 2023, and 2024?", "company": "microsoft"},
    timeout=30
)
data = response.json()
print(json.dumps(data, indent=2)[:3000])
