"""
test_query.py
-------------
Quick sanity-check for your Bangla-English RAG API.

Usage (from a second terminal, with the venv active or `requests` installed):

    python test_query.py
"""
import json
import requests

API_URL  = "http://127.0.0.1:8000/ask"
QUESTION = " বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"

def main() -> None:
    payload = {"question": QUESTION}
    resp = requests.post(API_URL, json=payload, timeout=30)
    resp.raise_for_status()                     # raises if HTTP != 200
    data = resp.json()

    print("ℹ️  Question:", QUESTION)
    print("✅ Answer  :", data.get("answer"), "\n")
    print("Full JSON:\n", json.dumps(data, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
