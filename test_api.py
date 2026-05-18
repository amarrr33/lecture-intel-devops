import os

key = os.getenv("GOOGLE_API_KEY")

if not key:
    raise Exception("API key missing")

print("API key exists ✅")