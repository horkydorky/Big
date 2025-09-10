import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the API key
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file.")
    genai.configure(api_key=api_key)
    print("API Key configured successfully.")
except Exception as e:
    print(f"Error configuring API Key: {e}")
    exit()

# List the available models that support content generation
print("\n--- Available Models for 'generateContent' ---")
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(f"Model Name: {m.name}")

print("\n--------------------------------------------")
print("ACTION: Copy one of the 'Model Name' values from the list above.")
print("Then, paste it into the 'model=' parameter in your app.py file.")