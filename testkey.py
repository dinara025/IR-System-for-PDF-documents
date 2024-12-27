import os
import google.generativeai as palm
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Ensure the API key is loaded
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set. Check your .env file and environment variables.")

# Configure Google Generative AI with the API key
palm.configure(api_key=GOOGLE_API_KEY)

# Test the API by listing available models
try:
    models = list(palm.list_models())  # Convert the generator to a list
    print("Available Models:")
    for model in models:
        print(model)
except Exception as e:
    print("Error accessing Google Generative AI API:", e)
