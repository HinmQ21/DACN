import os
import google.generativeai as genai
# Configure your API key first
import os
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def list_gemini_models():
    models = genai.list_models()
    for m in models:
        print("Name:", m.name)
        print("Description:", m.description)
        print("Supported methods:", m.supported_generation_methods)
        print("----")

if __name__ == "__main__":
    list_gemini_models()


