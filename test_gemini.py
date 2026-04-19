from google import genai
from dotenv import load_dotenv
import os
from google.genai import types

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


with open('images/test2.png', 'rb') as f:
    image_bytes = f.read()

client = genai.Client()
response = client.models.generate_content(
    model='gemini-3-flash-preview',
    contents=[
      types.Part.from_bytes(
        data=image_bytes,
        mime_type='image/png',
      ),
      'Describe what is happening this image using formal first-order predicates. Answer ONLY with the formal first-order predicates, and no additional explanation.'
    ]
  )

print(response.text)