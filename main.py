from groq import Groq
from dotenv import load_dotenv
import os
import base64

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")


client = Groq(api_key=api_key)

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

image_path = "images/dog.jpeg"

base64_image = encode_image(image_path)

response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ],
    model="meta-llama/llama-4-scout-17b-16e-instruct",
)

print(f'\033[92m{response.choices[0].message.content}\033[0m')


