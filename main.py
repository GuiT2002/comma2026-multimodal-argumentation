from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")


client = Groq(api_key=api_key)


response = client.chat.completions.create(messages=[{'role': 'user', 'content': 'What is argumentation? Be very concise'}],
                                          model='openai/gpt-oss-120b',
                                          temperature=0.4)

print(response.choices[0].message.content)


