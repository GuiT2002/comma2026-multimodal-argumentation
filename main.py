from groq import Groq
from dotenv import load_dotenv
import os
import base64
import csv
from google import genai
from google.genai import types
from openai import OpenAI

load_dotenv()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

with open("all_examples.txt", 'r') as r:
  examples_list = r.readlines()



with open("tests.txt", 'r') as t:
    tests = t.readlines()

results = [[test, "", "", "", ""] for test in tests]

for model in range(2,3):
    for test_n in range(1, len(tests)+1):

        prompt = f"""

        You are an AI assistant responsible for formalizing argumentative structures in the context provided by the user.
        You will receive three inputs: an enthymeme (an argument with missing premises or a missing conclusion), an image, and a brief description of the context in which the situation takes place.

        - The enthymeme will be an argumentative sentence that belongs to an argumentation scheme. You will receive only part of the argumentation scheme. The rest will be intentionally omitted.
        - The image depicts something the user observed but described using only a partial argumentative structure: the enthymeme mentioned earlier. Your task regarding the image will be to identify the elements necessary to complete the structure of the argumentation scheme based on the enthymeme provided by the user.
        - The context will be a brief description of how certain elements in the image might relate to the enthymeme written by the user.

        Your task is to use the enthymeme the user wrote, the image, and the provided context to instantiate the variables of the argumentation scheme to which that argumentative sentence belongs.
        The enthymeme is an argumentative sentence belonging to ONLY ONE of the argumentation schemes described below.
        The following list presents every argumentation scheme to be considered, along with a single example for each scheme showing how to translate an argument from natural language into a formal representation:

        {examples_list}


        You must think step by step to correctly instantiate the full argumentation scheme. Be specific when instantiating the variables using the content of the image.
        Your response should follow this exact structure:

        <reasoning>
        // Write your step-by-step reasoning process here.
        <reasoning_f>

        <final_output>
        // Write ONLY the final formal representation of the argument here, as specified earlier. Do not add anything other than the fully instantiated argumentation scheme.
        <final_output_f>

        Follow all these instructions and consider the following case:

        {tests[test_n - 1]}
        """


        image_path = f"images/test{test_n}.png"





        if model == 1:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            api_key = os.getenv("GEMINI_API_KEY")


            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/png',
                ),
                prompt
                ]
            )

            results[test_n - 1][2] = response.text

        if model == 2:
            client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

            b64 = encode_image(image_path)

            response = client.chat.completions.create(
                model="deepseek-flash",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                        ],
                    }
                ],
            )
            results[test_n - 1][3] = response.choices[0].message.content

with open("results.csv", "w", newline="", encoding="utf-8-sig") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Test", "Image", "gemini-3.8-flash", "deepSeek-flash"])
    writer.writerows(results)
