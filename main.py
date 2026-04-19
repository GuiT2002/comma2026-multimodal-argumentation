from groq import Groq
from dotenv import load_dotenv
import os
import base64
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

with open("all_examples.txt", 'r') as r:
  examples_list = r.readlines()

model = 2

test_n = 6

tests = ['',
  """Enthymeme: He said smoking causes cancer.

Context: No context for this image.
""",

"""
Enthymeme: The weatherman knows about weather forecasts.

Context: The user lives in Santa Rosa.
""",

"""
Enthymeme: Susan knows about mathematics.

Context: Susan is the user's teacher. The user didn't understand the solution to the first equation.
""",

"""
Enthymeme: All roses are flowers.

Context: The user received this.
""",

"""
Enthymeme: All BMWs are cars.

Context: The user just saw this.
""",

"""
Enthymeme: All laptops are electronic devices.

Context: The user wonders which specific laptop this is.
"""



]



### persona, role, elementos relevantes do ambiente para o papel da pessoa.

prompt = f"""

You are an AI assistant who has the responsability of formalizing argumentative structures that appear on the user's related context.
You will receive three elements as inputs: an enthymeme (an argument with missing premises or conclusion) , an image and a brief context related to what is happenning.

- The enthymeme will be an argumentative sentence that belongs to an argumentation scheme. You will not receive the full argumentation scheme, but only a part of it. The rest will be intentionally omitted.
- The image contains something that the user observed but only wrote a partial argumentative structure, which is the enthymeme mentioned earlier. So your task reagarding the image will be to capture the elements necessary to complete the argumentation scheme structure based on the enthymeme provided by the user.
- The context will be a brief description of how certain elements from the image might be related to the enthymeme written by the user.

Your task is to use enthymeme the user wrote, the image and the context provided to instantiate the variables of the argumentation scheme to which that argumentative sentence belongs.
The enthymeme is argumentative sentence of ONLY ONE of the argumentation schemes that will be described here. 
The following list shows every argumentation scheme to be considered, along with a single example for each argumentation scheme of how to translate the argument in natural language to a formal format:

{examples_list}


You must think step-by-step in order to correctly instantiate the full argumentation scheme. Be specific when instantiating the variables with what is in the image.
Your response should be specifically as the following structure:

<reasoning>
// here you will write your step-by-step reasoning process.
<reasoning_f>

<final_output>
// here you will write ONLY the final, formal format of the argument as specified earlier. Do not add anything other than the fully instantiated argumentation scheme.
<final_output_f>

Obey all these instructions and now consider the following case:

{tests[test_n]}


"""

image_path = f"images/test{test_n}.png"


if model == 1:
    client = Groq(api_key=api_key)

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


    base64_image = encode_image(image_path)




    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
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


if model == 2:
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    api_key = os.getenv("GEMINI_API_KEY")


    client = genai.Client()
    response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/png',
        ),
        prompt
        ]
    )

    print(response.text)

