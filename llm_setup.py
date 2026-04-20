from groq import Groq
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

api_key = os.getenv("GROQ_API_KEY")
model = os.getenv("MODEL")

print("API KEY:", api_key)
print("MODEL:", model)

client = Groq(api_key=api_key)

def get_response(messages):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_completion_tokens=8192,
        top_p=1,
        stream=False,
        stop=None
    )

    response = completion.choices[0].message.content

    return response
