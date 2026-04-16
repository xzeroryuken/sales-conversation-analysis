from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
model = os.getenv("model")

client = Groq(api_key=api_key)
completion = client.chat.completions.create(
    model=model,
    messages=[
      {
        "role": "user",
        "content": ""
      }
    ],
    temperature=1,
    max_completion_tokens=8192,
    top_p=1,
    reasoning_effort="medium",
    stream=False,
    stop=None
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
