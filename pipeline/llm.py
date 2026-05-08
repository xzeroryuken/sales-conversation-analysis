from abc import ABC, abstractmethod
from groq import Groq
from dotenv import load_dotenv, find_dotenv
import os


class LLMClient(ABC):
    @abstractmethod
    def get_response(self, messages: list) -> str: ...


class GroqClient(LLMClient):
    def __init__(self):
        load_dotenv(find_dotenv())
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("MODEL")

    def get_response(self, messages: list) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_completion_tokens=8192,
            top_p=1,
            stream=False,
            stop=None,
        )
        return completion.choices[0].message.content
