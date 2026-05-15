from abc import ABC, abstractmethod
from groq import Groq, RateLimitError, APIStatusError
from dotenv import load_dotenv, find_dotenv
import os


FALLBACK_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "groq/compound",
    "openai/gpt-oss-20b",
    "groq/compound-mini",
    "llama-3.1-8b-instant",
]


class LLMClient(ABC):
    @abstractmethod
    def get_response(self, messages: list) -> str: ...


class GroqClient(LLMClient):
    def __init__(self, models: list = None):
        load_dotenv(find_dotenv())
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.models = models or FALLBACK_MODELS

    def get_response(self, messages: list) -> str:
        last_error = None

        for model in self.models:
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                    max_completion_tokens=8192,
                    top_p=1,
                    stream=False,
                    stop=None,
                )
                return completion.choices[0].message.content

            except RateLimitError as e:
                print(f"  Rate limit hit on {model}, trying next...")
                last_error = e
                continue
            except APIStatusError as e:
                if e.status_code == 413:
                    print(f"  Request too large for {model}, trying next...")
                    last_error = e
                    continue
                raise

        raise last_error
