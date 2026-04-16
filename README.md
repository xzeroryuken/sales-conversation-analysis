# Sales Conversation Analysis

A pipeline that analyzes synthetic sales conversations using LLMs to classify customer intent, then clusters similar intents using embeddings to surface actionable sales insights.

## Tech Stack
- Python
- Groq API (Llama 4 Scout)
- HuggingFace Datasets
- pandas, tiktoken

## Project Structure
- `data_transformation.py` — loads and transforms raw conversation data
- `llm_classification_prompts.py` — prompt templates for intent classification
- `llm_setup.py` — LLM client setup (model-agnostic via .env)
- `main.py` — entry point, orchestrates the full pipeline

## Setup
1. Clone the repo
2. Create a virtual environment and activate it
3. Run `pip install -r requirements.txt`
4. Create a `.env` file with your credentials:
