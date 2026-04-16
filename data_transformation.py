from datasets import load_dataset
import pandas as pd
import tiktoken

def get_conversations():
    ds = load_dataset("goendalf666/sales-conversations")
    df = pd.DataFrame(ds["train"])
    df["conversation"] = df.apply(lambda x: '\n'.join(value for value in x if not pd.isnull(value)), axis=1)
    df = df[["conversation"]]
    return df

def get_token_count(df):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = df["conversation"].apply(lambda x: len(enc.encode(x)))
    avg_tokens = tokens.mean()
    return avg_tokens


