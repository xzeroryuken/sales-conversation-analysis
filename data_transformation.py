from datasets import load_dataset
import pandas as pd

# Set options to display everything
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


ds = load_dataset("goendalf666/sales-conversations")

df = pd.DataFrame(ds["train"])
# print(df.head())

df["conversation"] = df.apply(lambda x: '\n'.join(value for value in x if not pd.isnull(value)), axis=1)
    
print(df["conversation"].head())

df = df[["conversation"]]

print(df["conversation"][0])