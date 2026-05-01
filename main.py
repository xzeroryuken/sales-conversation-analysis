import data_transformation as dt
import llm_setup as llm
import llm_classification_prompts as prompts
import json
import pandas as pd
import os

if os.path.exists("classified_conversations.csv"):
    df = pd.read_csv("classified_conversations.csv")
else:
    df = dt.get_conversations()
    df["intent"] = None
    df["objections"] = None
    df["product_category"] = None
    df["engagement_level"] = None

batch_size = 10

for i in range(0, len(df), batch_size):
    print(f"Processing batch {i//batch_size + 1} of {len(df)//batch_size + 1}...")
    
    batch = df.iloc[i:i+batch_size]

    if batch["intent"].notna().any():
        continue

    batch_conversations = batch["conversation"].tolist()
    user_message = "\n\n".join([f"conversation_{j+1}: {conv}" for j, conv in enumerate(batch_conversations)])

    messages=[
    {"role": "system", "content": prompts.system_prompt + "\n" + prompts.classification_prompt},
    {"role": "user", "content": user_message}
    ]
    
    try:
        response = llm.get_response(messages)

        start = response.index("[")
        end = response.rindex("]") + 1
        clean_response = response[start:end]

        parsed = json.loads(clean_response)

        for idx, classification in zip(batch.index, parsed):
            df.at[idx, "intent"] = classification["intent"]
            df.at[idx, "objections"] = classification["objections"]
            df.at[idx, "product_category"] = classification["product_category"]
            df.at[idx, "engagement_level"] = classification["engagement_level"]
        
        df.to_csv("classified_conversations.csv", index=False)
    except Exception as e:
        print(f"Error processing batch {i//batch_size + 1}: {e}")

print("Classification complete. Saved to classified_conversations.csv")