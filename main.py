import data_transformation as dt
import llm_setup as llm
import llm_classification_prompts as prompts
import json

df = dt.get_conversations()

batch_size = 10

for i in range(0, len(df), batch_size):
    print(f"Processing batch {i//batch_size + 1} of {len(df)//batch_size + 1}...")
    
    batch = df.iloc[i:i+batch_size]
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
    except Exception as e:
        print(f"Error processing batch {i//batch_size + 1}: {e}")

df.to_csv("classified_conversations.csv", index=False)
print("Classification complete. Saved to classified_conversations.csv")