import json
import os
import pandas as pd

from configs.base import DomainConfig
from pipeline.llm import LLMClient
from pipeline.ingestion import DataSource


class ConversationPipeline:
    def __init__(
        self,
        config: DomainConfig,
        llm: LLMClient,
        source: DataSource,
        output_path: str,
        batch_size: int = 10,
    ):
        self.config = config
        self.llm = llm
        self.source = source
        self.output_path = output_path
        self.batch_size = batch_size

    def run(self) -> pd.DataFrame:
        if os.path.exists(self.output_path):
            df = pd.read_csv(self.output_path)
        else:
            df = self.source.load()
            for field in self.config.fields:
                df[field] = None

        total_batches = (len(df) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(df), self.batch_size):
            batch_num = i // self.batch_size + 1
            print(f"Processing batch {batch_num} of {total_batches}...")

            batch = df.iloc[i : i + self.batch_size]

            if batch[self.config.fields[0]].notna().any():
                continue

            conversations = batch["conversation"].tolist()
            user_message = "\n\n".join(
                f"conversation_{j + 1}: {conv}" for j, conv in enumerate(conversations)
            )

            messages = [
                {
                    "role": "system",
                    "content": self.config.system_prompt + "\n" + self.config.classification_prompt,
                },
                {"role": "user", "content": user_message},
            ]

            try:
                response = self.llm.get_response(messages)
                start = response.index("[")
                end = response.rindex("]") + 1
                parsed = json.loads(response[start:end])

                for idx, classification in zip(batch.index, parsed):
                    for field in self.config.fields:
                        df.at[idx, field] = classification.get(field)

                df.to_csv(self.output_path, index=False)
            except Exception as e:
                print(f"Error on batch {batch_num}: {e}")

        print(f"Classification complete. Saved to {self.output_path}")
        return df
