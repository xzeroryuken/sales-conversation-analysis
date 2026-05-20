import json
import os
import pandas as pd

from configs.base import DomainConfig
from pipeline.llm import LLMClient, BatchTooLargeError
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

    def _build_messages(self, conversations: list) -> list:
        user_message = "\n\n".join(
            f"conversation_{j + 1}: {conv}" for j, conv in enumerate(conversations)
        )
        return [
            {
                "role": "system",
                "content": self.config.system_prompt + "\n" + self.config.classification_prompt,
            },
            {"role": "user", "content": user_message},
        ]

    def _process_batch(self, df: pd.DataFrame, indices: list) -> None:
        """Process a batch of rows by index. Recursively halves on BatchTooLargeError."""
        if not indices:
            return

        conversations = df.loc[indices, "conversation"].tolist()
        messages = self._build_messages(conversations)

        try:
            response = self.llm.get_response(messages)
            start = response.index("[")
            end = response.rindex("]") + 1
            parsed = json.loads(response[start:end])

            for idx, classification in zip(indices, parsed):
                for field in self.config.fields:
                    df.at[idx, field] = classification.get(field)
                if hasattr(self.llm, "last_model_used") and self.llm.last_model_used:
                    df.at[idx, "model_used"] = self.llm.last_model_used

            df.to_csv(self.output_path, index=False)

        except BatchTooLargeError:
            if len(indices) == 1:
                print(f"  Row {indices[0]} is too large for all models, skipping.")
                return
            mid = len(indices) // 2
            print(f"  Batch too large, halving to {mid} conversations...")
            self._process_batch(df, indices[:mid])
            self._process_batch(df, indices[mid:])

        except Exception as e:
            print(f"  Error: {e}")

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

            self._process_batch(df, list(batch.index))

        print(f"Classification complete. Saved to {self.output_path}")
        return df
