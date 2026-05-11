import json
import numpy as np
import pandas as pd
from datetime import date

from configs.base import DomainConfig
from pipeline.llm import LLMClient


_SYSTEM_PROMPT = """You are an expert at identifying patterns and themes in customer data.
Your task is to analyze a sample of customer objections from a single cluster and generate a concise label and description.
Always return valid JSON only. No explanations, no preamble, no markdown."""

_LABELING_PROMPT = """The following customer objections were grouped together by a clustering algorithm because they share a common theme.

Analyze the objections and return a JSON object with:
- label: a concise 2-5 word phrase naming the theme (title case)
- description: one sentence explaining the common pattern across these objections

OBJECTIONS:
{objections}

Return only:
{{"label": "...", "description": "..."}}"""


class LabelingPipeline:
    def __init__(self, config: DomainConfig, llm: LLMClient, sample_size: int = 20):
        self.config = config
        self.llm = llm
        self.sample_size = sample_size

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        field = self.config.cluster_on
        cluster_col = f"{field}_cluster"

        if cluster_col not in df.columns:
            raise ValueError(f"Column '{cluster_col}' not found. Run ClusteringPipeline first.")

        cluster_ids = sorted(
            int(c) for c in df[cluster_col].dropna().unique() if int(c) != -1
        )

        records = []

        for cluster_id in cluster_ids:
            cluster_df = df[df[cluster_col] == cluster_id]
            unique_texts = cluster_df[field].dropna().unique()
            sample = np.random.choice(
                unique_texts,
                size=min(self.sample_size, len(unique_texts)),
                replace=False,
            )

            print(f"Labeling cluster {cluster_id} ({len(cluster_df)} conversations)...")

            objections_block = "\n".join(f"- {t}" for t in sample)
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _LABELING_PROMPT.format(objections=objections_block)},
            ]

            try:
                response = self.llm.get_response(messages)
                start = response.index("{")
                end = response.rindex("}") + 1
                parsed = json.loads(response[start:end])

                records.append({
                    "cluster_id": cluster_id,
                    "label": parsed["label"],
                    "description": parsed["description"],
                    "size": len(cluster_df),
                    "domain": self.config.domain,
                    "cluster_field": field,
                    "version": str(date.today()),
                })
            except Exception as e:
                print(f"Error labeling cluster {cluster_id}: {e}")

        return pd.DataFrame(records)
