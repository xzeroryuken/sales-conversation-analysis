import json
import os
import pandas as pd

from pipeline.analytics import compute, generate_insights
from pipeline.llm import GroqClient

CLASSIFIED_PATH = "classified_conversations.csv"
TAXONOMY_PATH = "cluster_taxonomy.csv"
ANALYTICS_OUTPUT = "analytics_summary.json"
INSIGHTS_OUTPUT = "insights_report.md"


df = pd.read_csv(CLASSIFIED_PATH)
taxonomy = pd.read_csv(TAXONOMY_PATH) if os.path.exists(TAXONOMY_PATH) else None

print(f"Computing analytics on {df['intent'].notna().sum()} classified conversations...")
analytics = compute(df, taxonomy)

with open(ANALYTICS_OUTPUT, "w") as f:
    json.dump(analytics, f, indent=2)
print(f"Analytics saved to {ANALYTICS_OUTPUT}")

print("Generating insights report...")
llm = GroqClient()
report = generate_insights(analytics, llm)

with open(INSIGHTS_OUTPUT, "w", encoding="utf-8") as f:
    f.write(report)
print(f"Insights saved to {INSIGHTS_OUTPUT}\n")
print(report)
