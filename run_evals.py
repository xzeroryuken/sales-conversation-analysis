"""
run_evals.py — evaluate the quality of your pipeline output

Run after run_pipeline.py has produced classified_conversations.csv.

Usage:
    python run_evals.py
"""

import pandas as pd

from pipeline.evals import (
    LLMJudge,
    eval_summary,
    schema_summary,
    validate_schema,
)
from pipeline.llm import GroqClient

CLASSIFIED_PATH = "classified_conversations.csv"

# Fields that require LLM judgment (qualitative, free-text)
QUALITATIVE_FIELDS = ["intent", "objections", "product_category"]

# Fields covered by programmatic checks (enums + non-empty)
# → handled automatically by validate_schema


def main():
    df = pd.read_csv(CLASSIFIED_PATH)
    print(f"Loaded {len(df)} rows from {CLASSIFIED_PATH}\n")

    # ── Track 1: programmatic checks ─────────────────────────────────────────
    print("Running schema validation...")
    check_df = validate_schema(df)
    summary = schema_summary(check_df)
    print("\n=== Schema validation ===")
    print(summary.to_string(index=False))

    # Flag rows with any failed check
    failed_rows = check_df[~check_df["passed"]]["row_idx"].unique()
    print(f"\nRows with at least one schema failure: {len(failed_rows)} / {len(df)}")

    # ── Track 2: LLM-as-judge ─────────────────────────────────────────────────
    print("\nRunning LLM-as-judge on qualitative fields (sample of 30 rows)...")
    llm = GroqClient()
    judge = LLMJudge(llm=llm, fields=QUALITATIVE_FIELDS)
    judge_df = judge.run(df, sample_size=30)

    eval_summary(judge_df)

    # Save results for inspection
    judge_df.to_csv("eval_results.csv", index=False)
    print("\nDetailed results saved to eval_results.csv")


if __name__ == "__main__":
    main()
