"""
run_evals.py — evaluate the quality of your pipeline output

Run after run_pipeline.py has produced classified_conversations.csv.

Usage:
    python run_evals.py

If golden_dataset.csv exists, also runs judge calibration: measures how often
the LLM judge agrees with your human-verified labels. That tells you whether
the judge itself is trustworthy before you use it at scale.
"""

import os
import pandas as pd

from pipeline.evals import (
    LLMJudge,
    eval_summary,
    schema_summary,
    validate_schema,
)
from pipeline.llm import GroqClient

CLASSIFIED_PATH = "classified_conversations.csv"
GOLDEN_PATH = "golden_dataset.csv"

# Fields that require LLM judgment (qualitative, free-text)
QUALITATIVE_FIELDS = ["intent", "objections", "product_category"]


def run_judge_calibration(llm, golden: pd.DataFrame):
    """
    Runs the LLM judge against your human-verified golden dataset and
    measures agreement. This tells you whether the judge can be trusted.

    Agreement is measured per field:
      - The judge scores the pipeline's extraction (1-3)
      - We then check: does the judge's pass/fail match the human label?
        (A human label that differs from the pipeline extraction = fail)

    A simple proxy: if the human accepted the pipeline's value (they match),
    the judge should score >= 2. If the human corrected it, the judge should
    score 1. We measure how often the judge gets this right.
    """
    print(f"\nRunning judge calibration on {len(golden)} golden conversations...")
    print("This measures how often the judge agrees with your human labels.\n")

    judge = LLMJudge(llm=llm, fields=QUALITATIVE_FIELDS)

    # Re-load the full classified data to get the pipeline's original extractions
    if not os.path.exists(CLASSIFIED_PATH):
        print("  classified_conversations.csv not found — skipping calibration.")
        return

    classified = pd.read_csv(CLASSIFIED_PATH)

    # Match golden rows to the pipeline's output by conversation text
    merged = golden.merge(
        classified[["conversation"] + QUALITATIVE_FIELDS],
        on="conversation",
        suffixes=("_human", "_pipeline"),
        how="inner",
    )

    if len(merged) == 0:
        print("  No matching rows between golden dataset and classified output.")
        print("  Run run_pipeline.py first to generate classified_conversations.csv.")
        return

    print(f"  Matched {len(merged)} golden rows to pipeline output.")

    # Build a DataFrame that looks like classified output so the judge can run
    # We judge the pipeline's extraction (not the human's) — that's the point
    judge_input = pd.DataFrame({
        "conversation": merged["conversation"],
        "intent":            merged.get("intent_pipeline",            merged.get("intent")),
        "objections":        merged.get("objections_pipeline",        merged.get("objections")),
        "product_category":  merged.get("product_category_pipeline",  merged.get("product_category")),
    })

    judge_df = judge.run(judge_input, sample_size=len(judge_input))

    if judge_df.empty:
        print("  Judge returned no results.")
        return

    # Compute human agreement signal:
    # For each (row, field), was the pipeline's extraction what the human verified?
    agreement_records = []
    for _, jrow in judge_df.iterrows():
        conv = judge_input.iloc[jrow["row_idx"]]["conversation"] if isinstance(jrow["row_idx"], int) else None
        if conv is None:
            continue

        human_match = merged[merged["conversation"] == conv]
        if human_match.empty:
            continue

        human_val = human_match.iloc[0].get(f"{jrow['field']}_human", None)
        pipeline_val = jrow["extracted_value"]

        # Human accepted pipeline value = extraction was correct = judge should pass
        human_approved = (str(human_val).strip().lower() == str(pipeline_val).strip().lower())
        judge_agreed = jrow["passed"]  # judge scored >= 2

        agreement_records.append({
            "field":          jrow["field"],
            "human_approved": human_approved,
            "judge_passed":   judge_agreed,
            "agreed":         human_approved == judge_agreed,
            "score":          jrow["score"],
            "reasoning":      jrow["reasoning"],
        })

    if not agreement_records:
        print("  Could not compute agreement — check that golden and classified data match.")
        return

    agreement_df = pd.DataFrame(agreement_records)
    overall_agreement = agreement_df["agreed"].mean()

    print("\n=== Judge calibration ===")
    by_field = (
        agreement_df.groupby("field")
        .agg(agreement_rate=("agreed", "mean"), n=("agreed", "count"))
        .round(3)
    )
    print(by_field.to_string())
    print(f"\nOverall judge-human agreement: {overall_agreement:.1%}")

    if overall_agreement >= 0.80:
        print("✓ Judge is well-calibrated (≥80% agreement). Safe to use at scale.")
    elif overall_agreement >= 0.65:
        print("⚠ Judge is moderately calibrated. Review low-agreement fields above.")
    else:
        print("✗ Judge agreement is low (<65%). Revisit the judge rubric or prompt.")

    agreement_df.to_csv("judge_calibration.csv", index=False)
    print("Detailed calibration saved to judge_calibration.csv")


def main():
    df = pd.read_csv(CLASSIFIED_PATH)
    print(f"Loaded {len(df)} rows from {CLASSIFIED_PATH}\n")

    # ── Track 1: programmatic checks ─────────────────────────────────────────
    print("Running schema validation...")
    check_df = validate_schema(df)
    summary = schema_summary(check_df)
    print("\n=== Schema validation ===")
    print(summary.to_string(index=False))

    failed_rows = check_df[~check_df["passed"]]["row_idx"].unique()
    print(f"\nRows with at least one schema failure: {len(failed_rows)} / {len(df)}")

    # ── Track 2: LLM-as-judge ────────────────────────────────────────────────
    print("\nRunning LLM-as-judge on qualitative fields (sample of 30 rows)...")
    llm = GroqClient()
    judge = LLMJudge(llm=llm, fields=QUALITATIVE_FIELDS)
    judge_df = judge.run(df, sample_size=30)

    eval_summary(judge_df)

    judge_df.to_csv("eval_results.csv", index=False)
    print("\nDetailed results saved to eval_results.csv")

    # ── Track 3: judge calibration (only if golden dataset exists) ───────────
    if os.path.exists(GOLDEN_PATH):
        golden = pd.read_csv(GOLDEN_PATH)
        print(f"\nFound golden dataset ({len(golden)} rows) — running calibration...")
        run_judge_calibration(llm, golden)
    else:
        print(
            f"\nNo golden dataset found at {GOLDEN_PATH}. "
            "Run `python build_golden_dataset.py` to build one."
        )


if __name__ == "__main__":
    main()
