import json
import random
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from pipeline.llm import LLMClient


# ── Programmatic validator ────────────────────────────────────────────────────

ENUM_CONSTRAINTS: Dict[str, set] = {
    "engagement_level": {"hot", "warm", "cold"},
    "journey_stage": {
        "awareness", "consideration", "objection",
        "negotiation", "close", "lost",
    },
}

TEXT_FIELDS = {"intent", "objections", "product_category"}


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs programmatic checks against every row.
    Returns a DataFrame of check results — one row per (field, check).
    No LLM calls. Run this on every batch before anything else.
    """
    records = []

    for idx, row in df.iterrows():
        for col, allowed in ENUM_CONSTRAINTS.items():
            value = row.get(col)
            records.append({
                "row_idx": idx,
                "field": col,
                "check": "enum_compliance",
                "value": value,
                "passed": str(value).strip().lower() in allowed if value else False,
            })

        for col in TEXT_FIELDS:
            value = row.get(col)
            records.append({
                "row_idx": idx,
                "field": col,
                "check": "non_empty",
                "value": value,
                "passed": bool(value and str(value).strip()),
            })

    return pd.DataFrame(records)


def schema_summary(check_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates validate_schema output into a per-field pass-rate table."""
    return (
        check_df.groupby(["field", "check"])
        .agg(pass_rate=("passed", "mean"), n=("passed", "count"))
        .round(3)
        .reset_index()
    )


# ── LLM-as-judge ─────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """You are an expert evaluator assessing the quality of information \
extracted from sales conversations.
Evaluate the extracted value against the source conversation using the rubric provided.
Always return valid JSON only. No explanations, no preamble, no markdown."""

_JUDGE_PROMPT = """Evaluate the extracted field from this sales conversation.

CONVERSATION:
{conversation}

FIELD: {field}
EXTRACTED VALUE: {extracted_value}

Score using this rubric:
- 3: Accurate and complete — directly supported by the conversation, captures the core meaning
- 2: Mostly accurate — right direction but missing nuance, slightly vague, or partially supported
- 1: Inaccurate or unsupported — wrong, fabricated, or not grounded in the conversation

Return only:
{{"score": 1|2|3, "reasoning": "one sentence explaining the score"}}"""


@dataclass
class JudgeResult:
    row_idx: int
    field: str
    extracted_value: str
    score: int          # 1, 2, or 3
    score_norm: float   # 0.0 to 1.0
    reasoning: str
    passed: bool        # True if score >= 2


class LLMJudge:
    """
    Evaluates qualitative text fields using a second LLM call.
    Use for: intent, objections, product_category.
    Not needed for enum fields — validate_schema handles those.

    Usage:
        judge = LLMJudge(llm=GroqClient(), fields=["intent", "objections"])
        results_df = judge.run(df, sample_size=30)
    """

    def __init__(self, llm: LLMClient, fields: List[str], pass_threshold: int = 2):
        self.llm = llm
        self.fields = fields
        self.pass_threshold = pass_threshold

    def _judge_field(
        self,
        row_idx: int,
        conversation: str,
        field: str,
        extracted_value: str,
    ) -> Optional[JudgeResult]:
        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {
                "role": "user",
                "content": _JUDGE_PROMPT.format(
                    conversation=conversation,
                    field=field,
                    extracted_value=extracted_value,
                ),
            },
        ]

        try:
            response = self.llm.get_response(messages)
            start = response.index("{")
            end = response.rindex("}") + 1
            parsed = json.loads(response[start:end])

            score = int(parsed["score"])
            return JudgeResult(
                row_idx=row_idx,
                field=field,
                extracted_value=extracted_value,
                score=score,
                score_norm=round((score - 1) / 2, 3),   # map 1-3 → 0.0-1.0
                reasoning=parsed["reasoning"],
                passed=score >= self.pass_threshold,
            )

        except Exception as e:
            print(f"  Judge error (row {row_idx}, field '{field}'): {e}")
            return None

    def run(self, df: pd.DataFrame, sample_size: int = 30) -> pd.DataFrame:
        """
        Evaluates a random sample of rows across all configured fields.
        Returns a DataFrame with one row per (row, field) judgment.
        """
        sample = df.sample(min(sample_size, len(df)), random_state=42)
        results: List[JudgeResult] = []

        total = len(sample) * len(self.fields)
        done = 0

        for idx, row in sample.iterrows():
            conversation = row.get("conversation", "")
            for f in self.fields:
                value = row.get(f)
                if not value or not str(value).strip():
                    done += 1
                    continue

                print(f"  Judging [{done + 1}/{total}] row {idx} — {f}...")
                result = self._judge_field(idx, conversation, f, str(value))
                if result:
                    results.append(result)
                done += 1

        return pd.DataFrame([r.__dict__ for r in results])


# ── Consistency checker ───────────────────────────────────────────────────────

def check_consistency(
    pipeline_fn,
    conversations: List[str],
    field: str,
    runs: int = 3,
) -> pd.DataFrame:
    """
    Runs the same conversations through the pipeline N times and checks
    whether the same field value is produced each time.

    Useful for catching high-variance prompts before they hit production.

    Args:
        pipeline_fn: callable that takes a list of conversations and returns
                     a list of dicts (one per conversation).
        conversations: list of raw conversation strings to test.
        field: which extracted field to check for consistency.
        runs: how many times to run each conversation.

    Example:
        results = check_consistency(
            pipeline_fn=my_extract_fn,
            conversations=sample_convos,
            field="engagement_level",
            runs=3,
        )
    """
    records = []

    for i, convo in enumerate(conversations):
        outputs = []
        for run in range(runs):
            result = pipeline_fn([convo])
            value = result[0].get(field) if result else None
            outputs.append(value)

        unique_values = set(v for v in outputs if v is not None)
        records.append({
            "conversation_idx": i,
            "field": field,
            "outputs": outputs,
            "consistent": len(unique_values) == 1,
            "unique_count": len(unique_values),
        })

    return pd.DataFrame(records)


# ── Summary reporting ─────────────────────────────────────────────────────────

def eval_summary(judge_df: pd.DataFrame) -> None:
    """Prints a field-level summary from LLMJudge.run() output."""
    print("\n=== LLM-as-judge results ===")
    by_field = (
        judge_df.groupby("field")
        .agg(
            avg_score=("score", "mean"),
            pass_rate=("passed", "mean"),
            n=("field", "count"),
        )
        .round(3)
    )
    print(by_field.to_string())
    print(f"\nOverall pass rate : {judge_df['passed'].mean():.1%}")
    print(f"Overall avg score : {judge_df['score'].mean():.2f} / 3")

    print("\n--- Low-scoring examples ---")
    low = judge_df[~judge_df["passed"]].head(5)
    for _, row in low.iterrows():
        print(f"  [{row['field']}] score={row['score']} | {row['extracted_value'][:80]}")
        print(f"    reason: {row['reasoning']}")
