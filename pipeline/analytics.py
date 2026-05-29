import json
import pandas as pd
from typing import Optional

from pipeline.llm import LLMClient


JOURNEY_ORDER = ["awareness", "consideration", "objection", "negotiation", "close", "lost"]

_INSIGHTS_SYSTEM = """You are a senior sales analyst. You will be given structured analytics
from a customer conversation dataset. Your job is to identify the 5 most actionable findings
and write them as a concise markdown report. Focus on patterns that have clear implications
for sales coaching, product positioning, or process improvement. Be specific and direct."""

_INSIGHTS_PROMPT = """Here is a structured analytics summary from {n_conversations} customer conversations.

JOURNEY STAGE FUNNEL:
{funnel}

ENGAGEMENT DISTRIBUTION:
{engagement}

ENGAGEMENT BY JOURNEY STAGE:
{engagement_by_stage}

TOP OBJECTION CLUSTERS:
{top_clusters}

OBJECTION CLUSTERS BY ENGAGEMENT:
{clusters_by_engagement}

HIGH-VALUE RECOVERABLE SEGMENT (warm/hot + stuck at objection or consideration):
{recoverable}

Write a markdown report with exactly 5 findings. Each finding must include:
- A bold headline (what you observed)
- One sentence explaining why it matters
- One concrete recommended action

Order by business impact, highest first."""


def compute(df: pd.DataFrame, taxonomy: Optional[pd.DataFrame] = None) -> dict:
    """
    Computes aggregate analytics from the classified conversation DataFrame.
    Returns a structured dict suitable for JSON serialization or LLM input.
    """
    df = df[df["intent"].notna()].copy()
    n = len(df)

    # Normalize
    df["engagement_level"] = df["engagement_level"].str.strip().str.lower()
    df["journey_stage"] = df["journey_stage"].str.strip().str.lower()

    # ── Funnel ────────────────────────────────────────────────────────────────
    stage_counts = df["journey_stage"].value_counts()
    funnel = {
        stage: {
            "count": int(stage_counts.get(stage, 0)),
            "pct": round(stage_counts.get(stage, 0) / n * 100, 1),
        }
        for stage in JOURNEY_ORDER
    }

    # ── Engagement distribution ───────────────────────────────────────────────
    eng_counts = df["engagement_level"].value_counts()
    engagement = {
        level: {
            "count": int(eng_counts.get(level, 0)),
            "pct": round(eng_counts.get(level, 0) / n * 100, 1),
        }
        for level in ["hot", "warm", "cold"]
    }

    # ── Engagement × journey stage ────────────────────────────────────────────
    crosstab = pd.crosstab(
        df["journey_stage"],
        df["engagement_level"],
        normalize="index",
    ).round(3)
    engagement_by_stage = {
        stage: crosstab.loc[stage].to_dict()
        for stage in JOURNEY_ORDER
        if stage in crosstab.index
    }

    # ── Objection clusters ────────────────────────────────────────────────────
    cluster_col = "objection_cluster"
    top_clusters = {}
    clusters_by_engagement = {}

    if cluster_col in df.columns:
        cluster_df = df[df[cluster_col].notna() & (df[cluster_col] != -1)].copy()
        cluster_df[cluster_col] = cluster_df[cluster_col].astype(int)

        # Build label map from taxonomy if provided
        label_map = {}
        if taxonomy is not None and not taxonomy.empty:
            label_map = dict(zip(taxonomy["cluster_id"].astype(int), taxonomy["label"]))

        counts = cluster_df[cluster_col].value_counts()
        for cid, count in counts.head(8).items():
            label = label_map.get(int(cid), f"Cluster {cid}")
            top_clusters[label] = {
                "count": int(count),
                "pct": round(count / n * 100, 1),
            }

        # Cluster × engagement
        eng_cross = pd.crosstab(
            cluster_df[cluster_col],
            cluster_df["engagement_level"],
            normalize="index",
        ).round(3)
        for cid in eng_cross.index:
            label = label_map.get(int(cid), f"Cluster {cid}")
            clusters_by_engagement[label] = eng_cross.loc[cid].to_dict()

    # ── Recoverable segment ───────────────────────────────────────────────────
    recoverable_mask = (
        df["engagement_level"].isin(["hot", "warm"]) &
        df["journey_stage"].isin(["objection", "consideration"])
    )
    recoverable_df = df[recoverable_mask]
    recoverable = {
        "count": int(len(recoverable_df)),
        "pct_of_total": round(len(recoverable_df) / n * 100, 1),
        "engagement_split": recoverable_df["engagement_level"].value_counts().to_dict(),
        "top_objection_clusters": (
            recoverable_df[cluster_col]
            .value_counts()
            .head(3)
            .to_dict()
            if cluster_col in recoverable_df.columns else {}
        ),
    }

    return {
        "n_conversations": n,
        "funnel": funnel,
        "engagement": engagement,
        "engagement_by_stage": engagement_by_stage,
        "top_clusters": top_clusters,
        "clusters_by_engagement": clusters_by_engagement,
        "recoverable_segment": recoverable,
    }


def generate_insights(analytics: dict, llm: LLMClient) -> str:
    """
    Sends the analytics summary to an LLM and returns a markdown insights report.
    """
    def fmt(d): return json.dumps(d, indent=2)

    prompt = _INSIGHTS_PROMPT.format(
        n_conversations=analytics["n_conversations"],
        funnel=fmt(analytics["funnel"]),
        engagement=fmt(analytics["engagement"]),
        engagement_by_stage=fmt(analytics["engagement_by_stage"]),
        top_clusters=fmt(analytics["top_clusters"]),
        clusters_by_engagement=fmt(analytics["clusters_by_engagement"]),
        recoverable=fmt(analytics["recoverable_segment"]),
    )

    messages = [
        {"role": "system", "content": _INSIGHTS_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    return llm.get_response(messages)
