"""
build_golden_dataset.py — interactive CLI for building a golden dataset

You read each conversation and either confirm or correct what the pipeline extracted.
Your verified labels become ground truth for calibrating the LLM judge.

Usage:
    python build_golden_dataset.py                  # start or resume labeling
    python build_golden_dataset.py --sample 30      # how many conversations to label
    python build_golden_dataset.py --show-stats     # show progress on existing golden set

Output:
    golden_dataset.csv — your verified ground truth, appended on each run
"""

import argparse
import os
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

CLASSIFIED_PATH = "classified_conversations.csv"
GOLDEN_PATH = "golden_dataset.csv"

ENUM_FIELDS = {
    "engagement_level": ["hot", "warm", "cold"],
    "journey_stage": [
        "awareness", "consideration", "objection",
        "negotiation", "close", "lost"
    ],
}
TEXT_FIELDS = ["intent", "objections", "product_category"]
ALL_FIELDS = TEXT_FIELDS + list(ENUM_FIELDS.keys())

# ── Display helpers ───────────────────────────────────────────────────────────

def clear():
    os.system("cls" if os.name == "nt" else "clear")


def print_header(current: int, total: int):
    print("=" * 70)
    print(f"  GOLDEN DATASET BUILDER   [{current}/{total}]")
    print("=" * 70)


def print_conversation(text: str):
    print("\n── CONVERSATION " + "─" * 54)
    # Wrap long lines for readability
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Simple word-wrap at 68 chars
        while len(line) > 68:
            split_at = line[:68].rfind(" ")
            if split_at == -1:
                split_at = 68
            print("  " + line[:split_at])
            line = line[split_at:].strip()
        print("  " + line)
    print("─" * 70)


def prompt_field(field: str, pipeline_value: str, valid_values: list = None) -> str:
    """
    Shows the pipeline's extracted value and asks for confirmation or correction.
    Returns the accepted or corrected value.
    """
    print(f"\n  Field: {field}")
    print(f"  Pipeline extracted: \"{pipeline_value}\"")

    if valid_values:
        print(f"  Valid values: {', '.join(valid_values)}")

    while True:
        raw = input("  Accept? [Enter to accept / type correction]: ").strip()

        if raw == "":
            # Accept pipeline value
            return pipeline_value

        if valid_values and raw.lower() not in valid_values:
            print(f"  ✗ Must be one of: {', '.join(valid_values)}")
            continue

        return raw.lower() if valid_values else raw


# ── Core labeling logic ───────────────────────────────────────────────────────

def label_conversation(row: pd.Series, current: int, total: int) -> dict:
    """
    Walks through a single conversation interactively.
    Returns a dict of verified field values.
    """
    clear()
    print_header(current, total)
    print_conversation(str(row["conversation"]))

    print("\n── LABEL EACH FIELD " + "─" * 50)
    print("  Press Enter to accept the pipeline's value, or type a correction.\n")

    verified = {"conversation": row["conversation"]}

    for field in TEXT_FIELDS:
        verified[field] = prompt_field(field, str(row.get(field, "")))

    for field, valid in ENUM_FIELDS.items():
        verified[field] = prompt_field(field, str(row.get(field, "")), valid)

    return verified


def save_label(label: dict, path: str):
    """Appends one verified row to the golden dataset CSV."""
    row_df = pd.DataFrame([label])
    write_header = not os.path.exists(path)
    row_df.to_csv(path, mode="a", header=write_header, index=False)


# ── Stats view ────────────────────────────────────────────────────────────────

def show_stats():
    if not os.path.exists(GOLDEN_PATH):
        print(f"No golden dataset found at {GOLDEN_PATH}. Run labeling first.")
        return

    df = pd.read_csv(GOLDEN_PATH)
    print(f"\n Golden dataset: {len(df)} labeled conversations\n")

    for field, valid in ENUM_FIELDS.items():
        print(f"  {field}:")
        counts = df[field].value_counts()
        for v in valid:
            print(f"    {v:15s}  {counts.get(v, 0)}")
        print()

    print("  Text fields (non-empty):")
    for f in TEXT_FIELDS:
        n_filled = df[f].notna().sum() if f in df else 0
        print(f"    {f:20s}  {n_filled} / {len(df)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=25,
                        help="How many conversations to label (default: 25)")
    parser.add_argument("--show-stats", action="store_true",
                        help="Show stats on the existing golden dataset and exit")
    args = parser.parse_args()

    if args.show_stats:
        show_stats()
        return

    # Load classified data
    df = pd.read_csv(CLASSIFIED_PATH)

    # Skip conversations already in the golden dataset
    already_labeled = set()
    if os.path.exists(GOLDEN_PATH):
        golden = pd.read_csv(GOLDEN_PATH)
        already_labeled = set(golden["conversation"].tolist())
        print(f"Resuming: {len(already_labeled)} conversations already labeled.")

    remaining = df[~df["conversation"].isin(already_labeled)]

    if len(remaining) == 0:
        print("All sampled conversations are already labeled!")
        return

    # Sample from what's left — use random_state for reproducibility
    to_label = remaining.sample(min(args.sample, len(remaining)), random_state=99)

    print(f"\nLabeling {len(to_label)} conversations. Press Ctrl+C to stop and save progress.\n")
    input("Press Enter to begin...")

    labeled_this_session = 0

    try:
        for i, (_, row) in enumerate(to_label.iterrows(), start=1):
            label = label_conversation(row, i, len(to_label))
            save_label(label, GOLDEN_PATH)
            labeled_this_session += 1

            print(f"\n  ✓ Saved. ({labeled_this_session} labeled this session)")
            if i < len(to_label):
                input("  Press Enter for next conversation...")

    except KeyboardInterrupt:
        print(f"\n\nStopped. {labeled_this_session} conversations labeled this session.")

    total = len(pd.read_csv(GOLDEN_PATH)) if os.path.exists(GOLDEN_PATH) else 0
    print(f"Golden dataset now has {total} verified conversations → {GOLDEN_PATH}")
    print("Run `python run_evals.py` to calibrate the judge against your ground truth.")


if __name__ == "__main__":
    main()
