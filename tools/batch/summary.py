"""Win rates of a batch run, per opponent, per map and overall. Given two runs, the difference.

    python -m tools.batch.summary runs/<run>
    python -m tools.batch.summary runs/<before> runs/<after>
"""

import argparse
import json
from collections import Counter
from pathlib import Path

OUTCOMES = ("Victory", "Defeat", "Tie", "Crash", "Timeout")


def load_results(run_dir: str | Path) -> list[dict]:
    path = Path(run_dir) / "results.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def aggregate(records: list[dict], key: str | None) -> dict[str, Counter]:
    """Outcome counts per value of `key` ("opponent", "map"), or one "all" group for None."""
    groups: dict[str, Counter] = {}
    for record in records:
        label = record[key] if key else "all"
        groups.setdefault(label, Counter())[record["result"]] += 1
    return dict(sorted(groups.items()))


def win_rate(counts: Counter) -> float:
    total = sum(counts.values())
    return counts["Victory"] / total if total else 0.0


def format_table(title: str, groups: dict[str, Counter]) -> str:
    width = max([len(title)] + [len(label) for label in groups])
    lines = [f"{title:<{width}}  games  win%  " + "  ".join(f"{o[:5]:>5}" for o in OUTCOMES)]
    for label, counts in groups.items():
        cells = "  ".join(f"{counts[o]:>5}" for o in OUTCOMES)
        lines.append(f"{label:<{width}}  {sum(counts.values()):>5}  {win_rate(counts):>4.0%}  {cells}")
    return "\n".join(lines)


def format_comparison(title: str, before: dict[str, Counter], after: dict[str, Counter]) -> str:
    labels = sorted(set(before) | set(after))
    width = max([len(title)] + [len(label) for label in labels])
    lines = [f"{title:<{width}}  before  after  change"]
    for label in labels:
        if label not in before or label not in after:
            lines.append(f"{label:<{width}}  (only in one run)")
            continue
        a, b = win_rate(before[label]), win_rate(after[label])
        lines.append(f"{label:<{width}}  {a:>6.0%}  {b:>5.0%}  {(b - a) * 100:>+5.0f}pp")
    return "\n".join(lines)


def summarize(run_dir: str | Path) -> str:
    records = load_results(run_dir)
    return "\n\n".join(
        format_table(title, aggregate(records, key))
        for title, key in (("opponent", "opponent"), ("map", "map"), ("overall", None))
    )


def compare(before_dir: str | Path, after_dir: str | Path) -> str:
    before, after = load_results(before_dir), load_results(after_dir)
    return "\n\n".join(
        format_comparison(title, aggregate(before, key), aggregate(after, key))
        for title, key in (("opponent", "opponent"), ("map", "map"), ("overall", None))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="one run folder, or two to compare")
    args = parser.parse_args()
    if len(args.runs) == 1:
        print(summarize(args.runs[0]))
    elif len(args.runs) == 2:
        print(compare(*args.runs))
    else:
        parser.error("give one run folder, or two to compare")


if __name__ == "__main__":
    main()
