"""Whether a training run improved and stayed improved, from its training win rate.

    python -m tools.rl.stability models/sweep-stable-1 models/sweep-stable-2 ...

Reads each run's logs/progress.csv (fights/all/win_rate: the last 100 fights, all scenarios) and
compares three windows: the start (first 100k decisions), the window where an earlier zergling
run peaked before collapsing (250k to 500k), and the end (last 500k). A run holds if its end is
at least 10 points above its start and at most 5 points below that peak window.
"""

import argparse
import csv
from pathlib import Path

START_END = 100_000
PEAK_WINDOW = (250_000, 500_000)
END_LENGTH = 500_000
MIN_GAIN = 0.10
MAX_DROP = 0.05

STEPS_COLUMN = "time/total_timesteps"
WIN_RATE_COLUMN = "fights/all/win_rate"


def read_win_rates(progress_csv: Path) -> list[tuple[int, float]]:
    """(decisions so far, training win rate) per logged update, skipping updates before the
    first finished fight."""
    with open(progress_csv, encoding="utf-8") as f:
        return [(int(float(row[STEPS_COLUMN])), float(row[WIN_RATE_COLUMN]))
                for row in csv.DictReader(f) if row.get(WIN_RATE_COLUMN)]


def windows(win_rates: list[tuple[int, float]]) -> tuple[float, float, float]:
    """Mean win rate at the start, in the peak window, and at the end."""
    last = max(steps for steps, _ in win_rates)
    return (
        _mean([rate for steps, rate in win_rates if steps <= START_END]),
        _mean([rate for steps, rate in win_rates if PEAK_WINDOW[0] <= steps <= PEAK_WINDOW[1]]),
        _mean([rate for steps, rate in win_rates if steps > last - END_LENGTH]),
    )


def holds(start: float, peak: float, end: float) -> bool:
    return end >= start + MIN_GAIN and end >= peak - MAX_DROP


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="training output folders (with logs/progress.csv)")
    args = parser.parse_args()
    print(f"{'run':<24} start   peak    end  holds")
    for run in args.runs:
        start, peak, end = windows(read_win_rates(Path(run) / "logs" / "progress.csv"))
        print(f"{Path(run).name:<24} {start:>5.0%}  {peak:>5.0%}  {end:>5.0%}  {'yes' if holds(start, peak, end) else 'no'}")


if __name__ == "__main__":
    main()
