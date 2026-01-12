"""Plot success-rate curves from generalization_eval analysis outputs.

This script is intentionally lightweight: it reads the `summary.csv` produced by
`generalization_eval/analysis/summarize.py` and generates two line charts:
- basic: shapes + heights + base
- extra: everything under extra/

Baseline is drawn as a horizontal line (e.g. 98.6%).

Usage:
  python generalization_eval/analysis/plot_success_curves.py \
      --run_dir generalization_eval/runs/<run_id> \
      --baseline 0.986
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _parse_csv_line(line: str) -> List[str]:
    # Minimal CSV parsing for our own generated summary (no quoted commas expected).
    return [cell.strip() for cell in line.rstrip("\n").split(",")]


def load_summary_csv(summary_csv: Path) -> List[Dict[str, str]]:
    lines = summary_csv.read_text().splitlines()
    if not lines:
        raise ValueError(f"Empty CSV: {summary_csv}")

    header = _parse_csv_line(lines[0])
    rows: List[Dict[str, str]] = []
    for raw in lines[1:]:
        if not raw.strip():
            continue
        parts = _parse_csv_line(raw)
        if len(parts) != len(header):
            # Skip malformed lines rather than fail hard.
            continue
        rows.append(dict(zip(header, parts)))
    return rows


def _to_float(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except Exception:
        return default


def _group_key(row: Dict[str, str]) -> str:
    # We infer category from the jsonl file path written by eval scripts, e.g.
    #   .../runs/<run_id>/shapes/sphere.jsonl           -> shapes
    #   .../runs/<run_id>/heights/initial/*.jsonl       -> heights/initial
    #   .../runs/<run_id>/extra/physics/mass/*.jsonl    -> extra/physics/mass
    file_path = row.get("file", "").strip()
    if not file_path:
        return "unknown"

    p = Path(file_path)
    parts = list(p.parts)
    # Find "runs/<run_id>" then take the remainder excluding filename
    try:
        runs_idx = parts.index("runs")
        # category path starts after run_id
        rel_parts = parts[runs_idx + 2 : -1]
    except ValueError:
        rel_parts = parts[-3:-1]  # best effort

    if not rel_parts:
        return "unknown"
    return "/".join(rel_parts)


def _label_for_row(row: Dict[str, str]) -> str:
    # Prefer human-friendly `condition_name`, else `condition_value`.
    name = row.get("condition_name", "").strip()
    if name:
        return name
    value = row.get("condition_value", "").strip()
    if value:
        return value
    return row.get("condition", "").strip() or "(unknown)"


def _is_basic_category(category: str) -> bool:
    category = category.strip().lower()
    return category.startswith("shapes") or category.startswith("heights") or category.startswith("base")


def _is_extra_category(category: str) -> bool:
    category = category.strip().lower()
    return category.startswith("extra")


def rows_to_series(
    rows: List[Dict[str, str]],
    category_filter,
) -> Dict[str, List[Tuple[str, float]]]:
    series: Dict[str, List[Tuple[str, float]]] = {}

    for row in rows:
        category = _group_key(row)
        if not category_filter(category):
            continue

        y = _to_float(row.get("success_rate_valid", row.get("success_rate", "0")), default=0.0)
        label = _label_for_row(row)

        series.setdefault(category, []).append((label, y))

    # Keep original order as much as possible, but for numeric labels (like cm or degrees),
    # try to sort by numeric value.
    signed_m_pattern = re.compile(r"_(?P<sign>[mp])(?P<val>\d+(?:\.\d+)?)m")
    cm_pattern = re.compile(r"(?P<val>\d+(?:\.\d+)?)cm")
    deg_pattern = re.compile(r"(?P<val>\d+(?:\.\d+)?)(?:deg|°)")
    any_number = re.compile(r"-?\d+(?:\.\d+)?")

    def maybe_numeric_key(item: Tuple[str, float]) -> Tuple[int, float, str]:
        label, _y = item

        m = signed_m_pattern.search(label)
        if m:
            val = float(m.group("val"))
            if m.group("sign") == "m":
                val = -val
            return (0, val, label)

        m = cm_pattern.search(label)
        if m:
            return (0, float(m.group("val")), label)

        m = deg_pattern.search(label)
        if m:
            return (0, float(m.group("val")), label)

        m = any_number.search(label)
        if m:
            return (0, float(m.group(0)), label)

        return (1, 0.0, label)

    for category, points in list(series.items()):
        series[category] = sorted(points, key=maybe_numeric_key)

    return series


def plot_series(
    out_path: Path,
    title: str,
    series: Dict[str, List[Tuple[str, float]]],
    baseline: float,
) -> None:
    if not series:
        raise ValueError(f"No data to plot for {title}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    categories = sorted(series.keys())
    n = len(categories)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, max(4, 3.2 * rows)), squeeze=False)
    fig.suptitle(title)

    for idx, category in enumerate(categories):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        points = series[category]

        x_labels = [p[0] for p in points]
        y = [p[1] * 100.0 for p in points]
        x = list(range(len(x_labels)))

        ax.plot(x, y, marker="o", linewidth=2)
        ax.axhline(baseline * 100.0, linestyle="--", linewidth=1.8, color="black")
        ax.set_title(category)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha="right")
        if c == 0:
            ax.set_ylabel("Success Rate (%)")

    # Hide unused axes
    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis("off")

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--baseline", type=float, default=0.986)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    # summarize.py currently writes to run_dir/summary.csv; keep compatibility if
    # a future version writes to run_dir/analysis/summary.csv.
    summary_csv = run_dir / "analysis" / "summary.csv"
    if not summary_csv.exists():
        fallback = run_dir / "summary.csv"
        if fallback.exists():
            summary_csv = fallback
        else:
            raise FileNotFoundError(
                f"Missing summary.csv, please run summarize.py first: {summary_csv} (or {fallback})"
            )

    rows = load_summary_csv(summary_csv)

    basic = rows_to_series(rows, _is_basic_category)
    extra = rows_to_series(rows, _is_extra_category)

    plots_dir = run_dir / "analysis" / "plots"
    plot_series(plots_dir / "basic_success.png", "Generalization (Basic)", basic, baseline=args.baseline)
    plot_series(plots_dir / "extra_success.png", "Generalization (Extra)", extra, baseline=args.baseline)


if __name__ == "__main__":
    main()
