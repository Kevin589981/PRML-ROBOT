"""Plot success-rate curves for vision generalization eval.

Reads `runs/<run_id>/summary.csv` produced by `analysis/summarize.py` and generates
one PNG with multiple subplots (one per perturbation dimension).

Usage:
  python 84.8%/84.8%train/generalization_eval_vision/analysis/plot_success_curves.py \
    --run_dir 84.8%/84.8%train/generalization_eval_vision/runs/<run_id>

This module is also invoked from run_all.py when --plot is passed.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _parse_csv_line(line: str) -> List[str]:
    return [cell.strip() for cell in line.rstrip("\n").split(",")]


def load_summary_csv(summary_csv: Path) -> List[Dict[str, str]]:
    lines = summary_csv.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty CSV: {summary_csv}")

    header = _parse_csv_line(lines[0])
    rows: List[Dict[str, str]] = []
    for raw in lines[1:]:
        if not raw.strip():
            continue
        parts = _parse_csv_line(raw)
        if len(parts) != len(header):
            continue
        rows.append(dict(zip(header, parts)))
    return rows


def _to_float(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except Exception:
        return default


def _group_and_level(condition: str) -> Tuple[str, Optional[float], str]:
    """Return (group, numeric_level, label)."""
    if condition == "baseline":
        return ("baseline", 0.0, "baseline")

    patterns = [
        ("basket_xnoise", re.compile(r"^basket_xnoise_(?P<val>[0-9.]+)$")),
        ("cam_pos_noise", re.compile(r"^cam_pos_noise_(?P<val>[0-9.]+)$")),
        ("action_noise", re.compile(r"^action_noise_(?P<val>[0-9.]+)$")),
        ("cube_friction", re.compile(r"^cube_friction_(?P<val>[0-9.]+)$")),
        ("cube_range", re.compile(r"^cube_range_k(?P<val>[0-9.]+)$")),
        ("ee_pos_noise", re.compile(r"^ee_pos_noise_(?P<val>[0-9.]+)$")),
    ]

    for group, rx in patterns:
        m = rx.match(condition)
        if not m:
            continue
        v = _to_float(m.group("val"), default=0.0)
        if group == "cube_range":
            return (group, v, f"x{v:.2f}")
        if group in {"basket_xnoise", "cam_pos_noise"}:
            return (group, v, f"{v:.2f}m")
        if group == "action_noise":
            return (group, v, f"{v:.3f}")
        return (group, v, f"{v:.2f}")

    return ("other", None, condition)


def rows_to_series(rows: List[Dict[str, str]]) -> Dict[str, List[Tuple[float, float, str]]]:
    """Map group -> list of (x, success_rate(0-1), label)."""
    series: Dict[str, List[Tuple[float, float, str]]] = {}

    for row in rows:
        cond = (row.get("condition") or "").strip()
        if not cond:
            continue

        group, level, label = _group_and_level(cond)
        if level is None:
            continue

        y = _to_float(row.get("success_rate", "0"), default=0.0)
        series.setdefault(group, []).append((float(level), float(y), label))

    for k in list(series.keys()):
        series[k] = sorted(series[k], key=lambda t: t[0])

    return series


def plot_series(
    out_path: Path,
    series: Dict[str, List[Tuple[float, float, str]]],
    baseline_line: Optional[float] = None,
) -> Path:
    if not series:
        raise ValueError("No series to plot")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    groups = [g for g in sorted(series.keys()) if g != "baseline"]
    # Keep baseline separate; draw as text only.

    n = len(groups)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, max(4, 3.2 * rows)), squeeze=False)
    fig.suptitle("Vision Generalization: Success Curves")

    for idx, group in enumerate(groups):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]

        pts = series[group]
        x = [p[0] for p in pts]
        y = [p[1] * 100.0 for p in pts]
        x_labels = [p[2] for p in pts]

        ax.plot(x, y, marker="o", linewidth=2)
        if baseline_line is not None:
            ax.axhline(float(baseline_line) * 100.0, linestyle="--", linewidth=1.8, color="black")
        ax.set_title(group)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_ylabel("Success Rate (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha="right")

    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis("off")

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_from_summary(run_dir: Path, baseline_line: Optional[float] = None) -> Path:
    summary_csv = run_dir / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(summary_csv)

    rows = load_summary_csv(summary_csv)
    series = rows_to_series(rows)

    plots_dir = run_dir / "plots"
    out_png = plots_dir / "success_curves.png"
    return plot_series(out_png, series, baseline_line=baseline_line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--baseline_line", type=float, default=None)
    args = parser.parse_args()

    out = plot_from_summary(Path(args.run_dir), baseline_line=args.baseline_line)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
