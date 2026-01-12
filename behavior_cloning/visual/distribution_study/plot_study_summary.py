#!/usr/bin/env python3
"""Plot success curves for distribution_study.

This script is modeled after 84.8%/generalization_eval_vision/run.py, but adapted to
this study's naming convention and JSON outputs.

Primary input:
- study_summary.json (a dict mapping condition -> success_rate)

Fallback:
- If study_summary.json is empty/invalid, it will scan experiment_*/result.json.

Usage examples:
- python plot_study_summary.py
- python plot_study_summary.py --summary_json study_summary.json
- python plot_study_summary.py --out plots/success_curves.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _load_json_maybe(path: Path) -> Optional[object]:
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def load_summary_mapping(summary_json: Path) -> Optional[Dict[str, float]]:
    """Try to load study_summary.json into a mapping: condition -> success_rate (percent)."""
    obj = _load_json_maybe(summary_json)
    if obj is None:
        return None

    # Common case: {"baseline": 28.0, "cube_75": 34.0, ...}
    if isinstance(obj, dict):
        # If it's already mapping to numbers
        ok = True
        out: Dict[str, float] = {}
        for k, v in obj.items():
            if isinstance(k, str):
                out[k] = _safe_float(v, default=float("nan"))
            else:
                ok = False
                break
        if ok and out:
            return out

        # If it is wrapped like {"results": {...}}
        maybe_results = obj.get("results") if hasattr(obj, "get") else None
        if isinstance(maybe_results, dict) and maybe_results:
            return {str(k): _safe_float(v) for k, v in maybe_results.items()}

    # Less likely: list of {condition, success_rate}
    if isinstance(obj, list):
        out2: Dict[str, float] = {}
        for row in obj:
            if not isinstance(row, dict):
                continue
            cond = str(row.get("condition", "")).strip()
            if not cond:
                continue
            out2[cond] = _safe_float(row.get("success_rate"), default=float("nan"))
        if out2:
            return out2

    return None


def scan_experiment_results(study_dir: Path) -> Dict[str, float]:
    """Scan experiment_*/result.json and return mapping condition -> success_rate (percent)."""
    mapping: Dict[str, float] = {}
    for exp_dir in sorted(study_dir.glob("experiment_*")):
        if not exp_dir.is_dir():
            continue
        cond = exp_dir.name[len("experiment_") :]
        result_path = exp_dir / "result.json"
        obj = _load_json_maybe(result_path)
        if not isinstance(obj, dict):
            continue
        sr = obj.get("success_rate")
        if sr is None:
            continue
        mapping[cond] = _safe_float(sr)
    return mapping


def _group_and_level(condition: str) -> Tuple[str, Optional[float], str]:
    """Extract group, numeric x-value, and x-axis label from condition name."""
    if condition == "baseline":
        return ("baseline", 0.0, "baseline")

    m = re.match(r"^ee_init_(?P<mm>\d+)mm$", condition)
    if m:
        mm = _safe_float(m.group("mm"), default=0.0)
        return ("ee_init", mm, f"{int(mm)}mm")

    m = re.match(r"^basket_(?P<pct>\d+)$", condition)
    if m:
        pct = _safe_float(m.group("pct"), default=0.0)
        return ("basket", pct, f"{int(pct)}%")

    m = re.match(r"^cube_(?P<pct>\d+)$", condition)
    if m:
        pct = _safe_float(m.group("pct"), default=0.0)
        return ("cube", pct, f"{int(pct)}%")

    return ("other", None, condition)


def mapping_to_series(mapping: Dict[str, float]) -> Dict[str, List[Tuple[float, float, str]]]:
    """Convert mapping into series: group -> [(x, y_percent, label)]."""
    series: Dict[str, List[Tuple[float, float, str]]] = {}
    
    # 1. Parse all items
    for cond, success_rate in mapping.items():
        group, level, label = _group_and_level(cond)
        if level is None:
            continue
        y = _safe_float(success_rate)
        series.setdefault(group, []).append((float(level), float(y), label))

    # 2. Extract baseline value if present
    baseline_val: Optional[float] = None
    if "baseline" in series and series["baseline"]:
        baseline_val = series["baseline"][0][1]

    # 3. Inject baseline into specific groups if available
    if baseline_val is not None:
        # Basket: 100% is baseline
        if "basket" in series:
             series["basket"].append((100.0, baseline_val, "100%"))
        # Cube: 100% is baseline
        if "cube" in series:
             series["cube"].append((100.0, baseline_val, "100%"))
        # EE Init: 40mm is baseline (based on run_study.py BASE definition)
        if "ee_init" in series:
             series["ee_init"].append((40.0, baseline_val, "40mm"))

    # 4. Sort and dedup
    for k in list(series.keys()):
        # Use dict to deduplicate by x-level
        unique = {}
        for item in series[k]:
            unique[item[0]] = item
        series[k] = sorted(unique.values(), key=lambda t: t[0])

    return series


def plot_series(out_path: Path, series: Dict[str, List[Tuple[float, float, str]]]) -> Path:
    if not series:
        raise ValueError("No data to plot")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_val: Optional[float] = None
    if "baseline" in series and series["baseline"]:
        baseline_val = series["baseline"][0][1]

    groups = [g for g in sorted(series.keys()) if g != "baseline"]
    if not groups:
        raise ValueError("No condition groups found (expected ee_init/basket/cube)")

    cols = 2
    rows = (len(groups) + cols - 1) // cols
    fig_height = max(4, 3.2 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=(14, fig_height), squeeze=False)

    fig.suptitle("Distribution Study: Success Curves", fontsize=16, fontweight="bold")

    display_names = {
        "ee_init": "Reset EE Init Noise",
        "basket": "Basket X Noise Scale",
        "cube": "Cube Range Scale",
    }

    for idx, group in enumerate(groups):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]

        pts = series[group]
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        labels = [p[2] for p in pts]

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.5,
            markersize=8,
            color="#2E86AB",
            markerfacecolor="#A23B72",
            markeredgewidth=1.5,
        )

        for xi, yi in zip(x, y):
            ax.annotate(
                f"{yi:.1f}%",
                (xi, yi),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color="#333",
            )

        if baseline_val is not None:
            ax.axhline(
                baseline_val,
                linestyle="--",
                linewidth=2,
                color="#F18F01",
                alpha=0.75,
                label=f"Baseline ({baseline_val:.1f}%)",
            )
            ax.legend(loc="lower left", fontsize=9)

        ax.set_title(display_names.get(group, group), fontsize=13, fontweight="bold", pad=12)
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle=":", alpha=0.3, linewidth=0.8)
        ax.set_ylabel("Success Rate (%)", fontsize=11)
        ax.set_xlabel("Level", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
        ax.set_facecolor("#F8F9FA")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(len(groups), rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis("off")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    if baseline_val is not None:
        fig.text(
            0.02,
            0.02,
            f"Baseline Success Rate: {baseline_val:.1f}%",
            fontsize=10,
            style="italic",
            color="#666",
        )

    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot distribution study success curves")
    parser.add_argument(
        "--study_dir",
        type=str,
        default=".",
        help="Distribution study directory (default: current directory)",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default="study_summary.json",
        help="Summary JSON path (default: study_summary.json)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="plots/success_curves.png",
        help="Output PNG path (default: plots/success_curves.png)",
    )
    parser.add_argument(
        "--write_reconstructed_summary",
        action="store_true",
        help="If summary_json is empty/invalid, reconstruct from experiment_*/result.json and write a new JSON next to output.",
    )

    args = parser.parse_args()

    study_dir = Path(args.study_dir)
    summary_path = (study_dir / args.summary_json).resolve() if not Path(args.summary_json).is_absolute() else Path(args.summary_json)

    mapping = load_summary_mapping(summary_path)
    source = "summary_json"
    if mapping is None or not mapping:
        mapping = scan_experiment_results(study_dir)
        source = "scan_experiment_results"

    if not mapping:
        raise SystemExit(f"No results found. Tried: {summary_path} and scanning {study_dir}/experiment_*/result.json")

    series = mapping_to_series(mapping)
    out_png = (study_dir / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_png = plot_series(out_png, series)

    print(f"Loaded {len(mapping)} conditions (source={source})")
    print(f"Plot generated: {out_png} ({out_png.stat().st_size/1024:.1f} KB)")

    if source != "summary_json" and args.write_reconstructed_summary:
        recon_path = out_png.parent / "study_summary.reconstructed.json"
        recon_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Reconstructed summary written: {recon_path}")


if __name__ == "__main__":
    main()
