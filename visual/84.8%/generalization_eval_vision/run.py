#!/usr/bin/env python3
"""
Temporary script: Generate generalization evaluation plots from existing run results
Usage: python plot_from_existing.py --run_id run_20260109_183433
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_csv_line(line: str) -> List[str]:
    return [cell.strip() for cell in line.rstrip("\n").split(",")]


def load_summary_csv(summary_csv: Path) -> List[Dict[str, str]]:
    """Load data from CSV file"""
    lines = summary_csv.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty CSV file: {summary_csv}")

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
    """Safely convert to float"""
    try:
        return float(s)
    except Exception:
        return default


def _group_and_level(condition: str) -> Tuple[str, Optional[float], str]:
    """Extract group, numeric value and label from condition name"""
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
        if group in {"basket_xnoise", "cam_pos_noise", "ee_pos_noise"}:
            return (group, v, f"{v:.2f}m")
        if group == "action_noise":
            return (group, v, f"{v:.3f}")
        return (group, v, f"{v:.2f}")

    return ("other", None, condition)


def rows_to_series(rows: List[Dict[str, str]]) -> Dict[str, List[Tuple[float, float, str]]]:
    """Convert rows to series data"""
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
    """Plot series charts"""
    if not series:
        raise ValueError("No series data to plot")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    groups = [g for g in sorted(series.keys()) if g != "baseline"]
    n = len(groups)
    
    if n == 0:
        print("Warning: No condition groups found")
        return out_path
    
    # Calculate plot layout
    cols = 2
    rows = (n + cols - 1) // cols
    fig_height = max(4, 3.2 * rows)
    
    print(f"Generating plots for {n} condition groups, layout: {rows} rows × {cols} columns")
    print(f"Condition groups: {groups}")

    fig, axes = plt.subplots(rows, cols, figsize=(14, fig_height), squeeze=False)
    
    # Set main title
    fig.suptitle("Vision Policy Generalization Evaluation", fontsize=16, fontweight='bold')

    for idx, group in enumerate(groups):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]

        pts = series[group]
        x = [p[0] for p in pts]
        y = [p[1] * 100.0 for p in pts]  # Convert to percentage
        labels = [p[2] for p in pts]
        
        # Print debug info
        print(f"  {group}: {len(pts)} points")
        for i, (xi, yi, lbl) in enumerate(zip(x, y, labels)):
            print(f"    Point {i}: x={xi}, y={yi:.1f}%, label='{lbl}'")

        # Plot main curve
        ax.plot(x, y, marker='o', linewidth=2.5, markersize=8, 
                color='#2E86AB', markerfacecolor='#A23B72', markeredgewidth=1.5)
        
        # Add data point labels
        for xi, yi, lbl in zip(x, y, labels):
            ax.annotate(f"{yi:.1f}%", (xi, yi), 
                       xytext=(0, 8), textcoords='offset points',
                       ha='center', fontsize=9, color='#333')

        # Baseline reference line
        if baseline_line is not None:
            ax.axhline(float(baseline_line) * 100.0, 
                      linestyle='--', linewidth=2, 
                      color='#F18F01', alpha=0.7,
                      label=f"Baseline ({baseline_line*100:.1f}%)")
            ax.legend(loc='lower left', fontsize=9)

        # Find baseline data
        if "baseline" in series and series["baseline"]:
            baseline_val = series["baseline"][0][1] * 100.0
            ax.axhline(baseline_val, linestyle=':', linewidth=1.5, 
                      color='#C73E1D', alpha=0.6,
                      label=f"Exp. baseline")
            ax.legend(loc='lower left', fontsize=9)

        # Set chart properties
        # Format group name for display
        display_name = group.replace('_', ' ').title()
        if group == "ee_pos_noise":
            display_name = "EE Pos Noise"
        elif group == "cam_pos_noise":
            display_name = "Cam Pos Noise"
        elif group == "basket_xnoise":
            display_name = "Basket X Noise"
            
        ax.set_title(f"{display_name}", fontsize=13, fontweight='bold', pad=12)
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.8)
        ax.set_ylabel("Success Rate (%)", fontsize=11)
        ax.set_xlabel("Perturbation Level", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
        
        # Set background color
        ax.set_facecolor('#F8F9FA')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide extra subplots
    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis('off')

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    # Add overall annotation
    if "baseline" in series and series["baseline"]:
        baseline_val = series["baseline"][0][1] * 100.0
        fig.text(0.02, 0.02, f"Baseline Success Rate: {baseline_val:.1f}%", 
                fontsize=10, style='italic', color='#666')
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return out_path


def plot_from_summary(run_dir: Path, baseline_line: Optional[float] = None) -> Path:
    """Generate plots from summary CSV"""
    summary_csv = run_dir / "summary.csv"
    if not summary_csv.exists():
        # Try to find in parent directory
        parent_summary = run_dir.parent / "summary.csv"
        if parent_summary.exists():
            summary_csv = parent_summary
            print(f"Found summary.csv in current directory: {summary_csv}")
        else:
            # Search in run directory
            for file in run_dir.rglob("summary.csv"):
                summary_csv = file
                print(f"Found summary.csv in subdirectory: {summary_csv}")
                break
    
    if not summary_csv.exists():
        raise FileNotFoundError(f"summary.csv not found, please check path: {run_dir}")

    print(f"Loading summary file: {summary_csv}")
    rows = load_summary_csv(summary_csv)
    print(f"Loaded {len(rows)} rows of data")
    
    # Show all conditions
    print("\nFound conditions:")
    for row in rows:
        cond = row.get("condition", "").strip()
        success_rate = row.get("success_rate", "0").strip()
        print(f"  - {cond}: {success_rate}")
    
    series = rows_to_series(rows)
    print(f"\nParsed series: {list(series.keys())}")

    # Create plots directory
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_png = plots_dir / "success_curves.png"
    
    return plot_series(out_png, series, baseline_line=baseline_line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate generalization evaluation plots from existing run results")
    parser.add_argument("--run_dir", type=str, required=False,
                       help="Path to run directory (e.g., runs/run_20260109_183433)")
    parser.add_argument("--run_id", type=str, required=False,
                       help="Run ID (e.g., run_20260109_183433, will look in runs folder of current directory)")
    parser.add_argument("--baseline_line", type=float, default=None,
                       help="Baseline success rate reference line (0-1)")
    parser.add_argument("--project_root", type=str, default=".",
                       help="Project root path (default: current directory)")
    
    args = parser.parse_args()
    
    # Determine run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.run_id:
        # Look in runs folder of current directory
        project_root = Path(args.project_root)
        run_dir = project_root / "runs" / args.run_id
    else:
        parser.error("Must provide either --run_dir or --run_id")
    
    if not run_dir.exists():
        # Try to automatically find recent runs
        project_root = Path(args.project_root)
        runs_dir = project_root / "runs"
        if runs_dir.exists():
            # Find all run directories
            run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
            if run_dirs:
                print(f"Specified run directory does not exist: {run_dir}")
                print(f"Available run directories:")
                for i, d in enumerate(run_dirs[-5:]):  # Show last 5
                    print(f"  {i+1}. {d.name}")
                choice = input(f"\nSelect run directory to use (1-{len(run_dirs)}, or 'q' to quit): ")
                if choice.lower() != 'q' and choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(run_dirs):
                        run_dir = run_dirs[idx]
                        print(f"Using run directory: {run_dir}")
                    else:
                        print("Invalid selection")
                        return
                else:
                    return
            else:
                print(f"No run directories found in {runs_dir}")
                return
        else:
            print(f"Runs directory not found: {runs_dir}")
            return
    
    print(f"Using run directory: {run_dir}")
    
    try:
        out_png = plot_from_summary(run_dir, baseline_line=args.baseline_line)
        print(f"\nPlot generated: {out_png}")
        print(f"File size: {out_png.stat().st_size / 1024:.1f} KB")
    except Exception as e:
        print(f"Error generating plot: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()