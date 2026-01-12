from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Dict

# 允许从仓库根目录直接运行该脚本：python generalization_eval/analysis/summarize.py ...
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from generalization_eval.common.logging_utils import read_jsonl  # noqa: E402


def summarize_file(path: Path) -> Dict[str, Any]:
    rows = read_jsonl(path)
    total = len(rows)
    setup_ok = [r for r in rows if r.get("setup_ok", False)]
    setup_ok_n = len(setup_ok)

    succ_all = sum(1 for r in rows if r.get("success", False))
    succ_valid = sum(1 for r in setup_ok if r.get("success", False))

    def safe_mean(values):
        values = [v for v in values if v is not None]
        if not values:
            return None
        return sum(values) / len(values)

    dist_valid = [r.get("distance_xy") for r in setup_ok if r.get("distance_xy") is not None]
    steps_valid = [r.get("steps") for r in setup_ok if r.get("steps") is not None]

    return {
        "file": str(path),
        "condition": path.stem,
        "total_trials": total,
        "setup_ok": setup_ok_n,
        "setup_fail": total - setup_ok_n,
        "success_all": succ_all,
        "success_rate_all": (succ_all / total) if total else None,
        "success_valid": succ_valid,
        "success_rate_valid": (succ_valid / setup_ok_n) if setup_ok_n else None,
        "mean_distance_xy_valid": safe_mean(dist_valid),
        "mean_steps_valid": safe_mean(steps_valid),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="例如 generalization_eval/runs/20250101_120000")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    jsonl_files = sorted(run_dir.rglob("*.jsonl"))
    if not jsonl_files:
        print("No .jsonl found")
        return

    out_csv = run_dir / "summary.csv"
    summaries = [summarize_file(p) for p in jsonl_files]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
