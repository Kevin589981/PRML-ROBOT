from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SummaryRow:
    file: str
    condition: str
    total_trials: int
    success: int
    success_rate: float
    mean_steps: float
    mean_distance_xy: float


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def summarize_run(run_dir: str | Path) -> Path:
    """Summarize all jsonl under run_dir into summary.csv."""
    run_dir = Path(run_dir)
    out_csv = run_dir / "summary.csv"

    jsonl_files = sorted(run_dir.rglob("*.jsonl"))

    rows: list[SummaryRow] = []
    for jf in jsonl_files:
        trials = list(_iter_jsonl(jf))
        if not trials:
            continue

        total = len(trials)
        succ = sum(1 for t in trials if t.get("success"))
        sr = float(succ) / float(total) if total else 0.0

        steps_list = [float(t.get("steps", 0)) for t in trials]
        dist_list = [float(t.get("final_distance_xy", float("inf"))) for t in trials]

        mean_steps = sum(steps_list) / float(len(steps_list)) if steps_list else 0.0
        finite_dists = [d for d in dist_list if d != float("inf")]
        mean_dist = sum(finite_dists) / float(len(finite_dists)) if finite_dists else float("inf")

        condition = jf.stem
        rows.append(
            SummaryRow(
                file=str(jf),
                condition=condition,
                total_trials=total,
                success=succ,
                success_rate=sr,
                mean_steps=mean_steps,
                mean_distance_xy=mean_dist,
            )
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "file",
                "condition",
                "total_trials",
                "success",
                "success_rate",
                "mean_steps",
                "mean_distance_xy",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.file,
                    r.condition,
                    r.total_trials,
                    r.success,
                    f"{r.success_rate:.6f}",
                    f"{r.mean_steps:.3f}",
                    f"{r.mean_distance_xy:.6f}" if r.mean_distance_xy != float("inf") else "inf",
                ]
            )

    return out_csv
