from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    path = Path(path)
    out: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    category_dir: Path


def make_run_dirs(base_dir: str | Path, category: str, run_id: Optional[str] = None) -> RunPaths:
    base_dir = Path(base_dir)
    run_id = run_id or now_run_id()

    run_dir = ensure_dir(base_dir / "runs" / run_id)
    category_dir = ensure_dir(run_dir / category)
    return RunPaths(run_dir=run_dir, category_dir=category_dir)
