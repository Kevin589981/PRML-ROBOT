"""Dataset sanity check for basket demo dataset.

What it checks:
- action delta stats (per-dim, norm percentiles)
- gripper open/close ratio
- rough phase segmentation using gripper_action transitions:
  approach: before first close
  carry: between first close and first re-open
  retreat: after first re-open

Run:
  python dataset_sanity_check.py
  python dataset_sanity_check.py --split train
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np


@dataclass
class OnlineStats:
    n: int = 0
    mean: Optional[np.ndarray] = None
    m2: Optional[np.ndarray] = None
    min_v: Optional[np.ndarray] = None
    max_v: Optional[np.ndarray] = None

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if self.mean is None:
            self.mean = np.zeros_like(x)
            self.m2 = np.zeros_like(x)
            self.min_v = x.copy()
            self.max_v = x.copy()
            self.n = 0

        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2
        self.min_v = np.minimum(self.min_v, x)
        self.max_v = np.maximum(self.max_v, x)

    def finalize(self) -> Dict[str, np.ndarray]:
        if self.mean is None or self.m2 is None:
            raise RuntimeError("No data")
        var = self.m2 / max(self.n - 1, 1)
        std = np.sqrt(var)
        return {
            "count": np.array([self.n], dtype=np.int64),
            "mean": self.mean,
            "std": std,
            "min": self.min_v,
            "max": self.max_v,
        }


def _first_crossing(binary: np.ndarray, target: int) -> Optional[int]:
    # binary: 0/1
    idx = np.where(binary == target)[0]
    return int(idx[0]) if idx.size else None


def _phase_indices(gripper_bin: np.ndarray) -> Tuple[slice, slice, slice]:
    """Return (approach, carry, retreat) slices."""
    # close event: first time it becomes 0
    close_idx = _first_crossing(gripper_bin, 0)
    if close_idx is None:
        # never closes: everything is approach
        return slice(0, len(gripper_bin)), slice(0, 0), slice(0, 0)

    # open event after close: first 1 after close
    after_close = gripper_bin[close_idx:]
    open_rel = _first_crossing(after_close, 1)
    if open_rel is None:
        # closes and never reopens
        return slice(0, close_idx), slice(close_idx, len(gripper_bin)), slice(0, 0)

    open_idx = close_idx + open_rel
    return slice(0, close_idx), slice(close_idx, open_idx), slice(open_idx, len(gripper_bin))


def iter_indices(meta: h5py.Group, split: str) -> Iterable[int]:
    if split == "all":
        # rely on num_trajectories attr for order
        n = int(meta.attrs.get("num_trajectories", 0))
        return range(n)
    key = "train_indices" if split == "train" else "val_indices"
    return [int(i) for i in meta[key][:]]


def _compute_split(f: h5py.File, split: str, depth_sample_step: int) -> Dict[str, object]:
    act_stats = OnlineStats()
    aux_stats = OnlineStats()

    phase_names = ["approach", "carry", "retreat"]
    phase_act_stats = {name: OnlineStats() for name in phase_names}
    phase_norms: Dict[str, List[float]] = {name: [] for name in phase_names}

    norms_all: List[float] = []
    gripper_open_count = 0
    gripper_close_count = 0
    transition_close_positions: List[int] = []
    transition_open_positions: List[int] = []
    missing_close = 0
    missing_reopen = 0

    depth_min = float("inf")
    depth_max = float("-inf")

    meta = f["metadata"]
    indices = list(iter_indices(meta, split))
    for ti, idx in enumerate(indices):
        g = f[f"trajectory_{idx:04d}"]
        actions = g["actions"][:]  # (T,4)
        aux = g["aux"][:]          # (T,aux_dim)

        for a in actions:
            act_stats.update(a)
            norms_all.append(float(np.linalg.norm(a[:3])))

        for u in aux:
            aux_stats.update(u)

        gripper_bin = (actions[:, 3] > 0.5).astype(np.int32)
        gripper_open_count += int(np.sum(gripper_bin == 1))
        gripper_close_count += int(np.sum(gripper_bin == 0))

        close_idx = _first_crossing(gripper_bin, 0)
        if close_idx is None:
            missing_close += 1
            slc = _phase_indices(gripper_bin)
        else:
            transition_close_positions.append(close_idx)
            after_close = gripper_bin[close_idx:]
            open_rel = _first_crossing(after_close, 1)
            if open_rel is None:
                missing_reopen += 1
            else:
                transition_open_positions.append(close_idx + open_rel)
            slc = _phase_indices(gripper_bin)

        slices = dict(zip(phase_names, slc))
        for name, s in slices.items():
            if s.stop is None or s.start is None:
                continue
            if s.stop - s.start <= 0:
                continue
            for a in actions[s]:
                phase_act_stats[name].update(a)
                phase_norms[name].append(float(np.linalg.norm(a[:3])))

        if "depth" in g and (ti % max(depth_sample_step, 1) == 0):
            d = g["depth"][:]
            depth_min = min(depth_min, float(np.min(d)))
            depth_max = max(depth_max, float(np.max(d)))

    norms = np.asarray(norms_all, dtype=np.float64)
    tot_g = gripper_open_count + gripper_close_count

    out: Dict[str, object] = {
        "split": split,
        "action": act_stats.finalize(),
        "aux": aux_stats.finalize(),
        "delta_norm": {
            "p50": float(np.percentile(norms, 50)),
            "p90": float(np.percentile(norms, 90)),
            "p95": float(np.percentile(norms, 95)),
            "p99": float(np.percentile(norms, 99)),
            "max": float(np.max(norms)),
        },
        "gripper": {
            "open": int(gripper_open_count),
            "close": int(gripper_close_count),
            "open_ratio": float(gripper_open_count / max(tot_g, 1)),
        },
        "transitions": {
            "close_positions": transition_close_positions,
            "open_positions": transition_open_positions,
            "missing_close": int(missing_close),
            "missing_reopen": int(missing_reopen),
        },
        "phases": {
            name: {
                "action": phase_act_stats[name].finalize() if phase_act_stats[name].n else None,
                "delta_norm": {
                    "p50": float(np.percentile(np.asarray(phase_norms[name]), 50)) if phase_norms[name] else None,
                    "p90": float(np.percentile(np.asarray(phase_norms[name]), 90)) if phase_norms[name] else None,
                },
            }
            for name in phase_names
        },
        "depth_sampled": None if depth_min == float("inf") else {"min": depth_min, "max": depth_max},
    }
    return out


def _print_report(path: str, meta: h5py.Group, cam_names: List[str], split_stats: Dict[str, object]) -> None:
    split = split_stats["split"]
    print("=== Dataset ===")
    print("path:", path)
    print("split:", split)
    print("num_trajectories:", int(meta.attrs.get("num_trajectories", -1)))
    print("camera_names:", cam_names)
    print("depth_min/max(attrs):", meta.attrs.get("depth_min"), meta.attrs.get("depth_max"))
    print("action_mean/std(meta):", meta["action_mean"][:], meta["action_std"][:])
    print("aux_mean/std(meta):", meta["aux_mean"][:], meta["aux_std"][:])

    a = split_stats["action"]
    print("\n=== Overall action stats (raw) ===")
    print("frames:", int(a["count"][0]))
    print("action mean:", a["mean"].astype(np.float32))
    print("action std :", a["std"].astype(np.float32))
    print("action min :", a["min"].astype(np.float32))
    print("action max :", a["max"].astype(np.float32))

    dn = split_stats["delta_norm"]
    print("delta_norm p50:", dn["p50"])
    print("delta_norm p90:", dn["p90"])
    print("delta_norm p95:", dn["p95"])
    print("delta_norm p99:", dn["p99"])
    print("delta_norm max:", dn["max"])

    g = split_stats["gripper"]
    print("\n=== Gripper action distribution ===")
    print("open frames  :", g["open"], f"({g['open_ratio']*100:.1f}%)")
    print("close frames :", g["close"], f"({(1.0-g['open_ratio'])*100:.1f}%)")

    tr = split_stats["transitions"]
    print("\n=== Transition stats (per-trajectory indices) ===")
    if tr["close_positions"]:
        c = np.asarray(tr["close_positions"], dtype=np.float64)
        print("close@ mean/p50/p90:", float(np.mean(c)), int(np.median(c)), int(np.percentile(c, 90)))
    print("missing close trajectories:", tr["missing_close"])
    if tr["open_positions"]:
        o = np.asarray(tr["open_positions"], dtype=np.float64)
        print("reopen@ mean/p50/p90:", float(np.mean(o)), int(np.median(o)), int(np.percentile(o, 90)))
    print("missing reopen trajectories:", tr["missing_reopen"])

    print("\n=== Phase-wise stats (rough) ===")
    for name, v in split_stats["phases"].items():
        pa = v["action"]
        if pa is None:
            print(name, ": no frames")
            continue
        print(f"[{name}] frames={int(pa['count'][0])}")
        print("  mean:", pa["mean"].astype(np.float32))
        print("  std :", pa["std"].astype(np.float32))
        print("  delta_norm p50/p90:", v["delta_norm"]["p50"], v["delta_norm"]["p90"])

    if split_stats["depth_sampled"] is not None:
        print("\n=== Depth sampled min/max ===")
        print("depth min:", split_stats["depth_sampled"]["min"])
        print("depth max:", split_stats["depth_sampled"]["max"])


def _print_compare_delta(train_stats: Dict[str, object], val_stats: Dict[str, object]) -> None:
    print("\n=== Train vs Val delta (quick) ===")
    ta = train_stats["action"]
    va = val_stats["action"]
    mean_delta = (va["mean"] - ta["mean"]).astype(np.float32)
    std_ratio = (va["std"] / (ta["std"] + 1e-12)).astype(np.float32)
    print("action mean(val-train):", mean_delta)
    print("action std ratio(val/train):", std_ratio)

    tg = train_stats["gripper"]
    vg = val_stats["gripper"]
    print("gripper open_ratio train/val:", float(tg["open_ratio"]), float(vg["open_ratio"]))

    tdn = train_stats["delta_norm"]
    vdn = val_stats["delta_norm"]
    print("delta_norm p95 train/val:", float(tdn["p95"]), float(vdn["p95"]))
    print("delta_norm max train/val:", float(tdn["max"]), float(vdn["max"]))

    ttr = train_stats["transitions"]
    vtr = val_stats["transitions"]
    if ttr["close_positions"] and vtr["close_positions"]:
        tc = np.asarray(ttr["close_positions"], dtype=np.float64)
        vc = np.asarray(vtr["close_positions"], dtype=np.float64)
        print("close@ mean train/val:", float(np.mean(tc)), float(np.mean(vc)))
    if ttr["open_positions"] and vtr["open_positions"]:
        to = np.asarray(ttr["open_positions"], dtype=np.float64)
        vo = np.asarray(vtr["open_positions"], dtype=np.float64)
        print("reopen@ mean train/val:", float(np.mean(to)), float(np.mean(vo)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/basket_demos_dense_temporal.h5")
    ap.add_argument("--split", choices=["train", "val", "all"], default="all")
    ap.add_argument("--compare", action="store_true", help="print train vs val reports + delta summary")
    ap.add_argument("--depth-sample-step", type=int, default=20, help="sample every k trajectories for depth min/max")
    args = ap.parse_args()

    with h5py.File(args.path, "r") as f:
        meta = f["metadata"]
        cam_names_raw = meta.attrs.get("camera_names", "[]")
        if isinstance(cam_names_raw, (bytes, bytearray)):
            cam_names_raw = cam_names_raw.decode("utf-8")
        cam_names = json.loads(cam_names_raw)

        if args.compare:
            train_stats = _compute_split(f, "train", args.depth_sample_step)
            val_stats = _compute_split(f, "val", args.depth_sample_step)
            _print_report(args.path, meta, cam_names, train_stats)
            print("\n" + "-" * 60 + "\n")
            _print_report(args.path, meta, cam_names, val_stats)
            _print_compare_delta(train_stats, val_stats)
            return

        split_stats = _compute_split(f, args.split, args.depth_sample_step)
        _print_report(args.path, meta, cam_names, split_stats)


if __name__ == "__main__":
    main()
