from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from data_collector import ExpertDemoCollector, NoiseType  # noqa: E402
from generalization_eval.common.logging_utils import make_run_dirs, write_jsonl  # noqa: E402
from generalization_eval.common.policy_loader import load_policy  # noqa: E402
from generalization_eval.common.rollout import run_policy_episode  # noqa: E402


def _obstacle_spec_from_scene(config: dict, height: float) -> list[dict]:
    # 在 cube 与 target 的中点放一个静态 box
    cube = np.array(config["cube_initial_pos"], dtype=float)
    tgt = np.array(config["target_pos"], dtype=float)
    center_xy = (cube[:2] + tgt[:2]) * 0.5

    # 放在桌面上方：中心 z = table + height/2
    center_world = [float(center_xy[0]), float(center_xy[1]), float(0.625 + height * 0.5)]

    # 让障碍物更像“挡板”：薄厚、较宽
    half_extents = [0.01, 0.12, float(height * 0.5)]
    rgba = [0.1, 0.1, 0.1, 1.0]
    return [{"center_world": center_world, "half_extents": half_extents, "rgba": rgba}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="data/policy_checkpoint.pth")
    parser.add_argument("--num_trials", type=int, default=100)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_step", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--out_base", type=str, default="generalization_eval")
    parser.add_argument("--run_id", type=str, default=None, help="将输出写入指定 run_id（runs/<run_id>）。")
    args = parser.parse_args()

    # 控制变量：只加障碍物（高度变化）
    object_spec = {"shape": "box", "half_extents": [0.025, 0.025, 0.025], "rgba": [0.85, 0.2, 0.2, 1.0]}

    heights = [0.0, 0.03, 0.06, 0.10]

    run_paths = make_run_dirs(args.out_base, category="extra/obstacles/box", run_id=args.run_id)
    policy = load_policy(args.checkpoint, device="cpu")
    collector = ExpertDemoCollector(gui=bool(args.render), noise_type=NoiseType.NONE)

    seeds = [args.seed_start + i * args.seed_step for i in range(args.num_trials)]

    for h in heights:
        name = f"obstacle_h_{int(h*100):02d}cm"
        out_file = run_paths.category_dir / f"{name}.jsonl"
        records = []

        for trial_idx, seed in enumerate(seeds):
            # 先 setup 一次拿到 cube/target，再用 obstacle_specs 重新 setup（保持同 seed 的随机性）
            base_cfg = collector.setup_scene(randomize=True, seed=seed, object_spec=object_spec)
            if base_cfg is None:
                records.append({"category": "extra_obstacles", "condition": name, "trial": trial_idx, "seed": seed, "setup_ok": False, "success": False, "reason": "setup_failed", "obstacle_height": float(h)})
                continue

            obstacle_specs = _obstacle_spec_from_scene(base_cfg, height=float(h)) if h > 1e-9 else None

            config = collector.setup_scene(randomize=True, seed=seed, object_spec=object_spec, obstacle_specs=obstacle_specs)
            if config is None:
                records.append({"category": "extra_obstacles", "condition": name, "trial": trial_idx, "seed": seed, "setup_ok": False, "success": False, "reason": "setup_failed", "obstacle_height": float(h)})
                continue

            res = run_policy_episode(collector=collector, model=policy.model, stats=policy.stats, render=bool(args.render))

            records.append({
                "category": "extra_obstacles",
                "condition": name,
                "trial": trial_idx,
                "seed": seed,
                "setup_ok": res.setup_ok,
                "success": res.success,
                "steps": res.steps,
                "distance_xy": res.distance_xy,
                "final_pos": res.final_pos,
                "release_step": res.release_step,
                "reason": res.reason,
                "obstacle_height": float(h),
                "obstacles_n": config.get("obstacles_n"),
            })

        write_jsonl(out_file, records)
        print(f"[extra/obstacles/box] wrote {len(records)} -> {out_file}")

    collector.close()
    print(f"Run dir: {run_paths.run_dir}")


if __name__ == "__main__":
    main()
