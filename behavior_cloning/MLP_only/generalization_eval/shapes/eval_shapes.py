from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

# 让脚本可以从仓库根目录导入 data_collector/train_bc
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from data_collector import ExpertDemoCollector, NoiseType  # noqa: E402

from generalization_eval.common.logging_utils import make_run_dirs, write_jsonl  # noqa: E402
from generalization_eval.common.policy_loader import load_policy  # noqa: E402
from generalization_eval.common.rollout import run_policy_episode  # noqa: E402


def shape_conditions(assets_dir: Path) -> dict[str, dict]:
    tri_mesh = assets_dir / "triangular_prism.obj"
    return {
        # 基线：与训练 cube 尺寸一致的 box（5cm 立方体）
        "box_cube": {"shape": "box", "half_extents": [0.025, 0.025, 0.025], "rgba": [0.85, 0.2, 0.2, 1.0]},
        "sphere": {"shape": "sphere", "radius": 0.025, "rgba": [0.2, 0.6, 0.9, 1.0]},
        "tall_box": {"shape": "box", "half_extents": [0.02, 0.02, 0.04], "rgba": [0.9, 0.6, 0.2, 1.0]},
        "flat_box": {"shape": "box", "half_extents": [0.04, 0.02, 0.02], "rgba": [0.6, 0.2, 0.9, 1.0]},
        "cylinder": {"shape": "cylinder", "radius": 0.025, "height": 0.05, "rgba": [0.2, 0.9, 0.4, 1.0]},
        "triangular_prism": {"shape": "mesh", "mesh_path": str(tri_mesh), "mesh_scale": [1.0, 1.0, 1.0], "half_height": 0.025,
                              "rgba": [0.7, 0.7, 0.2, 1.0]},
    }


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

    assets_dir = (Path(__file__).resolve().parents[1] / "assets").resolve()
    conds = shape_conditions(assets_dir)

    run_paths = make_run_dirs(args.out_base, category="shapes", run_id=args.run_id)

    policy = load_policy(args.checkpoint, device="cpu")

    collector = ExpertDemoCollector(gui=bool(args.render), noise_type=NoiseType.NONE)

    seeds = [args.seed_start + i * args.seed_step for i in range(args.num_trials)]

    for name, object_spec in conds.items():
        out_file = run_paths.category_dir / f"{name}.jsonl"
        records = []

        for trial_idx, seed in enumerate(seeds):
            config = collector.setup_scene(
                randomize=True,
                seed=seed,
                object_spec=object_spec,
                cube_support_height=0.0,
                target_support_height=0.0,
            )

            if config is None:
                records.append({
                    "category": "shapes",
                    "condition": name,
                    "trial": trial_idx,
                    "seed": seed,
                    "setup_ok": False,
                    "success": False,
                    "reason": "setup_failed",
                })
                continue

            res = run_policy_episode(
                collector=collector,
                model=policy.model,
                stats=policy.stats,
                render=bool(args.render),
            )

            records.append({
                "category": "shapes",
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
                "config": {
                    "cube_support_height": config.get("cube_support_height"),
                    "target_support_height": config.get("target_support_height"),
                    "cube_surface_height": config.get("cube_surface_height"),
                    "target_surface_height": config.get("target_surface_height"),
                },
            })

        write_jsonl(out_file, records)
        print(f"[shapes] wrote {len(records)} -> {out_file}")

    collector.close()
    print(f"Run dir: {run_paths.run_dir}")


if __name__ == "__main__":
    main()
