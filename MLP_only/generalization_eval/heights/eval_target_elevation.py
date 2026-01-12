from __future__ import annotations

import argparse
from pathlib import Path
import sys

# 让脚本可以从仓库根目录导入 data_collector/train_bc
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from data_collector import ExpertDemoCollector, NoiseType  # noqa: E402

from generalization_eval.common.logging_utils import make_run_dirs, write_jsonl  # noqa: E402
from generalization_eval.common.policy_loader import load_policy  # noqa: E402
from generalization_eval.common.rollout import run_policy_episode  # noqa: E402


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

    # 固定物体为 5cm 立方体（控制变量：只改目标海拔）
    object_spec = {"shape": "box", "half_extents": [0.025, 0.025, 0.025], "rgba": [0.85, 0.2, 0.2, 1.0]}

    heights = [0.00, 0.01, 0.02, 0.05, 0.10, 0.20]

    run_paths = make_run_dirs(args.out_base, category="heights/target", run_id=args.run_id)

    policy = load_policy(args.checkpoint, device="cpu")
    collector = ExpertDemoCollector(gui=bool(args.render), noise_type=NoiseType.NONE)

    seeds = [args.seed_start + i * args.seed_step for i in range(args.num_trials)]

    for h in heights:
        name = f"target_support_{int(h*100):02d}cm"
        out_file = run_paths.category_dir / f"{name}.jsonl"
        records = []

        for trial_idx, seed in enumerate(seeds):
            config = collector.setup_scene(
                randomize=True,
                seed=seed,
                object_spec=object_spec,
                cube_support_height=0.0,
                target_support_height=float(h),
            )

            if config is None:
                records.append({
                    "category": "heights_target",
                    "condition": name,
                    "trial": trial_idx,
                    "seed": seed,
                    "setup_ok": False,
                    "success": False,
                    "reason": "setup_failed",
                    "target_support_height": float(h),
                })
                continue

            res = run_policy_episode(
                collector=collector,
                model=policy.model,
                stats=policy.stats,
                render=bool(args.render),
            )

            records.append({
                "category": "heights_target",
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
                "cube_support_height": 0.0,
                "target_support_height": float(h),
            })

        write_jsonl(out_file, records)
        print(f"[heights/target] wrote {len(records)} -> {out_file}")

    collector.close()
    print(f"Run dir: {run_paths.run_dir}")


if __name__ == "__main__":
    main()
