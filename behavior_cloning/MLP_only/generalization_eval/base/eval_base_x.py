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

    # 固定物体为 5cm 立方体（控制变量：只改基座 X）
    object_spec = {"shape": "box", "half_extents": [0.025, 0.025, 0.025], "rgba": [0.85, 0.2, 0.2, 1.0]}

    # 单位：米。X 方向（前后）偏移，测试对基座位置的敏感性。
    offsets = [0.00, 0.01, 0.02, 0.05, 0.10, 0.20]
    offsets = sorted(set([-h for h in offsets[1:]] + offsets))

    run_paths = make_run_dirs(args.out_base, category="base/x", run_id=args.run_id)

    policy = load_policy(args.checkpoint, device="cpu")
    collector = ExpertDemoCollector(gui=bool(args.render), noise_type=NoiseType.NONE)

    seeds = [args.seed_start + i * args.seed_step for i in range(args.num_trials)]

    for dx in offsets:
        name = f"base_x_{dx:+.2f}m".replace("+", "p").replace("-", "m")
        out_file = run_paths.category_dir / f"{name}.jsonl"
        records = []

        for trial_idx, seed in enumerate(seeds):
            config = collector.setup_scene(
                randomize=True,
                seed=seed,
                object_spec=object_spec,
                cube_support_height=0.0,
                target_support_height=0.0,
                panda_base_offset=[float(dx), 0.0, 0.0],
            )

            if config is None:
                records.append({
                    "category": "base_x",
                    "condition": name,
                    "trial": trial_idx,
                    "seed": seed,
                    "setup_ok": False,
                    "success": False,
                    "reason": "setup_failed",
                    "panda_base_offset": [float(dx), 0.0, 0.0],
                })
                continue

            res = run_policy_episode(
                collector=collector,
                model=policy.model,
                stats=policy.stats,
                render=bool(args.render),
            )

            records.append({
                "category": "base_x",
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
                "panda_base": config.get("panda_base"),
                "panda_base_offset": config.get("panda_base_offset"),
            })

        write_jsonl(out_file, records)
        print(f"[base/x] wrote {len(records)} -> {out_file}")

    collector.close()
    print(f"Run dir: {run_paths.run_dir}")


if __name__ == "__main__":
    main()
