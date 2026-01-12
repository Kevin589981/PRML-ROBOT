from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
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

    # 控制变量：只改恢复系数
    object_spec = {"shape": "box", "half_extents": [0.025, 0.025, 0.025], "rgba": [0.85, 0.2, 0.2, 1.0]}

    rs = [0.0, 0.2, 0.5, 0.9]

    run_paths = make_run_dirs(args.out_base, category="extra/physics/restitution", run_id=args.run_id)
    policy = load_policy(args.checkpoint, device="cpu")
    collector = ExpertDemoCollector(gui=bool(args.render), noise_type=NoiseType.NONE)

    seeds = [args.seed_start + i * args.seed_step for i in range(args.num_trials)]

    for r in rs:
        name = f"restitution_{r:.2f}"
        out_file = run_paths.category_dir / f"{name}.jsonl"
        records = []

        for trial_idx, seed in enumerate(seeds):
            config = collector.setup_scene(
                randomize=True,
                seed=seed,
                object_spec=object_spec,
                restitution=float(r),
            )

            if config is None:
                records.append({
                    "category": "extra_restitution",
                    "condition": name,
                    "trial": trial_idx,
                    "seed": seed,
                    "setup_ok": False,
                    "success": False,
                    "reason": "setup_failed",
                    "restitution": float(r),
                })
                continue

            res = run_policy_episode(collector=collector, model=policy.model, stats=policy.stats, render=bool(args.render))

            records.append({
                "category": "extra_restitution",
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
                "restitution": float(r),
            })

        write_jsonl(out_file, records)
        print(f"[extra/physics/restitution] wrote {len(records)} -> {out_file}")

    collector.close()
    print(f"Run dir: {run_paths.run_dir}")


if __name__ == "__main__":
    main()
