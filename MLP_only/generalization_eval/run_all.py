"""One-click runner for all generalization evaluation scripts.

Runs every eval script with a shared run_id so all logs land in one directory.
At the end it generates `analysis/summary.csv` and two comparison plots.

Typical usage:
  conda activate robot
  python generalization_eval/run_all.py --num_trials 100 --baseline 0.986

You can also use conda-run style:
  conda run -n robot python generalization_eval/run_all.py --num_trials 100
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


def _repo_root() -> Path:
    # This file lives at <repo>/generalization_eval/run_all.py
    return Path(__file__).resolve().parents[1]


def _run(cmd: List[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _eval_commands(
    python_bin: str,
    checkpoint: str,
    out_base: str,
    run_id: str,
    num_trials: int,
    seed_start: int,
    seed_step: int,
    render: bool,
    include_large_base_z: bool,
) -> List[Tuple[str, List[str]]]:
    render_flag = ["--render"] if render else []

    items: List[Tuple[str, List[str]]] = []

    # Basic
    items.append(
        (
            "shapes",
            [
                python_bin,
                "generalization_eval/shapes/eval_shapes.py",
                "--checkpoint",
                checkpoint,
                "--num_trials",
                str(num_trials),
                "--seed_start",
                str(seed_start),
                "--seed_step",
                str(seed_step),
                "--out_base",
                out_base,
                "--run_id",
                run_id,
                *render_flag,
            ],
        )
    )

    items.append(
        (
            "heights-initial",
            [
                python_bin,
                "generalization_eval/heights/eval_initial_elevation.py",
                "--checkpoint",
                checkpoint,
                "--num_trials",
                str(num_trials),
                "--seed_start",
                str(seed_start),
                "--seed_step",
                str(seed_step),
                "--out_base",
                out_base,
                "--run_id",
                run_id,
                *render_flag,
            ],
        )
    )

    items.append(
        (
            "heights-target",
            [
                python_bin,
                "generalization_eval/heights/eval_target_elevation.py",
                "--checkpoint",
                checkpoint,
                "--num_trials",
                str(num_trials),
                "--seed_start",
                str(seed_start),
                "--seed_step",
                str(seed_step),
                "--out_base",
                out_base,
                "--run_id",
                run_id,
                *render_flag,
            ],
        )
    )

    items.append(
        (
            "base-x",
            [
                python_bin,
                "generalization_eval/base/eval_base_x.py",
                "--checkpoint",
                checkpoint,
                "--num_trials",
                str(num_trials),
                "--seed_start",
                str(seed_start),
                "--seed_step",
                str(seed_step),
                "--out_base",
                out_base,
                "--run_id",
                run_id,
                *render_flag,
            ],
        )
    )

    items.append(
        (
            "base-y",
            [
                python_bin,
                "generalization_eval/base/eval_base_y.py",
                "--checkpoint",
                checkpoint,
                "--num_trials",
                str(num_trials),
                "--seed_start",
                str(seed_start),
                "--seed_step",
                str(seed_step),
                "--out_base",
                out_base,
                "--run_id",
                run_id,
                *render_flag,
            ],
        )
    )

    base_z_cmd = [
        python_bin,
        "generalization_eval/base/eval_base_z.py",
        "--checkpoint",
        checkpoint,
        "--num_trials",
        str(num_trials),
        "--seed_start",
        str(seed_start),
        "--seed_step",
        str(seed_step),
        "--out_base",
        out_base,
        "--run_id",
        run_id,
        *render_flag,
    ]
    if include_large_base_z:
        base_z_cmd.append("--include_large")
    items.append(("base-z", base_z_cmd))

    # Extra
    items.append(
        (
            "extra-pose",
            [
                python_bin,
                "generalization_eval/extra/pose/eval_initial_pose.py",
                "--checkpoint",
                checkpoint,
                "--num_trials",
                str(num_trials),
                "--seed_start",
                str(seed_start),
                "--seed_step",
                str(seed_step),
                "--out_base",
                out_base,
                "--run_id",
                run_id,
                *render_flag,
            ],
        )
    )

    for name, rel in [
        ("extra-mass", "generalization_eval/extra/physics/eval_mass.py"),
        ("extra-friction", "generalization_eval/extra/physics/eval_friction.py"),
        ("extra-restitution", "generalization_eval/extra/physics/eval_restitution.py"),
        ("extra-object-scale", "generalization_eval/extra/physics/eval_object_scale.py"),
        ("extra-gravity", "generalization_eval/extra/sim/eval_gravity.py"),
        ("extra-solver", "generalization_eval/extra/sim/eval_solver_iterations.py"),
        ("extra-timestep", "generalization_eval/extra/sim/eval_timestep.py"),
        ("extra-obs-noise", "generalization_eval/extra/noise/eval_obs_noise.py"),
        ("extra-obs-delay-drop", "generalization_eval/extra/noise/eval_obs_delay_drop.py"),
        ("extra-action-noise-delay", "generalization_eval/extra/noise/eval_action_noise_delay.py"),
        ("extra-push", "generalization_eval/extra/disturbance/eval_external_push.py"),
        ("extra-target-xy", "generalization_eval/extra/target/eval_target_xy_offset.py"),
        ("extra-obstacle", "generalization_eval/extra/obstacles/eval_obstacle_box.py"),
    ]:
        items.append(
            (
                name,
                [
                    python_bin,
                    rel,
                    "--checkpoint",
                    checkpoint,
                    "--num_trials",
                    str(num_trials),
                    "--seed_start",
                    str(seed_start),
                    "--seed_step",
                    str(seed_step),
                    "--out_base",
                    out_base,
                    "--run_id",
                    run_id,
                    *render_flag,
                ],
            )
        )

    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=str, default=sys.executable, help="Python interpreter to use")
    parser.add_argument("--checkpoint", type=str, default="data/policy_checkpoint.pth")
    parser.add_argument("--out_base", type=str, default="generalization_eval")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--num_trials", type=int, default=100)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_step", type=int, default=100)
    parser.add_argument("--baseline", type=float, default=0.986)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--include_large_base_z",
        action="store_true",
        help="base_z 额外加入 10/20cm（可能大量 setup_failed）",
    )
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")

    repo = _repo_root()

    cmds = _eval_commands(
        python_bin=args.python,
        checkpoint=args.checkpoint,
        out_base=args.out_base,
        run_id=run_id,
        num_trials=args.num_trials,
        seed_start=args.seed_start,
        seed_step=args.seed_step,
        render=args.render,
        include_large_base_z=args.include_large_base_z,
    )

    print(f"== Generalization all-in-one ==\nrun_id: {run_id}\nnum_trials: {args.num_trials}\n", flush=True)

    for name, cmd in cmds:
        print(f"\n---- {name} ----", flush=True)
        _run(cmd, cwd=repo)

    # Summarize
    run_dir = repo / args.out_base / "runs" / run_id
    _run(
        [
            args.python,
            "generalization_eval/analysis/summarize.py",
            "--run_dir",
            str(run_dir),
        ],
        cwd=repo,
    )

    # Plot
    _run(
        [
            args.python,
            "generalization_eval/analysis/plot_success_curves.py",
            "--run_dir",
            str(run_dir),
            "--baseline",
            str(args.baseline),
        ],
        cwd=repo,
    )

    print("\nDone.")
    print(f"Run directory: {run_dir}")
    print(f"Plots: {run_dir / 'analysis' / 'plots'}")


if __name__ == "__main__":
    main()
