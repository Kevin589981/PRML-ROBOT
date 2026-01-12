from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


# Ensure we can import 84.8%train siblings (data_collector.py, eval_full_trajectory.py)
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent))


from analysis.summarize import summarize_run
from common.rollout import episode_to_json, rollout_one_episode, write_jsonl

# Import agent and collector from 84.8%train code
from eval_full_trajectory import FullTrajectoryAgent
from data_collector import (
    CameraType,
    DataCollectionConfig,
    RandomizationConfig,
    VisualExpertDemoCollector,
)


@dataclass(frozen=True)
class Condition:
    name: str
    rand_cfg: RandomizationConfig
    camera_position_noise_std: float
    action_noise_std: float = 0.0
    cube_friction: Optional[float] = None
    max_steps: int = 250
    meta: Dict[str, Any] | None = None


def _fmt_float(x: float, ndigits: int = 3) -> str:
    s = f"{float(x):.{ndigits}f}"
    # Keep filenames stable and sortable
    return s.replace("-", "m")


def _make_ranges_with_factor(
    *,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    factor: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    cx = (x_range[0] + x_range[1]) / 2.0
    cy = (y_range[0] + y_range[1]) / 2.0
    hx = (x_range[1] - x_range[0]) / 2.0
    hy = (y_range[1] - y_range[0]) / 2.0
    hx *= float(factor)
    hy *= float(factor)
    return (cx - hx, cx + hx), (cy - hy, cy + hy)


def _make_data_cfg(agent: FullTrajectoryAgent, camera_noise_std: float) -> DataCollectionConfig:
    cam_enums = [CameraType[c.upper()] for c in agent.config["camera_names"]]
    return DataCollectionConfig(
        image_width=112,
        image_height=112,
        use_depth=agent.config["use_depth"],
        camera_types=tuple(cam_enums),
        include_object_relative_pos=False,
        camera_position_noise_std=float(camera_noise_std),
    )


def _worker(
    worker_id: int,
    checkpoint: str,
    seeds: list[int],
    condition: Condition,
    device: str,
) -> list[dict]:
    # 打印worker开始的日志
    print(f"[Worker-{worker_id}] 开始处理 condition: {condition.name}, seeds: {seeds}")
    agent = FullTrajectoryAgent(checkpoint, device=device)
    data_cfg = _make_data_cfg(agent, condition.camera_position_noise_std)
    collector = VisualExpertDemoCollector(data_config=data_cfg, rand_config=condition.rand_cfg)

    try:
        rows: list[dict] = []
        for idx, seed in enumerate(seeds):
            # 打印每个seed开始执行的日志
            print(f"[Worker-{worker_id}] 开始执行 seed {seed}/{seeds[-1]} (condition: {condition.name})")
            start_seed = time.time()
            result = rollout_one_episode(
                agent=agent,
                collector=collector,
                seed=seed,
                max_steps=condition.max_steps,
                action_noise_std=condition.action_noise_std,
                cube_friction=condition.cube_friction,
                meta={
                    "condition": condition.name,
                    "camera_position_noise_std": condition.camera_position_noise_std,
                    "action_noise_std": condition.action_noise_std,
                    **(condition.meta or {}),
                },
            )
            # 打印每个seed执行完成的日志，包含耗时和成功状态
            seed_duration = time.time() - start_seed
            success = result.success
            print(f"[Worker-{worker_id}] 完成 seed {seed} (condition: {condition.name}) - 耗时: {seed_duration:.2f}s, 成功: {success}")
            rows.append(episode_to_json(result))
        
        # 打印worker完成的日志
        print(f"[Worker-{worker_id}] 完成处理 condition: {condition.name}, 共处理 {len(seeds)} 个seeds")
        return rows
    finally:
        collector.close()
        print(f"[Worker-{worker_id}] 已关闭collector (condition: {condition.name})")


def _split_seeds(seeds: list[int], workers: int) -> list[list[int]]:
    workers = max(1, min(int(workers), len(seeds)))
    buckets: list[list[int]] = [[] for _ in range(workers)]
    for i, s in enumerate(seeds):
        buckets[i % workers].append(s)
    return buckets


def _default_conditions() -> list[Condition]:
    # Baseline: match 84.8% eval_full_trajectory.py (including basket randomization)
    base_x = (0.35, 0.55)
    base_y = (-0.2, 0.2)
    base_scale = (0.03, 0.035)
    base_basket_noise = 0.08
    base_cam_noise = 0.01

    baseline_rand = RandomizationConfig(
        cube_pos_x_range=base_x,
        cube_pos_y_range=base_y,
        cube_scale_range=base_scale,
        basket_pos_x_noise=base_basket_noise,
    )

    conds: list[Condition] = []

    # 0) 基础成功率（单独保留一个 baseline 名称，便于课程报告引用）
    conds.append(
        Condition(
            name="baseline",
            rand_cfg=baseline_rand,
            camera_position_noise_std=base_cam_noise,
            meta={"group": "baseline", "level": 0.0, "desc": "与84.8%测评脚本一致的分布"},
        )
    )

    # 1) 篓子 X 随机化梯度
    for v in [0.00, 0.04, 0.08, 0.12, 0.16]:
        conds.append(
            Condition(
                name=f"basket_xnoise_{_fmt_float(v, 2)}",
                rand_cfg=RandomizationConfig(
                    cube_pos_x_range=base_x,
                    cube_pos_y_range=base_y,
                    cube_scale_range=base_scale,
                    basket_pos_x_noise=float(v),
                ),
                camera_position_noise_std=base_cam_noise,
                meta={"group": "basket_xnoise", "level": float(v), "unit": "m"},
            )
        )

    # 2) 相机位姿噪声梯度（轻微视觉 domain shift）
    for v in [0.00, 0.01, 0.02, 0.03, 0.05]:
        conds.append(
            Condition(
                name=f"cam_pos_noise_{_fmt_float(v, 2)}",
                rand_cfg=baseline_rand,
                camera_position_noise_std=float(v),
                meta={"group": "cam_pos_noise", "level": float(v), "unit": "m"},
            )
        )

    # 3) 动作噪声梯度（控制噪声鲁棒性）
    for v in [0.000, 0.001, 0.002, 0.003, 0.005]:
        conds.append(
            Condition(
                name=f"action_noise_{_fmt_float(v, 3)}",
                rand_cfg=baseline_rand,
                camera_position_noise_std=base_cam_noise,
                action_noise_std=float(v),
                meta={"group": "action_noise", "level": float(v)},
            )
        )

    # 4) 方块摩擦梯度（物理泛化）
    for v in [0.50, 0.75, 1.00, 1.50, 2.00]:
        conds.append(
            Condition(
                name=f"cube_friction_{_fmt_float(v, 2)}",
                rand_cfg=baseline_rand,
                camera_position_noise_std=base_cam_noise,
                cube_friction=float(v),
                meta={"group": "cube_friction", "level": float(v)},
            )
        )

    # 5) 方块初始分布范围梯度（扩大/缩小）
    for k in [1.00, 1.25, 1.50, 1.75, 2.00]:
        xr, yr = _make_ranges_with_factor(x_range=base_x, y_range=base_y, factor=float(k))
        conds.append(
            Condition(
                name=f"cube_range_k{_fmt_float(k, 2)}",
                rand_cfg=RandomizationConfig(
                    cube_pos_x_range=xr,
                    cube_pos_y_range=yr,
                    cube_scale_range=base_scale,
                    basket_pos_x_noise=base_basket_noise,
                ),
                camera_position_noise_std=base_cam_noise,
                meta={"group": "cube_range", "level": float(k), "desc": "扩大初始分布范围倍数"},
            )
        )

    # 6) 抓夹末端绝对坐标位置随机化
    for v in [0.00, 0.01, 0.02, 0.03, 0.05]:
        conds.append(
            Condition(
                name=f"ee_pos_noise_{_fmt_float(v, 2)}",
                rand_cfg=RandomizationConfig(
                    cube_pos_x_range=base_x,
                    cube_pos_y_range=base_y,
                    cube_scale_range=base_scale,
                    basket_pos_x_noise=base_basket_noise,
                    ee_pos_noise=float(v),
                ),
                camera_position_noise_std=base_cam_noise,
                meta={"group": "ee_pos_noise", "level": float(v), "unit": "m"},
            )
        )

    return conds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to 84.8% vision policy checkpoint (best_policy.pth)",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, generate success-rate curve plot after summary.csv",
    )
    parser.add_argument(
        "--baseline_line",
        type=float,
        default=None,
        help="Optional baseline success-rate (0-1) to draw as horizontal line on plots",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional run id (default: run_YYYYmmdd_HHMMSS)",
    )
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("run_%Y%m%d_%H%M%S")
    out_root = THIS_DIR / "runs" / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    episodes = int(args.episodes)
    seeds = [int(args.seed) + i for i in range(episodes)]

    conditions = _default_conditions()

    # 打印总任务信息
    print(f"\n[GenEvalVision] 开始执行评估任务")
    print(f"[GenEvalVision] 总共 {len(conditions)} 个conditions，每个condition执行 {episodes} 个episodes")
    print(f"[GenEvalVision] checkpoint: {args.checkpoint}")
    print(f"[GenEvalVision] episodes: {episodes}, workers: {args.workers}, device: {args.device}")
    print(f"[GenEvalVision] output: {out_root}\n")

    # Multiprocessing: only inside condition to keep output deterministic per-condition
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    # 遍历每个condition执行
    for cond_idx, cond in enumerate(conditions):
        # 打印condition开始的日志
        cond_start_time = time.time()
        print(f"\n" + "="*80)
        print(f"[GenEvalVision] 开始处理第 {cond_idx+1}/{len(conditions)} 个condition: {cond.name}")
        print(f"[GenEvalVision] Condition详情: 相机噪声={cond.camera_position_noise_std}, 动作噪声={cond.action_noise_std}, 方块摩擦={cond.cube_friction}")
        print("="*80 + "\n")

        cond_dir = out_root / "conditions"
        cond_dir.mkdir(parents=True, exist_ok=True)
        out_jsonl = cond_dir / f"{cond.name}.jsonl"

        buckets = _split_seeds(seeds, int(args.workers))
        pool_args = [
            (i, args.checkpoint, buckets[i], cond, args.device) for i in range(len(buckets))
        ]

        start = time.time()
        with mp.Pool(processes=len(buckets)) as pool:
            results = pool.starmap(_worker, pool_args)

        flat: list[dict] = []
        for r in results:
            flat.extend(r)

        write_jsonl(out_jsonl, flat)

        # 计算并打印condition的详细统计信息
        cond_duration = time.time() - cond_start_time
        succ = sum(1 for r in flat if r.get("success"))
        success_rate = succ / len(flat) * 100
        
        # 打印condition完成的详细日志
        print(f"\n" + "-"*80)
        print(f"[GenEvalVision] 完成处理condition: {cond.name}")
        print(f"[GenEvalVision] 统计结果: 成功 {succ}/{len(flat)} = {success_rate:.2f}%")
        print(f"[GenEvalVision] 总耗时: {cond_duration:.2f}s (平均每个episode: {cond_duration/len(flat):.2f}s)")
        print(f"[GenEvalVision] 结果已保存至: {out_jsonl}")
        print("-"*80 + "\n")

    # 打印汇总信息
    csv_path = summarize_run(out_root)
    print(f"\n[GenEvalVision] 所有conditions执行完成！")
    print(f"[GenEvalVision] 汇总文件已生成: {csv_path}")

    if args.plot:
        try:
            from analysis.plot_success_curves import plot_from_summary  # type: ignore

            out_png = plot_from_summary(out_root, baseline_line=args.baseline_line)
            print(f"[GenEvalVision] 成功率曲线图已生成: {out_png}")
        except Exception as e:
            print(f"[GenEvalVision] 绘制图表失败: {e}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())