from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pybullet as p
import torch


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    success: bool
    steps: int
    max_steps: int
    final_distance_xy: float
    final_cube_pos: Tuple[float, float, float]
    target_pos: Tuple[float, float, float]
    meta: Dict[str, Any]


def _distance_xy(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def _is_open_gripper(collector) -> bool:
    j_state = p.getJointState(
        collector.panda_id,
        collector.gripper_indices[0],
        physicsClientId=collector.client_id,
    )[0]
    return bool(j_state > 0.03)


def _is_cube_in_basket_xy(collector, cube_pos: Tuple[float, float, float]) -> bool:
    target = np.asarray(collector.target_pos, dtype=np.float32)
    inner = collector.data_config.basket_config.inner_size
    return (
        abs(cube_pos[0] - target[0]) < inner[0] / 2 * 0.9
        and abs(cube_pos[1] - target[1]) < inner[1] / 2 * 0.9
    )


def _is_cube_in_basket_z(collector, cube_pos: Tuple[float, float, float]) -> bool:
    inner = collector.data_config.basket_config.inner_size
    return bool(collector.table_height - inner[2] < cube_pos[2] < collector.table_height)


def rollout_one_episode(
    *,
    agent,
    collector,
    seed: int,
    max_steps: int,
    action_noise_std: float = 0.0,
    cube_friction: Optional[float] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> EpisodeResult:
    """Roll out one evaluation episode and return metrics.

    Notes:
    - Uses the same success criterion as 84.8% eval: cube in basket (xy+z) and gripper open,
      sustained for 10 consecutive checks.
    - Optional action_noise_std perturbs policy delta output.
    - Optional cube_friction overrides cube lateralFriction after scene reset.
    """
    meta = dict(meta or {})

    agent.reset()
    collector.setup_scene(seed=seed)

    if cube_friction is not None:
        p.changeDynamics(
            collector.cube_id,
            -1,
            lateralFriction=float(cube_friction),
            physicsClientId=collector.client_id,
        )
        meta["cube_friction"] = float(cube_friction)

    consecutive_success = 0
    success = False
    steps_used = 0

    for step in range(int(max_steps)):
        obs = collector.get_obs()
        delta, grip, phase_idx = agent.predict(obs)

        # keep consistent with baseline eval
        delta = np.clip(delta, -0.05, 0.05)

        if action_noise_std > 0:
            delta = delta + np.random.randn(3).astype(np.float32) * float(action_noise_std)
            delta = np.clip(delta, -0.05, 0.05)

        from data_collector import Action  # type: ignore

        action = Action(delta, grip)

        collector.execute_action(action, steps=10)
        steps_used = step + 1

        cube_pos = p.getBasePositionAndOrientation(
            collector.cube_id, physicsClientId=collector.client_id
        )[0]

        if (
            _is_cube_in_basket_xy(collector, cube_pos)
            and _is_cube_in_basket_z(collector, cube_pos)
            and _is_open_gripper(collector)
        ):
            consecutive_success += 1
        else:
            consecutive_success = 0

        if consecutive_success >= 10:
            success = True
            break

    cube_pos = p.getBasePositionAndOrientation(
        collector.cube_id, physicsClientId=collector.client_id
    )[0]
    target = np.asarray(collector.target_pos, dtype=np.float32)
    cube_np = np.asarray(cube_pos, dtype=np.float32)

    return EpisodeResult(
        seed=int(seed),
        success=bool(success),
        steps=int(steps_used),
        max_steps=int(max_steps),
        final_distance_xy=_distance_xy(cube_np, target),
        final_cube_pos=(float(cube_pos[0]), float(cube_pos[1]), float(cube_pos[2])),
        target_pos=(float(target[0]), float(target[1]), float(target[2])),
        meta=meta,
    )


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def episode_to_json(result: EpisodeResult) -> dict:
    return {
        "seed": result.seed,
        "success": result.success,
        "steps": result.steps,
        "max_steps": result.max_steps,
        "final_distance_xy": result.final_distance_xy,
        "final_cube_pos": list(result.final_cube_pos),
        "target_pos": list(result.target_pos),
        "meta": result.meta,
    }
