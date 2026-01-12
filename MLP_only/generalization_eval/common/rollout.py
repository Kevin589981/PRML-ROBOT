from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from collections import deque

import numpy as np
import pybullet as p
import torch

from data_collector import Action, ExpertDemoCollector

from .metrics import compute_success


@dataclass(frozen=True)
class RolloutResult:
    setup_ok: bool
    success: bool
    steps: int
    distance_xy: float
    final_pos: Optional[list[float]]
    release_step: int
    reason: str


def _infer_obj_half_height(collector: ExpertDemoCollector) -> float:
    cube_pos, _ = p.getBasePositionAndOrientation(collector.cube_id)
    surface_z = float(getattr(collector, "current_cube_surface_height", collector.table_height))
    return max(0.0, float(cube_pos[2]) - surface_z)


def check_object_settled_any_surface(
    collector: ExpertDemoCollector,
    obj_half_height: float,
    velocity_threshold: float = 0.05,
) -> bool:
    cube_pos, _ = p.getBasePositionAndOrientation(collector.cube_id)
    cube_vel, _ = p.getBaseVelocity(collector.cube_id)

    linear_vel = float(np.linalg.norm(cube_vel))
    if linear_vel > velocity_threshold:
        return False

    # 允许在桌面或目标平台附近稳定
    surfaces = [collector.table_height, float(getattr(collector, "current_target_surface_height", collector.table_height))]
    for surface in surfaces:
        expected_center_z = surface + obj_half_height
        if (expected_center_z - 0.03) < cube_pos[2] < (expected_center_z + 0.12):
            return True
    return False


def wait_for_object_settle(
    collector: ExpertDemoCollector,
    obj_half_height: float,
    max_wait_steps: int = 120,
    render: bool = False,
) -> bool:
    for _ in range(max_wait_steps):
        p.stepSimulation()
        if render:
            time.sleep(1.0 / 240.0)
        if check_object_settled_any_surface(collector, obj_half_height=obj_half_height):
            return True
    return check_object_settled_any_surface(collector, obj_half_height=obj_half_height)


def run_policy_episode(
    collector: ExpertDemoCollector,
    model: torch.nn.Module,
    stats: Dict[str, Any],
    max_steps: int = 500,
    steps_per_action: int = 5,
    render: bool = False,
    distance_threshold: float = 0.06,
    obj_half_height: Optional[float] = None,
    # ===== 观测侧（控制变量泛化）=====
    obs_noise_std_norm: float = 0.0,
    obs_drop_prob: float = 0.0,
    obs_delay_steps: int = 0,
    # ===== 动作侧（控制变量泛化）=====
    action_noise_std: float = 0.0,
    action_delay_steps: int = 0,
    # ===== 外部扰动 =====
    push_force: Optional[list[float]] = None,
    push_step: int = -1,
    push_duration_steps: int = 0,
) -> RolloutResult:
    if collector.cube_id is None or collector.target_pos is None:
        return RolloutResult(
            setup_ok=False,
            success=False,
            steps=0,
            distance_xy=float("inf"),
            final_pos=None,
            release_step=-1,
            reason="setup_not_done",
        )

    obs = collector.get_observation()

    if obj_half_height is None:
        obj_half_height = _infer_obj_half_height(collector)

    obs_delay_steps = max(0, int(obs_delay_steps))
    action_delay_steps = max(0, int(action_delay_steps))

    obs_queue = deque(maxlen=max(1, obs_delay_steps + 1))
    action_queue = deque(maxlen=max(1, action_delay_steps + 1))

    last_obs_vec = obs.to_vector().copy()

    gripper_was_closed = False
    gripper_opened_after_close = False
    release_step = -1

    for step in range(max_steps):
        obs_vec = obs.to_vector()

        # 观测丢帧：用上一帧替代
        if obs_drop_prob and (np.random.random() < float(obs_drop_prob)):
            obs_vec = last_obs_vec.copy()
        else:
            last_obs_vec = obs_vec.copy()

        obs_norm = (obs_vec - stats["obs_mean"]) / stats["obs_std"]

        # 观测延迟：把归一化后的观测放入队列，取延迟后的值
        obs_queue.append(obs_norm)
        if obs_delay_steps > 0 and len(obs_queue) < (obs_delay_steps + 1):
            obs_norm_delayed = obs_queue[0]
        else:
            obs_norm_delayed = obs_queue[0] if obs_delay_steps > 0 else obs_norm

        # 观测噪声（在归一化空间加噪）
        if obs_noise_std_norm and float(obs_noise_std_norm) > 0:
            obs_norm_delayed = obs_norm_delayed + np.random.randn(*obs_norm_delayed.shape) * float(obs_noise_std_norm)

        with torch.no_grad():
            pred_act_norm = model(torch.FloatTensor(obs_norm_delayed).unsqueeze(0)).numpy()[0]

        pred_act_real = pred_act_norm * stats["action_std"] + stats["action_mean"]

        delta_pos = np.clip(pred_act_real[:3], -0.03, 0.03)
        gripper_raw = float(pred_act_real[3])
        gripper_action = 1.0 if gripper_raw > 0.5 else 0.0

        # 动作噪声：在 delta_pos 上加高斯噪声（米）
        if action_noise_std and float(action_noise_std) > 0:
            delta_pos = delta_pos + np.random.randn(3) * float(action_noise_std)

        # 动作延迟：把预测动作放入队列，执行延迟后的动作
        action_queue.append((delta_pos.copy(), float(gripper_action)))
        if action_delay_steps > 0 and len(action_queue) < (action_delay_steps + 1):
            exec_delta_pos, exec_gripper_action = action_queue[0]
        else:
            exec_delta_pos, exec_gripper_action = action_queue[0] if action_delay_steps > 0 else (delta_pos, gripper_action)

        if gripper_action < 0.5:
            gripper_was_closed = True
        elif gripper_was_closed and gripper_action > 0.5:
            if not gripper_opened_after_close:
                gripper_opened_after_close = True
                release_step = step

        # 外部扰动：在抓取后、释放前施加侧向推力
        if push_force is not None and push_step >= 0 and push_duration_steps > 0:
            if step >= int(push_step) and step < int(push_step) + int(push_duration_steps):
                if gripper_was_closed and (not gripper_opened_after_close):
                    try:
                        p.applyExternalForce(
                            collector.cube_id,
                            -1,
                            forceObj=list(push_force),
                            posObj=[0, 0, 0],
                            flags=p.LINK_FRAME,
                        )
                    except Exception:
                        pass

        action = Action(delta_position=np.array(exec_delta_pos), gripper_action=float(exec_gripper_action))
        obs = collector.execute_action(action, steps=steps_per_action)

        if render:
            time.sleep(0.01)

        if gripper_opened_after_close and step > release_step + 10:
            # 等待稳定
            if wait_for_object_settle(collector, obj_half_height=float(obj_half_height), max_wait_steps=60, render=render):
                cube_pos, _ = p.getBasePositionAndOrientation(collector.cube_id)
                cube_pos = np.array(cube_pos)

                target_surface = float(getattr(collector, "current_target_surface_height", collector.table_height))
                success, distance_xy = compute_success(
                    final_pos=cube_pos,
                    target_pos=np.array(collector.target_pos),
                    target_surface_height=target_surface,
                    obj_half_height=float(obj_half_height),
                    distance_threshold=distance_threshold,
                )

                return RolloutResult(
                    setup_ok=True,
                    success=success,
                    steps=step,
                    distance_xy=distance_xy,
                    final_pos=cube_pos.tolist(),
                    release_step=release_step,
                    reason="ok" if success else "placed_far_or_wrong_height",
                )

    return RolloutResult(
        setup_ok=True,
        success=False,
        steps=max_steps,
        distance_xy=float("inf"),
        final_pos=None,
        release_step=release_step,
        reason="timeout",
    )
