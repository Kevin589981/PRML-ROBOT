from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class EpisodeMetrics:
    success: bool
    setup_ok: bool
    steps: int
    distance_xy: float
    final_pos: Optional[list[float]]
    reason: str


def compute_distance_xy(pos: np.ndarray, target_pos: np.ndarray) -> float:
    return float(np.linalg.norm(pos[:2] - target_pos[:2]))


def compute_success(
    final_pos: np.ndarray,
    target_pos: np.ndarray,
    target_surface_height: float,
    obj_half_height: float,
    distance_threshold: float = 0.06,
) -> tuple[bool, float]:
    distance_xy = compute_distance_xy(final_pos, target_pos)

    # 以“接触平面高度”为基准，检查物体是否确实落在该平面附近
    expected_center_z = target_surface_height + obj_half_height
    on_surface = (expected_center_z - 0.03) < final_pos[2] < (expected_center_z + 0.12)

    return bool(distance_xy < distance_threshold and on_surface), distance_xy
