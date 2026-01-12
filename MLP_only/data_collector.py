import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import pickle
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from enum import Enum


# ==================== 数据结构 ====================

@dataclass
class Observation:
    """观测空间 - 32维"""
    joint_positions: np.ndarray      # (7,)
    joint_velocities: np.ndarray     # (7,)
    ee_position: np.ndarray          # (3,)
    ee_orientation: np.ndarray       # (4,)
    gripper_state: float             # (1,)
    object_rel_position: np.ndarray  # (3,)
    object_rel_orientation: np.ndarray # (4,) 
    target_rel_position: np.ndarray  # (3,) 
    
    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.joint_positions, 
            self.joint_velocities,
            self.ee_position, 
            self.ee_orientation,
            [self.gripper_state],
            self.object_rel_position,
            self.object_rel_orientation, # 向量里也要加上
            self.target_rel_position     # 加上目标位置
        ])
    
    @staticmethod
    def vector_dim() -> int:
        return 32  # 7+7+3+4+1+3+4+3 = 32


@dataclass  
class Action:
    """动作空间 - 4维"""
    delta_position: np.ndarray  # (3,)
    gripper_action: float       # 0=闭合, 1=张开
    
    def to_vector(self) -> np.ndarray:
        return np.concatenate([self.delta_position, [self.gripper_action]])
    
    @staticmethod
    def vector_dim() -> int:
        return 4


@dataclass
class Trajectory:
    """完整轨迹"""
    observations: np.ndarray
    actions: np.ndarray
    success: bool = False
    task_config: Dict = field(default_factory=dict)
    length: int = 0
    
    def __post_init__(self):
        self.length = len(self.observations)


# ==================== 随机化配置 ====================

@dataclass
class RandomizationConfig:
    """
    保守的随机化参数配置
    
    坐标系说明（相对于机械臂基座）：
    - X轴: 机械臂前方为正
    - Y轴: 机械臂左侧为正  
    - Z轴: 向上为正
    
    Franka Panda有效工作范围（保守估计）：
    - 前向距离: 0.3m - 0.6m
    - 侧向距离: -0.3m - 0.3m
    - 高度: 桌面以上 0 - 0.4m
    """
    
    # === 物体位置随机化（相对于基座前方） ===
    # 保守范围，确保机械臂能够到达
    cube_pos_x_range: Tuple[float, float] = (0.35, 0.50)   # 前方距离
    cube_pos_y_range: Tuple[float, float] = (-0.15, 0.15)  # 左右偏移
    cube_rotation_z_range: Tuple[float, float] = (-math.pi/4, math.pi/4)  # 限制旋转
    cube_scale_range: Tuple[float, float] = (0.045, 0.055)
    
    # === 目标位置随机化 ===
    target_pos_x_range: Tuple[float, float] = (0.35, 0.55)
    target_pos_y_range: Tuple[float, float] = (-0.20, 0.20)
    min_cube_target_distance: float = 0.12  # 物体和目标最小距离
    max_cube_target_distance: float = 0.35  # 最大距离，避免轨迹过长
    
    # === 机械臂初始状态随机化（小幅度） ===
    arm_joint_noise_std: float = 0.03  # 减小关节噪声
    
    # === 运动噪声/抖动（适度） ===
    action_gaussian_noise_std: float = 0.002
    action_noise_prob: float = 0.3
    
    # OU过程噪声
    ou_theta: float = 0.15
    ou_sigma: float = 0.003
    
    # 偶发扰动（降低概率）
    large_perturbation_prob: float = 0.01
    large_perturbation_std: float = 0.008
    
    # === 轨迹参数变化 ===
    approach_height_range: Tuple[float, float] = (0.10, 0.15)
    grasp_height_offset_range: Tuple[float, float] = (-0.002, 0.005)
    place_height_offset_range: Tuple[float, float] = (0.03, 0.06)
    
    # === 控制参数 ===
    max_delta_range: Tuple[float, float] = (0.015, 0.022)
    gripper_close_duration_range: Tuple[float, float] = (0.35, 0.50)
    gripper_open_duration_range: Tuple[float, float] = (0.35, 0.55)
    
    # === 物理参数 ===
    cube_mass_range: Tuple[float, float] = (0.08, 0.15)
    friction_range: Tuple[float, float] = (2.5, 3.5)


class NoiseType(Enum):
    NONE = 0
    GAUSSIAN = 1
    OU = 2
    COMBINED = 3


# ==================== 增强版专家示范收集器 ====================

class ExpertDemoCollector:
    """增强版专家示范收集器 - 安全的随机化范围"""
    
    def __init__(self, 
                 gui: bool = True, 
                 config: Optional[RandomizationConfig] = None,
                 noise_type: NoiseType = NoiseType.COMBINED):
        
        self.render = gui  # <--- 新增：记录是否需要渲染
        # 如果 gui=True 使用图形界面，否则使用 DIRECT (后台极速模式)
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0)
        self.dt = 1./240.
        
        self.rand_config = config if config else RandomizationConfig()
        self.noise_type = noise_type
        
        # 机械臂参数
        self.ee_index = 11
        self.gripper_indices = [9, 10]
        self.arm_joint_indices = list(range(7))
        self.gripper_orn_down = p.getQuaternionFromEuler([math.pi, 0, 0])
        
        # 机械臂基座位置（世界坐标）
        self.default_panda_base_pos = np.array([-0.5, 0, 0.625], dtype=float)
        self.panda_base_pos = self.default_panda_base_pos.copy()
        self.table_height = 0.625
        
        # 当前episode参数
        self.current_max_delta = 0.02
        self.current_approach_height = 0.12
        self.current_grasp_offset = 0.0
        self.current_place_offset = 0.05
        
        # OU噪声状态
        self.ou_state = np.zeros(3)
        
        # 数据存储
        self.trajectories: List[Trajectory] = []
        
        # 场景对象
        self.panda_id = None
        self.cube_id = None
        self.table_id = None
        self.cube_support_id = None
        self.target_support_id = None
        self.target_pos = None

        # 评估/泛化扩展：记录当前接触平面高度（用于稳定性与成功判定）
        self.current_cube_surface_height = self.table_height
        self.current_target_surface_height = self.table_height

    def _create_support_platform(self,
                                 xy_pos_world: np.ndarray,
                                 height: float,
                                 half_extents_xy: Tuple[float, float] = (0.06, 0.06),
                                 rgba: Tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0)) -> Optional[int]:
        """在桌面上创建一个垫高平台（固定不动）。

        Args:
            xy_pos_world: 平台中心的 x,y（世界坐标），z 由 table_height 与 height 决定
            height: 平台总高度（米）
        """
        if height is None or height <= 1e-9:
            return None

        half_height = height * 0.5
        # 平台中心放在桌面之上 half_height 处
        base_pos = [float(xy_pos_world[0]), float(xy_pos_world[1]), float(self.table_height + half_height)]
        half_extents = [float(half_extents_xy[0]), float(half_extents_xy[1]), float(half_height)]

        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=list(rgba))
        body_id = p.createMultiBody(baseMass=0.0,
                                    baseCollisionShapeIndex=col,
                                    baseVisualShapeIndex=vis,
                                    basePosition=base_pos,
                                    baseOrientation=[0, 0, 0, 1])
        return body_id

    def _create_object(self,
                       pos_world: np.ndarray,
                       orn_quat: List[float],
                       mass: float,
                       friction: float,
                       object_spec: Optional[Dict] = None,
                       cube_scale: float = 0.05) -> Tuple[int, float]:
        """创建被操作物体。

        Returns:
            (body_id, half_height)
        """
        # 默认：保持原逻辑，加载 cube.urdf
        if object_spec is None:
            body_id = p.loadURDF(
                "cube.urdf",
                pos_world.tolist(),
                orn_quat,
                globalScaling=cube_scale
            )
            half_height = float(cube_scale) * 0.5
            return body_id, half_height

        shape = str(object_spec.get('shape', 'box')).lower()
        rgba = object_spec.get('rgba', [0.8, 0.2, 0.2, 1.0])

        if shape == 'sphere':
            radius = float(object_spec.get('radius', 0.025))
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
            body_id = p.createMultiBody(baseMass=float(mass),
                                        baseCollisionShapeIndex=col,
                                        baseVisualShapeIndex=vis,
                                        basePosition=pos_world.tolist(),
                                        baseOrientation=orn_quat)
            return body_id, radius

        if shape == 'cylinder':
            radius = float(object_spec.get('radius', 0.025))
            height = float(object_spec.get('height', 0.05))
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=rgba)
            body_id = p.createMultiBody(baseMass=float(mass),
                                        baseCollisionShapeIndex=col,
                                        baseVisualShapeIndex=vis,
                                        basePosition=pos_world.tolist(),
                                        baseOrientation=orn_quat)
            return body_id, height * 0.5

        if shape == 'mesh':
            mesh_path = object_spec.get('mesh_path', None)
            mesh_scale = object_spec.get('mesh_scale', [1.0, 1.0, 1.0])
            if mesh_path is None:
                raise ValueError('object_spec.shape=mesh 但未提供 mesh_path')
            col = p.createCollisionShape(p.GEOM_MESH, fileName=str(mesh_path), meshScale=mesh_scale)
            vis = p.createVisualShape(p.GEOM_MESH, fileName=str(mesh_path), meshScale=mesh_scale, rgbaColor=rgba)
            body_id = p.createMultiBody(baseMass=float(mass),
                                        baseCollisionShapeIndex=col,
                                        baseVisualShapeIndex=vis,
                                        basePosition=pos_world.tolist(),
                                        baseOrientation=orn_quat)
            # mesh 的 half_height 难以精确；用用户提供的估计值
            half_height = float(object_spec.get('half_height', 0.03))
            return body_id, half_height

        # 默认 box
        half_extents = object_spec.get('half_extents', [0.025, 0.025, 0.025])
        hx, hy, hz = [float(x) for x in half_extents]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=rgba)
        body_id = p.createMultiBody(baseMass=float(mass),
                                    baseCollisionShapeIndex=col,
                                    baseVisualShapeIndex=vis,
                                    basePosition=pos_world.tolist(),
                                    baseOrientation=orn_quat)
        return body_id, hz

    def _scale_object_spec(self, object_spec: Dict, scale: float) -> Dict:
        """对 object_spec 做统一缩放（不修改入参对象）。"""
        if object_spec is None:
            return None
        if scale is None or abs(float(scale) - 1.0) < 1e-9:
            return dict(object_spec)

        out = dict(object_spec)
        shape = str(out.get('shape', 'box')).lower()

        if shape == 'sphere':
            out['radius'] = float(out.get('radius', 0.025)) * float(scale)
            return out

        if shape == 'cylinder':
            out['radius'] = float(out.get('radius', 0.025)) * float(scale)
            out['height'] = float(out.get('height', 0.05)) * float(scale)
            return out

        if shape == 'mesh':
            mesh_scale = out.get('mesh_scale', [1.0, 1.0, 1.0])
            out['mesh_scale'] = [float(x) * float(scale) for x in mesh_scale]
            if 'half_height' in out:
                out['half_height'] = float(out['half_height']) * float(scale)
            return out

        # box
        he = out.get('half_extents', [0.025, 0.025, 0.025])
        out['half_extents'] = [float(he[0]) * float(scale), float(he[1]) * float(scale), float(he[2]) * float(scale)]
        return out

    def _create_obstacle_box(self,
                             center_world: np.ndarray,
                             half_extents: Tuple[float, float, float],
                             rgba: Tuple[float, float, float, float] = (0.2, 0.2, 0.2, 1.0),
                             euler: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> int:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(half_extents))
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=list(half_extents), rgbaColor=list(rgba))
        body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[float(center_world[0]), float(center_world[1]), float(center_world[2])],
            baseOrientation=p.getQuaternionFromEuler(list(euler)),
        )
        return body_id
        
    def _base_to_world(self, pos_relative: np.ndarray) -> np.ndarray:
        """将相对于机械臂基座的坐标转换为世界坐标"""
        return self.panda_base_pos + pos_relative
    
    def _world_to_base(self, pos_world: np.ndarray) -> np.ndarray:
        """将世界坐标转换为相对于机械臂基座的坐标"""
        return pos_world - self.panda_base_pos
        
    def _reset_noise_state(self):
        """重置噪声状态"""
        self.ou_state = np.zeros(3)
        
    def _get_ou_noise(self) -> np.ndarray:
        """Ornstein-Uhlenbeck过程噪声"""
        cfg = self.rand_config
        self.ou_state += cfg.ou_theta * (-self.ou_state) + \
                         cfg.ou_sigma * np.random.randn(3)
        return self.ou_state.copy()
    
    def _get_action_noise(self) -> np.ndarray:
        """获取动作噪声"""
        cfg = self.rand_config
        noise = np.zeros(3)
        
        if self.noise_type == NoiseType.NONE:
            return noise
            
        if np.random.random() > cfg.action_noise_prob:
            return noise
            
        if self.noise_type == NoiseType.GAUSSIAN:
            noise = np.random.randn(3) * cfg.action_gaussian_noise_std
        elif self.noise_type == NoiseType.OU:
            noise = self._get_ou_noise()
        elif self.noise_type == NoiseType.COMBINED:
            noise = np.random.randn(3) * cfg.action_gaussian_noise_std + self._get_ou_noise()
            
        if np.random.random() < cfg.large_perturbation_prob:
            noise += np.random.randn(3) * cfg.large_perturbation_std
            
        return noise
    
    def _randomize_episode_params(self):
        """随机化当前episode的控制参数"""
        cfg = self.rand_config
        self.current_max_delta = np.random.uniform(*cfg.max_delta_range)
        self.current_approach_height = np.random.uniform(*cfg.approach_height_range)
        self.current_grasp_offset = np.random.uniform(*cfg.grasp_height_offset_range)
        self.current_place_offset = np.random.uniform(*cfg.place_height_offset_range)
    
    def _check_reachability(self, pos_world: np.ndarray, verbose: bool = False) -> bool:
        """
        检查位置是否在机械臂可达范围内
        
        使用简单的几何检查 + IK验证
        """
        pos_rel = self._world_to_base(pos_world)
        
        # 1. 基本几何检查
        horizontal_dist = np.sqrt(pos_rel[0]**2 + pos_rel[1]**2)
        
        # Panda臂展约0.855m，但考虑到关节限制，有效范围更小
        if horizontal_dist < 0.25 or horizontal_dist > 0.65:
            if verbose:
                print(f"    位置超出水平范围: {horizontal_dist:.3f}m")
            return False
            
        # 高度检查（相对于基座）
        height_rel = pos_world[2] - self.panda_base_pos[2]
        if height_rel < -0.1 or height_rel > 0.5:
            if verbose:
                print(f"    位置超出高度范围: {height_rel:.3f}m")
            return False
        
        # 2. IK验证（如果机械臂已加载）
        if self.panda_id is not None:
            try:
                joint_poses = p.calculateInverseKinematics(
                    self.panda_id,
                    self.ee_index,
                    pos_world.tolist(),
                    targetOrientation=self.gripper_orn_down,
                    maxNumIterations=100,
                    residualThreshold=1e-4
                )
                
                # 检查关节是否在限制内
                joint_limits = [
                    (-2.8973, 2.8973),
                    (-1.7628, 1.7628),
                    (-2.8973, 2.8973),
                    (-3.0718, -0.0698),
                    (-2.8973, 2.8973),
                    (-0.0175, 3.7525),
                    (-2.8973, 2.8973)
                ]
                
                for i, (pos, (lo, hi)) in enumerate(zip(joint_poses[:7], joint_limits)):
                    if pos < lo or pos > hi:
                        if verbose:
                            print(f"    关节{i}超限: {pos:.3f} not in [{lo:.3f}, {hi:.3f}]")
                        return False
                        
            except Exception as e:
                if verbose:
                    print(f"    IK求解失败: {e}")
                return False
                
        return True
    
    def _generate_valid_cube_position(self, max_attempts: int = 50) -> Optional[np.ndarray]:
        """生成有效的物体位置（确保可达）"""
        cfg = self.rand_config
        
        for _ in range(max_attempts):
            # 生成相对于基座的位置
            pos_rel = np.array([
                np.random.uniform(*cfg.cube_pos_x_range),
                np.random.uniform(*cfg.cube_pos_y_range),
                self.table_height - self.panda_base_pos[2] + 0.025  # 桌面上
            ])
            
            pos_world = self._base_to_world(pos_rel)
            
            # 检查抓取位置和上方位置都可达
            grasp_pos = pos_world.copy()
            above_pos = pos_world + np.array([0, 0, 0.15])
            
            if self._check_reachability(grasp_pos) and self._check_reachability(above_pos):
                return pos_world
                
        return None
    
    def _generate_valid_target_position(self, 
                                         cube_pos: np.ndarray,
                                         max_attempts: int = 50) -> Optional[np.ndarray]:
        """生成有效的目标位置（确保可达且与物体有合适距离）"""
        cfg = self.rand_config
        
        for _ in range(max_attempts):
            pos_rel = np.array([
                np.random.uniform(*cfg.target_pos_x_range),
                np.random.uniform(*cfg.target_pos_y_range),
                self.table_height - self.panda_base_pos[2] + 0.025
            ])
            
            pos_world = self._base_to_world(pos_rel)
            
            # 检查与物体的距离
            dist = np.linalg.norm(pos_world[:2] - cube_pos[:2])
            if dist < cfg.min_cube_target_distance or dist > cfg.max_cube_target_distance:
                continue
                
            # 检查可达性
            place_pos = pos_world + np.array([0, 0, 0.05])
            above_pos = pos_world + np.array([0, 0, 0.15])
            
            if self._check_reachability(place_pos) and self._check_reachability(above_pos):
                return pos_world
                
        return None
        
    def setup_scene(self, 
                    cube_pos: Optional[List] = None,
                    target_pos: Optional[List] = None,
                    randomize: bool = True,
                    seed: Optional[int] = None,
                    object_spec: Optional[Dict] = None,
                    cube_support_height: float = 0.0,
                    target_support_height: float = 0.0,
                    panda_base_pos: Optional[List[float]] = None,
                    panda_base_offset: Optional[List[float]] = None,
                    cube_initial_euler: Optional[List[float]] = None,
                    cube_initial_euler_deg: Optional[List[float]] = None,
                    mass_scale: float = 1.0,
                    friction_override: Optional[float] = None,
                    restitution: Optional[float] = None,
                    gravity_scale: float = 1.0,
                    sim_time_step: Optional[float] = None,
                    solver_iterations: Optional[int] = None,
                    target_xy_offset: Optional[List[float]] = None,
                    object_scale: float = 1.0,
                    obstacle_specs: Optional[List[Dict]] = None) -> Optional[Dict]:
        """
        设置场景 - 带可达性验证
        
        Returns:
            配置字典，如果无法生成有效场景则返回None
        """
        if seed is not None:
            np.random.seed(seed)

        # === 基座位置（控制变量泛化）===
        # 每个 episode 都先回到默认值，避免上一次设置“残留”
        self.panda_base_pos = self.default_panda_base_pos.copy()
        base_offset = np.array([0.0, 0.0, 0.0], dtype=float)
        if panda_base_pos is not None:
            self.panda_base_pos = np.array(panda_base_pos, dtype=float)
            base_offset = self.panda_base_pos - self.default_panda_base_pos
        elif panda_base_offset is not None:
            base_offset = np.array(panda_base_offset, dtype=float)
            self.panda_base_pos = self.default_panda_base_pos + base_offset
            
        self._reset_noise_state()
        
        if randomize:
            self._randomize_episode_params()
        else:
            self.current_max_delta = 0.02
            self.current_approach_height = 0.12
            self.current_grasp_offset = 0.0
            self.current_place_offset = 0.05
            
        # 重置仿真
        p.resetSimulation()
        p.setGravity(0, 0, -9.8 * float(gravity_scale))

        # 物理引擎参数（控制变量泛化）
        if sim_time_step is not None:
            self.dt = float(sim_time_step)
            p.setTimeStep(self.dt)
        if solver_iterations is not None:
            p.setPhysicsEngineParameter(numSolverIterations=int(solver_iterations))
        
        # 地面和桌子
        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF(
            "table/table.urdf",
            [0, 0, 0],
            p.getQuaternionFromEuler([0, 0, 0])
        )
        
        # 机械臂
        self.panda_id = p.loadURDF(
            "franka_panda/panda.urdf",
            self.panda_base_pos.tolist(),
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True
        )
        
        # 先重置到home位置，便于IK检查
        self._reset_arm(randomize=False)
        for _ in range(10):
            p.stepSimulation()
        
        cfg = self.rand_config

        # 记录当前接触平面高度（默认为桌面）
        self.current_cube_surface_height = self.table_height + float(cube_support_height)
        self.current_target_surface_height = self.table_height + float(target_support_height)
        
        # === 确定物体位置 ===
        if cube_pos is not None:
            cube_pos = np.array(cube_pos)
            if not self._check_reachability(cube_pos):
                print("  警告: 指定的物体位置不可达")
                return None
        elif randomize:
            cube_pos = self._generate_valid_cube_position()
            if cube_pos is None:
                print("  警告: 无法生成有效的物体位置")
                return None
        else:
            # 默认位置（机械臂正前方）
            cube_pos = self._base_to_world(np.array([0.45, 0, 0.025]))
            
        # === 确定目标位置 ===
        if target_pos is not None:
            self.target_pos = np.array(target_pos)
            if not self._check_reachability(self.target_pos):
                print("  警告: 指定的目标位置不可达")
                return None
        elif randomize:
            self.target_pos = self._generate_valid_target_position(cube_pos)
            if self.target_pos is None:
                print("  警告: 无法生成有效的目标位置")
                return None
        else:
            self.target_pos = self._base_to_world(np.array([0.45, -0.20, 0.025]))

        # 目标点 XY 偏移（例如目标标注误差/目标偏移鲁棒性）
        if target_xy_offset is not None:
            off = np.array(target_xy_offset, dtype=float)
            if off.shape[0] >= 2:
                self.target_pos = np.array(self.target_pos, dtype=float)
                self.target_pos[0] += float(off[0])
                self.target_pos[1] += float(off[1])
        
        # === 创建物体 ===
        if randomize:
            yaw = np.random.uniform(*cfg.cube_rotation_z_range)
            cube_orn = p.getQuaternionFromEuler([0, 0, yaw])
            cube_scale = np.random.uniform(*cfg.cube_scale_range)
            mass = np.random.uniform(*cfg.cube_mass_range)
            friction = np.random.uniform(*cfg.friction_range)
        else:
            cube_orn = [0, 0, 0, 1]
            cube_scale = 0.05
            mass = 0.1
            friction = 3.0

        # 初始姿态（覆盖 yaw/pitch/roll）
        if cube_initial_euler_deg is not None:
            e = np.array(cube_initial_euler_deg, dtype=float)
            cube_orn = p.getQuaternionFromEuler([float(e[0]) * math.pi / 180.0, float(e[1]) * math.pi / 180.0, float(e[2]) * math.pi / 180.0])
        elif cube_initial_euler is not None:
            e = np.array(cube_initial_euler, dtype=float)
            cube_orn = p.getQuaternionFromEuler([float(e[0]), float(e[1]), float(e[2])])

        # 尺寸缩放（几何扰动：整体缩放）
        cube_scale = float(cube_scale) * float(object_scale)

        # 质量缩放
        mass = float(mass) * float(mass_scale)

        # 摩擦覆盖
        if friction_override is not None:
            friction = float(friction_override)

        # object_spec 也要跟随缩放
        if object_spec is not None:
            object_spec = self._scale_object_spec(object_spec, float(object_scale))

        # 如果传入 object_spec，则忽略 cube_scale（由 object_spec 决定尺寸）

        # --- 支撑平台（用于初始/目标海拔泛化）---
        self.cube_support_id = None
        self.target_support_id = None
        if cube_support_height and cube_support_height > 1e-9:
            self.cube_support_id = self._create_support_platform(
                xy_pos_world=np.array([cube_pos[0], cube_pos[1]]),
                height=float(cube_support_height),
                half_extents_xy=(0.07, 0.07),
                rgba=(0.5, 0.5, 0.8, 1.0)
            )

        if target_support_height and target_support_height > 1e-9:
            self.target_support_id = self._create_support_platform(
                xy_pos_world=np.array([self.target_pos[0], self.target_pos[1]]),
                height=float(target_support_height),
                half_extents_xy=(0.09, 0.09),
                rgba=(0.5, 0.8, 0.5, 1.0)
            )

        # --- 根据物体形状修正 z，使其落在当前表面上 ---
        # baseline: cube.urdf 的 half_height 由 cube_scale 决定
        default_half_height = float(cube_scale) * 0.5
        if object_spec is not None:
            # 用临时创建前的几何信息估算 half_height
            shape = str(object_spec.get('shape', 'box')).lower()
            if shape == 'sphere':
                default_half_height = float(object_spec.get('radius', default_half_height))
            elif shape == 'cylinder':
                default_half_height = float(object_spec.get('height', default_half_height * 2)) * 0.5
            elif shape == 'mesh':
                default_half_height = float(object_spec.get('half_height', default_half_height))
            else:
                he = object_spec.get('half_extents', [default_half_height, default_half_height, default_half_height])
                default_half_height = float(he[2])

        cube_pos = np.array(cube_pos, dtype=float)
        cube_pos[2] = self.current_cube_surface_height + default_half_height

        self.target_pos = np.array(self.target_pos, dtype=float)
        # 目标点 z 作为“希望放置后的物体中心高度”
        self.target_pos[2] = self.current_target_surface_height + default_half_height
            
        # 创建物体（支持多形状）
        self.cube_id, obj_half_height = self._create_object(
            pos_world=cube_pos,
            orn_quat=list(cube_orn),
            mass=float(mass),
            friction=float(friction),
            object_spec=object_spec,
            cube_scale=float(cube_scale)
        )
        
        # 物理参数
        p.changeDynamics(
            self.cube_id, -1,
            mass=mass,
            lateralFriction=friction,
            spinningFriction=friction * 0.3,
            restitution=(float(restitution) if restitution is not None else 0.0)
        )
        for i in self.gripper_indices:
            p.changeDynamics(self.panda_id, i, lateralFriction=friction)

        # 障碍物（静态 box）
        obstacle_ids = []
        if obstacle_specs:
            for spec in obstacle_specs:
                try:
                    center = np.array(spec.get('center_world', [0.0, 0.0, self.table_height + 0.05]), dtype=float)
                    he = spec.get('half_extents', [0.03, 0.03, 0.05])
                    rgba = tuple(spec.get('rgba', [0.2, 0.2, 0.2, 1.0]))
                    euler = tuple(spec.get('euler', [0.0, 0.0, 0.0]))
                    obstacle_ids.append(self._create_obstacle_box(center_world=center, half_extents=(float(he[0]), float(he[1]), float(he[2])), rgba=rgba, euler=euler))
                except Exception:
                    continue
        
        # 重置机械臂（可能带噪声）
        self._reset_arm(randomize=randomize)
        
        # 稳定场景
        for _ in range(100):
            p.stepSimulation()
            
        # 可视化目标
        if p.getConnectionInfo(self.client)['connectionMethod'] == p.GUI:
            self._visualize_target()
            
        config = {
            'cube_initial_pos': cube_pos.tolist(),
            'cube_initial_orn': list(cube_orn),
            'cube_scale': cube_scale,
            'object_spec': object_spec,
            'object_scale': float(object_scale),
            'mass_scale': float(mass_scale),
            'friction_override': (float(friction_override) if friction_override is not None else None),
            'restitution': (float(restitution) if restitution is not None else None),
            'gravity_scale': float(gravity_scale),
            'sim_time_step': (float(sim_time_step) if sim_time_step is not None else None),
            'solver_iterations': (int(solver_iterations) if solver_iterations is not None else None),
            'target_xy_offset': (list(target_xy_offset) if target_xy_offset is not None else None),
            'obstacles_n': len(obstacle_ids),
            'cube_support_height': float(cube_support_height),
            'target_support_height': float(target_support_height),
            'cube_surface_height': float(self.current_cube_surface_height),
            'target_surface_height': float(self.current_target_surface_height),
            'target_pos': self.target_pos.tolist(),
            'randomized': randomize,
            'max_delta': self.current_max_delta,
            'approach_height': self.current_approach_height,
            'mass': mass,
            'friction': friction,
            'panda_base': self.panda_base_pos.tolist(),
            'panda_base_offset': base_offset.tolist()
        }
        
        return config
    
    def _visualize_target(self):
        """可视化目标位置"""
        visual_id = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.03,
            length=0.002,
            rgbaColor=[0, 1, 0, 0.5]
        )
        p.createMultiBody(
            baseVisualShapeIndex=visual_id,
            basePosition=self.target_pos.tolist()
        )
    
    def _reset_arm(self, randomize: bool = False):
        """重置机械臂"""
        home = np.array([0, -0.3, 0, -2.4, 0, 2.1, 0.78, 0.04, 0.04])
        
        if randomize:
            cfg = self.rand_config
            joint_noise = np.zeros(9)
            joint_noise[:7] = np.random.randn(7) * cfg.arm_joint_noise_std
            
            joint_limits = [
                (-2.8, 2.8), (-1.7, 1.7), (-2.8, 2.8), (-3.0, -0.1),
                (-2.8, 2.8), (0.0, 3.7), (-2.8, 2.8), (0, 0.04), (0, 0.04)
            ]
            
            home_noisy = home + joint_noise
            for i, (low, high) in enumerate(joint_limits):
                home_noisy[i] = np.clip(home_noisy[i], low, high)
            home_noisy[7:] = 0.04
            home = home_noisy
            
        for i, val in enumerate(home):
            p.resetJointState(self.panda_id, i, val)
            
    def get_observation(self) -> Observation:
        """获取观测"""
        joint_states = p.getJointStates(self.panda_id, self.arm_joint_indices)
        joint_pos = np.array([s[0] for s in joint_states])
        joint_vel = np.array([s[1] for s in joint_states])
        
        ee_state = p.getLinkState(self.panda_id, self.ee_index, computeLinkVelocity=1)
        ee_pos = np.array(ee_state[0])
        ee_orn = np.array(ee_state[1])
        
        gripper_state = p.getJointState(self.panda_id, self.gripper_indices[0])[0]
        gripper_normalized = gripper_state / 0.04
        
        obj_pos, obj_orn = p.getBasePositionAndOrientation(self.cube_id)
        obj_rel_pos = np.array(obj_pos) - ee_pos
        
        # === 新增：计算目标相对位置 ===
        if self.target_pos is None:
            # 防止初始化时报错
            target_rel_pos = np.zeros(3)
        else:
            target_rel_pos = self.target_pos - ee_pos

        return Observation(
            joint_positions=joint_pos,
            joint_velocities=joint_vel,
            ee_position=ee_pos,
            ee_orientation=ee_orn,
            gripper_state=gripper_normalized,
            object_rel_position=obj_rel_pos,
            object_rel_orientation=np.array(obj_orn), # 这里的参数现在能对应上了
            target_rel_position=target_rel_pos        # 新增参数
        )
    
    def _get_workspace_bounds(self) -> Dict:
        """获取安全工作空间边界（世界坐标）"""
        return {
            'x': (self.panda_base_pos[0] + 0.25, self.panda_base_pos[0] + 0.65),
            'y': (self.panda_base_pos[1] - 0.35, self.panda_base_pos[1] + 0.35),
            'z': (self.table_height + 0.02, self.table_height + 0.45)
        }
    
    def _clip_to_workspace(self, pos: np.ndarray) -> np.ndarray:
        """限制在工作空间内"""
        bounds = self._get_workspace_bounds()
        clipped = pos.copy()
        clipped[0] = np.clip(clipped[0], *bounds['x'])
        clipped[1] = np.clip(clipped[1], *bounds['y'])
        clipped[2] = np.clip(clipped[2], *bounds['z'])
        return clipped
    
    def execute_action(self, action: Action, steps: int = 10) -> Observation:
        """执行动作"""
        ee_state = p.getLinkState(self.panda_id, self.ee_index)
        current_pos = np.array(ee_state[0])
        
        noise = self._get_action_noise()
        noisy_delta = action.delta_position + noise
        
        target_pos = current_pos + noisy_delta
        target_pos = self._clip_to_workspace(target_pos)
        
        gripper_target = action.gripper_action * 0.04
        
        for _ in range(steps):
            joint_targets = p.calculateInverseKinematics(
                self.panda_id,
                self.ee_index,
                target_pos.tolist(),
                targetOrientation=self.gripper_orn_down,
                maxNumIterations=50
            )
            
            for i in range(7):
                p.setJointMotorControl2(
                    self.panda_id, i,
                    p.POSITION_CONTROL,
                    targetPosition=joint_targets[i],
                    force=240,
                    maxVelocity=2.0
                )
            
            gripper_force = 50 if action.gripper_action < 0.5 else 20
            for i in self.gripper_indices:
                p.setJointMotorControl2(
                    self.panda_id, i,
                    p.POSITION_CONTROL,
                    targetPosition=gripper_target,
                    force=gripper_force
                )
                
            p.stepSimulation()
            if self.render:
                time.sleep(self.dt*0.1)

            
        return self.get_observation()
    
    def _compute_action(self, current_pos: np.ndarray, 
                        target_pos: np.ndarray,
                        gripper_open: bool) -> Action:
        """计算动作"""
        delta = target_pos - current_pos
        delta_norm = np.linalg.norm(delta)
        
        if delta_norm > self.current_max_delta:
            delta = delta / delta_norm * self.current_max_delta
            
        return Action(
            delta_position=delta,
            gripper_action=1.0 if gripper_open else 0.0
        )
    
    def move_to_position(self, 
                         target_pos: np.ndarray,
                         gripper_open: bool = True,
                         threshold: float = 0.008,
                         add_waypoint_noise: bool = False) -> List[Tuple[np.ndarray, np.ndarray]]:
        """移动到目标位置"""
        trajectory_data = []
        
        # 确保目标在工作空间内
        target_pos = self._clip_to_workspace(target_pos)
        
        if add_waypoint_noise:
            noise = np.random.randn(3) * 0.015
            noise[2] = abs(noise[2]) * 0.5
            target_pos = self._clip_to_workspace(target_pos + noise)
        
        max_steps = 400
        stuck_count = 0
        prev_pos = None
        
        for step in range(max_steps):
            obs = self.get_observation()
            current_pos = obs.ee_position
            
            # 检查是否卡住
            if prev_pos is not None:
                if np.linalg.norm(current_pos - prev_pos) < 0.0005:
                    stuck_count += 1
                    if stuck_count > 30:
                        break
                else:
                    stuck_count = 0
            prev_pos = current_pos.copy()
            
            if np.linalg.norm(current_pos - target_pos) < threshold:
                break
                
            action = self._compute_action(current_pos, target_pos, gripper_open)
            trajectory_data.append((obs.to_vector(), action.to_vector()))
            self.execute_action(action, steps=5)
            
        return trajectory_data
    
    def control_gripper(self, open: bool, duration: float = None, hold_still: bool = False,
                        extra_hold_steps: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
        """控制夹爪"""
        cfg = self.rand_config
        
        if duration is None:
            if open:
                duration = np.random.uniform(*cfg.gripper_open_duration_range)
            else:
                duration = np.random.uniform(*cfg.gripper_close_duration_range)
                
        trajectory_data = []
        steps = int(duration / self.dt / 5)
        
        for _ in range(steps):
            obs = self.get_observation()
            delta = np.zeros(3) if hold_still else (self._get_action_noise() * 0.2)
            action = Action(
                delta_position=delta,
                gripper_action=1.0 if open else 0.0
            )
            trajectory_data.append((obs.to_vector(), action.to_vector()))
            self.execute_action(action, steps=5)
        
        # 额外保持张开几步，提供更强的释放信号
        for _ in range(extra_hold_steps):
            obs = self.get_observation()
            action = Action(delta_position=np.zeros(3), gripper_action=1.0 if open else 0.0)
            trajectory_data.append((obs.to_vector(), action.to_vector()))
            self.execute_action(action, steps=5)
            
        return trajectory_data
    
    def check_success(self, threshold: float = 0.05) -> Tuple[bool, bool]:
        """检查任务成功"""
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        cube_pos = np.array(cube_pos)
        
        distance = np.linalg.norm(cube_pos[:2] - self.target_pos[:2])
        on_table = self.table_height - 0.01 < cube_pos[2] < self.table_height + 0.08
        
        place_success = distance < threshold and on_table
        return True, place_success
    
    def collect_trajectory(self,
                           cube_pos: Optional[List] = None,
                           target_pos: Optional[List] = None,
                           randomize: bool = True,
                           seed: Optional[int] = None,
                           add_waypoint_noise: bool = True) -> Optional[Trajectory]:
        """收集一条轨迹"""
        
        config = self.setup_scene(cube_pos, target_pos, randomize, seed)
        
        if config is None:
            return None
            
        cube_actual_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        cube_actual_pos = np.array(cube_actual_pos)
        
        all_data = []
        
        # 阶段1: 物体上方
        above_pos = cube_actual_pos + np.array([0, 0, self.current_approach_height])
        data = self.move_to_position(above_pos, gripper_open=True, 
                                     add_waypoint_noise=add_waypoint_noise)
        all_data.extend(data)
        
        # 阶段2: 下降抓取
        grasp_pos = cube_actual_pos + np.array([0, 0, self.current_grasp_offset])
        data = self.move_to_position(grasp_pos, gripper_open=True)
        all_data.extend(data)
        
        # 阶段3: 闭合夹爪
        data = self.control_gripper(open=False)
        all_data.extend(data)
        
        # 阶段4: 提升
        lift_pos = cube_actual_pos + np.array([0, 0, self.current_approach_height + 0.03])
        data = self.move_to_position(lift_pos, gripper_open=False)
        all_data.extend(data)
        
        # 验证抓取
        cube_now, _ = p.getBasePositionAndOrientation(self.cube_id)
        grasp_success = cube_now[2] > self.table_height + 0.06
        
        if not grasp_success:
            print("  抓取失败")
            
        # 阶段5: 目标上方
        target_above = self.target_pos + np.array([0, 0, self.current_approach_height])
        data = self.move_to_position(target_above, gripper_open=False,
                                     add_waypoint_noise=add_waypoint_noise)
        all_data.extend(data)
        
        # 阶段6: 放置
        place_pos = self.target_pos + np.array([0, 0, self.current_place_offset])
        data = self.move_to_position(place_pos, gripper_open=False)
        all_data.extend(data)
        
        # 阶段7: 张开夹爪，保持静止多采样释放信号
        data = self.control_gripper(open=True, hold_still=True, extra_hold_steps=10)
        all_data.extend(data)
        
        # 阶段8: 后撤
        data = self.move_to_position(target_above, gripper_open=True)
        all_data.extend(data)
        
        # 稳定
        for _ in range(60):
            p.stepSimulation()
            
        _, place_success = self.check_success()
        success = grasp_success and place_success
        
        if len(all_data) == 0:
            return None
            
        observations = np.array([d[0] for d in all_data])
        actions = np.array([d[1] for d in all_data])
        
        trajectory = Trajectory(
            observations=observations,
            actions=actions,
            success=success,
            task_config=config
        )
        
        self.trajectories.append(trajectory)
        return trajectory
    
    def collect_dataset(self,
                        num_trajectories: int = 100,
                        randomize: bool = True,
                        verbose: bool = True,
                        max_attempts_per_traj: int = 5) -> List[Trajectory]:
        """收集数据集"""
        
        successful = []
        total_attempts = 0
        failed_setup = 0
        failed_execution = 0
        
        while len(successful) < num_trajectories:
            total_attempts += 1
            
            if total_attempts > num_trajectories * max_attempts_per_traj:
                print(f"\n达到最大尝试次数")
                break
                
            if verbose:
                print(f"轨迹 {len(successful)+1}/{num_trajectories} "
                      f"(尝试 {total_attempts})...", end=" ")
                
            traj = self.collect_trajectory(
                randomize=randomize,
                seed=total_attempts * 7919,
                add_waypoint_noise=randomize
            )
            
            if traj is None:
                failed_setup += 1
                if verbose:
                    print("场景设置失败")
                continue
                
            if traj.success:
                successful.append(traj)
                if verbose:
                    print(f"✓ 成功 (长度: {traj.length})")
            else:
                failed_execution += 1
                if verbose:
                    print("✗ 执行失败")
                    
        if verbose:
            print(f"\n=== 统计 ===")
            print(f"成功: {len(successful)}")
            print(f"场景设置失败: {failed_setup}")
            print(f"执行失败: {failed_execution}")
            print(f"总尝试: {total_attempts}")
            if successful:
                print(f"总步数: {sum(t.length for t in successful)}")
                print(f"平均轨迹长度: {np.mean([t.length for t in successful]):.1f}")
            
        return successful

    def collect_dataset_parallel(self,
                                 num_trajectories: int = 100,
                                 workers: int = 4,
                                 randomize: bool = True,
                                 verbose: bool = True,
                                 max_attempts_per_traj: int = 5) -> List[Trajectory]:
        """多进程收集数据集"""
        if workers <= 1:
            return self.collect_dataset(num_trajectories, randomize, verbose, max_attempts_per_traj)

        workers = max(1, workers)
        # 平均分配任务
        base = num_trajectories // workers
        remainder = num_trajectories % workers
        plan = [base + (1 if i < remainder else 0) for i in range(workers)]

        ctx = mp.get_context("spawn")
        params = [
            (i, plan[i], randomize, 1234 + i * 1000000, max_attempts_per_traj, randomize)
            for i in range(workers)
            if plan[i] > 0
        ]

        if verbose:
            print(f"使用 {len(params)} 个进程并行收集，共 {num_trajectories} 条目标轨迹")

        with ctx.Pool(processes=len(params)) as pool:
            results = pool.map(_collect_worker, params)

        gathered = [traj for sub in results for traj in sub]
        self.trajectories.extend(gathered)

        if verbose:
            print(f"并行收集完成，成功 {len(gathered)}/{num_trajectories}")

        return gathered
    
    def save_dataset(self, filepath: str, only_successful: bool = True):
        """保存数据集"""
        trajectories = [t for t in self.trajectories if t.success] if only_successful else self.trajectories
        
        if len(trajectories) == 0:
            print("没有轨迹可保存")
            return
            
        dataset = {
            'observations': np.concatenate([t.observations for t in trajectories]),
            'actions': np.concatenate([t.actions for t in trajectories]),
            'trajectory_lengths': [t.length for t in trajectories],
            'num_trajectories': len(trajectories),
            'obs_dim': Observation.vector_dim(),
            'action_dim': Action.vector_dim(),
            'configs': [t.task_config for t in trajectories]
        }
        
        dataset['obs_mean'] = dataset['observations'].mean(axis=0)
        dataset['obs_std'] = dataset['observations'].std(axis=0) + 1e-8
        dataset['action_mean'] = dataset['actions'].mean(axis=0)
        dataset['action_std'] = dataset['actions'].std(axis=0) + 1e-8
        dataset['total_steps'] = len(dataset['observations'])
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f)
            
        print(f"\n保存: {filepath}")
        print(f"  轨迹数: {dataset['num_trajectories']}")
        print(f"  总步数: {dataset['total_steps']}")
        
    def close(self):
        p.disconnect()


# ==================== 工具函数 ====================

def load_dataset(filepath: str) -> Dict:
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def _collect_worker(args):
    """子进程收集若干成功轨迹"""
    worker_id, target_num, randomize, seed_base, max_attempts_per_traj, add_waypoint_noise = args

    np.random.seed(seed_base)
    collector = ExpertDemoCollector(gui=False, noise_type=NoiseType.COMBINED)
    successful = []
    attempts = 0

    while len(successful) < target_num and attempts < target_num * max_attempts_per_traj:
        attempts += 1
        seed = seed_base + attempts + worker_id * 100000
        traj = collector.collect_trajectory(
            randomize=randomize,
            seed=seed,
            add_waypoint_noise=add_waypoint_noise
        )

        if traj and traj.success:
            successful.append(traj)

    collector.close()
    return successful


def print_workspace_info(collector: ExpertDemoCollector):
    """打印工作空间信息"""
    print("\n=== 工作空间信息 ===")
    print(f"机械臂基座: {collector.panda_base_pos}")
    print(f"桌面高度: {collector.table_height}")
    
    bounds = collector._get_workspace_bounds()
    print(f"工作空间边界:")
    print(f"  X: [{bounds['x'][0]:.3f}, {bounds['x'][1]:.3f}]")
    print(f"  Y: [{bounds['y'][0]:.3f}, {bounds['y'][1]:.3f}]")
    print(f"  Z: [{bounds['z'][0]:.3f}, {bounds['z'][1]:.3f}]")
    
    cfg = collector.rand_config
    print(f"\n物体随机范围 (相对基座):")
    print(f"  X: {cfg.cube_pos_x_range}")
    print(f"  Y: {cfg.cube_pos_y_range}")
    print(f"\n目标随机范围 (相对基座):")
    print(f"  X: {cfg.target_pos_x_range}")
    print(f"  Y: {cfg.target_pos_y_range}")


# ==================== 主程序 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("专家示范收集器 - 安全随机化版本")
    print("=" * 60)
    
    collector = ExpertDemoCollector(
        gui=False,
        noise_type=NoiseType.COMBINED
    )
    
    # 打印工作空间信息
    print_workspace_info(collector)
    
    print("\n[1] 测试单条轨迹...")
    traj = collector.collect_trajectory(randomize=True)
    if traj:
        print(f"    长度: {traj.length}, 成功: {traj.success}")
    
    print("\n[2] 收集数据集...")
    workers = max(1, min(mp.cpu_count() or 1, 20))
    successful = collector.collect_dataset_parallel(
        num_trajectories=3000,
        workers=workers,
        randomize=True,
        verbose=True
    )
    
    print("\n[3] 保存数据...")
    collector.save_dataset("data/expert_demos.pkl")
    
    print("\n完成! 按回车退出...")
    input()
    collector.close()