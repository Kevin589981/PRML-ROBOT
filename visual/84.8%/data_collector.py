# data_collector.py

import pybullet as p
import pybullet_data
import math
import numpy as np
import h5py
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum, auto
import json
import os
import warnings
import cv2
import copy

warnings.filterwarnings('ignore')


# ==================== 0. 视频录制器 ====================

class VideoRecorder:
    """多视角并排录制的视频录制器"""
    def __init__(self, filepath, camera_manager, camera_types_to_record, client_id, fps=25, width=480, height=360):
        self.filepath = filepath
        self.camera_manager = camera_manager
        self.camera_types_to_record = camera_types_to_record
        self.client_id = client_id
        self.num_cameras = len(camera_types_to_record)
        self.fps = fps
        self.width = width
        self.height = height
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        final_width = self.width * self.num_cameras
        self.writer = cv2.VideoWriter(self.filepath, fourcc, self.fps, (final_width, self.height))
        
        if not self.writer.isOpened():
            raise IOError(f"无法打开视频文件进行写入: {self.filepath}")

    def add_frame(self, ee_pos, ee_orn):
        """添加一帧（多视角并排）"""
        panel_images = []
        
        for cam_type in self.camera_types_to_record:
            cam_name = cam_type.name.lower()
            cfg = self.camera_manager.cameras[cam_name]
            
            if cam_type.name == 'WRIST':
                rot = np.array(p.getMatrixFromQuaternion(ee_orn, physicsClientId=self.client_id)).reshape(3,3)
                cam_pos = ee_pos + rot @ cfg.wrist_offset
                target = ee_pos + rot @ cfg.wrist_look_offset
                up = rot @ [0, 1, 0]
                vm = p.computeViewMatrix(cam_pos, target, up, physicsClientId=self.client_id)
                pm = p.computeProjectionMatrixFOV(cfg.fov, self.width/self.height, cfg.near, cfg.far, physicsClientId=self.client_id)
            else:
                vm = self.camera_manager._view_matrices[cam_name]
                pm = p.computeProjectionMatrixFOV(cfg.fov, self.width/self.height, cfg.near, cfg.far, physicsClientId=self.client_id)
            
            _, _, rgb, _, _ = p.getCameraImage(
                width=self.width, height=self.height,
                viewMatrix=vm, projectionMatrix=pm,
                renderer=p.ER_TINY_RENDERER,
                physicsClientId=self.client_id
            )
            
            rgb = np.array(rgb, dtype=np.uint8).reshape(self.height, self.width, 4)[:, :, :3]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            panel_images.append(bgr)
        
        final_frame = cv2.hconcat(panel_images)
        self.writer.write(final_frame)

    def close(self):
        if self.writer.isOpened():
            self.writer.release()


# ==================== 1. 配置模块 ====================

class CameraType(Enum):
    OVERHEAD = auto()
    FRONT_45 = auto()
    SIDE = auto()
    WRIST = auto()

class TaskPhase(Enum):
    APPROACH = 0      # 粗略接近方块区域
    ALIGN_GRASP = 1   # [新增] 在方块上方精细对齐XY
    GRASP = 2         # 下降并抓取
    LIFT = 3          # 提升
    TRANSPORT = 4     # 粗略平移到篓子区域
    ALIGN_RELEASE = 5 # [新增] 在篓子上方精细对齐XY
    DESCEND = 6       # 下降到篓子内
    RELEASE = 7       # 释放
    RETREAT = 8       # 撤退

@dataclass
class CameraConfig:
    camera_type: CameraType
    width: int = 112
    height: int = 112
    fov: float = 60.0
    near: float = 0.01
    far: float = 1.5
    position: Optional[np.ndarray] = None
    target: Optional[np.ndarray] = None
    up_vector: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1]))
    wrist_offset: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0.05]))
    wrist_look_offset: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0.15]))


@dataclass
class BasketConfig:
    """篓子配置"""
    position_relative: np.ndarray = field(default_factory=lambda: np.array([0.5, -0.56, 0.0]))
    inner_size: Tuple[float, float, float] = (0.12, 0.12, 0.08)
    wall_thickness: float = 0.008
    base_color: Tuple[float, float, float, float] = (0.2, 0.5, 0.8, 1.0)
    wall_color: Tuple[float, float, float, float] = (0.3, 0.6, 0.9, 1.0)
    
@dataclass
class DataCollectionConfig:
    image_width: int = 112
    image_height: int = 112
    use_depth: bool = True
    
    camera_types: Tuple[CameraType, ...] = (
        CameraType.FRONT_45,
        CameraType.WRIST
    )
    
    include_gripper_state: bool = True
    include_joint_positions: bool = True
    include_ee_position: bool = True
    include_ee_velocity: bool = True
    include_gripper_width: bool = True
    
    # 增强状态
    include_contact_sensor: bool = True
    # [Hard Mode] 纯视觉挑战，关闭相对坐标作弊
    include_object_relative_pos: bool = False
    
    randomize_camera_position: bool = True
    camera_position_noise_std: float = 0.01
    randomize_object_color: bool = True
    
    basket_config: BasketConfig = field(default_factory=BasketConfig)

@dataclass
class RandomizationConfig:
    cube_pos_x_range: Tuple[float, float] = (0.25, 0.65)
    cube_pos_y_range: Tuple[float, float] = (-0.35, 0.40)
    cube_rotation_z_range: Tuple[float, float] = (-math.pi, math.pi)
    cube_scale_range: Tuple[float, float] = (0.028, 0.035)
    
    # [新增] 篓子位置随机化 (只动X，不动Y)
    basket_pos_x_noise: float = 0.08 # +/- 8cm 的随机范围
    
    # [新增] 专家操作误差 (模拟人操作时的不完美对齐)
    expert_alignment_noise: float = 0.03 # 粗略阶段允许 3cm 的误差
    
    # [新增] 抓夹末端绝对坐标位置随机化
    ee_pos_noise: float = 0.0 # 默认不开启

    action_noise_prob: float = 0.3
    action_gaussian_noise_std: float = 0.002
    
    max_delta_range: Tuple[float, float] = (0.02, 0.03)
    drop_detection_distance: float = 0.08


# ==================== 2. 数据结构 ====================

@dataclass
class VisualObservation:
    rgb_images: Dict[str, np.ndarray]
    depth_images: Optional[Dict[str, np.ndarray]] = None
    gripper_state: Optional[float] = None
    joint_positions: Optional[np.ndarray] = None
    ee_position: Optional[np.ndarray] = None
    ee_velocity: Optional[np.ndarray] = None
    gripper_width: Optional[float] = None
    is_grasping: Optional[float] = None
    object_relative_pos: Optional[np.ndarray] = None
    phase_label: Optional[int] = None
    
    def get_stacked_rgb(self, camera_order: List[str]) -> np.ndarray:
        return np.stack([self.rgb_images[cam] for cam in camera_order], axis=0)
    
    def get_stacked_depth(self, camera_order: List[str]) -> Optional[np.ndarray]:
        if self.depth_images is None: return None
        return np.stack([self.depth_images[cam] for cam in camera_order if cam in self.depth_images], axis=0)
    
    def get_auxiliary_state(self) -> np.ndarray:
        states = []
        if self.gripper_state is not None: states.append([self.gripper_state])
        if self.gripper_width is not None: states.append([self.gripper_width])
        if self.joint_positions is not None: states.append(self.joint_positions)
        if self.ee_position is not None: states.append(self.ee_position)
        if self.ee_velocity is not None: states.append(self.ee_velocity)
        if self.is_grasping is not None: states.append([self.is_grasping])
        if self.object_relative_pos is not None: states.append(self.object_relative_pos)
        return np.concatenate(states) if states else np.array([])

@dataclass
class Action:
    delta_position: np.ndarray
    gripper_action: float
    
    def to_vector(self) -> np.ndarray:
        return np.concatenate([self.delta_position, [self.gripper_action]])

@dataclass
class VisualTrajectory:
    rgb_images: np.ndarray
    depth_images: Optional[np.ndarray]
    actions: np.ndarray
    auxiliary_states: Optional[np.ndarray]
    camera_names: List[str]
    phase_labels: np.ndarray 
    success: bool = False
    
    @property
    def length(self) -> int:
        return len(self.rgb_images)


# ==================== 3. 场景组件构建器 ====================

class BasketBuilder:
    @staticmethod
    def create_basket(config, base_position, table_height, client_id):
        body_ids = []
        inner_l, inner_w, inner_h = config.inner_size
        wall_t = config.wall_thickness
        
        # 计算世界坐标
        basket_center_x = base_position[0] + config.position_relative[0]
        basket_center_y = base_position[1] + config.position_relative[1]
        basket_floor_z = table_height - inner_h - wall_t
        
        base_half_extents = [inner_l/2 + wall_t, inner_w/2 + wall_t, wall_t/2]
        base_center = [basket_center_x, basket_center_y, basket_floor_z + wall_t/2]
        
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=base_half_extents, physicsClientId=client_id)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=base_half_extents, rgbaColor=list(config.base_color), physicsClientId=client_id)
        body_ids.append(p.createMultiBody(0, col, vis, base_center, physicsClientId=client_id))
        
        wall_center_z = table_height - inner_h/2
        walls_config = [
            {'half': [wall_t/2, inner_w/2 + wall_t, inner_h/2],
             'center': [basket_center_x + inner_l/2 + wall_t/2, basket_center_y, wall_center_z]},
            {'half': [wall_t/2, inner_w/2 + wall_t, inner_h/2],
             'center': [basket_center_x - inner_l/2 - wall_t/2, basket_center_y, wall_center_z]},
            {'half': [inner_l/2 + wall_t, wall_t/2, inner_h/2],
             'center': [basket_center_x, basket_center_y + inner_w/2 + wall_t/2, wall_center_z]},
            {'half': [inner_l/2 + wall_t, wall_t/2, inner_h/2],
             'center': [basket_center_x, basket_center_y - inner_w/2 - wall_t/2, wall_center_z]},
        ]
        
        for wall in walls_config:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=wall['half'], physicsClientId=client_id)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=wall['half'], rgbaColor=list(config.wall_color), physicsClientId=client_id)
            body_ids.append(p.createMultiBody(0, col, vis, wall['center'], physicsClientId=client_id))
        
        target_pos = np.array([basket_center_x, basket_center_y, table_height - inner_h * 0.4])
        return body_ids, target_pos


class MultiCameraManager:
    def __init__(self, config, rand_config, panda_base_pos, table_height, basket_pos, client_id):
        self.config = config
        self.client_id = client_id
        self.cameras = {}
        self._view_matrices = {}
        self._proj_matrices = {}
        
        center = panda_base_pos + np.array([0.45, 0.0, 0.05])
        for cam_type in self.config.camera_types:
            name = cam_type.name.lower()
            if cam_type == CameraType.OVERHEAD:
                self.cameras[name] = CameraConfig(
                    cam_type, config.image_width, config.image_height, 
                    fov=58.0, position=center+[0,0,0.85], target=center, up_vector=[1,0,0]
                )
            elif cam_type == CameraType.FRONT_45:
                self.cameras[name] = CameraConfig(
                    cam_type, config.image_width, config.image_height,
                    fov=65.0, position=panda_base_pos+[0.7, 0.3, 0.45], target=center
                )
            elif cam_type == CameraType.SIDE:
                self.cameras[name] = CameraConfig(
                    cam_type, config.image_width, config.image_height,
                    fov=55.0, position=center+[0.0, 0.6, 0.4], target=center
                )
            elif cam_type == CameraType.WRIST:
                self.cameras[name] = CameraConfig(
                    cam_type, config.image_width, config.image_height,
                    fov=85.0, near=0.01, far=1.0, 
                    wrist_offset=[0.10, 0, -0.05], wrist_look_offset=[-0.06, 0, 0.06]
                )
        self._update_fixed_cameras(False)

    def _update_fixed_cameras(self, add_noise):
        for name, cfg in self.cameras.items():
            if cfg.camera_type == CameraType.WRIST: continue
            pos, target = cfg.position.copy(), cfg.target.copy()
            if add_noise and self.config.randomize_camera_position:
                noise = np.random.randn(3) * self.config.camera_position_noise_std
                pos += noise
                target += noise * 0.2
            self._view_matrices[name] = p.computeViewMatrix(pos, target, cfg.up_vector, physicsClientId=self.client_id)
            self._proj_matrices[name] = p.computeProjectionMatrixFOV(cfg.fov, cfg.width/cfg.height, cfg.near, cfg.far, physicsClientId=self.client_id)

    def capture_all(self, ee_pos, ee_orn) -> VisualObservation:
        rgbs, depths = {}, {} if self.config.use_depth else None
        
        for name, cfg in self.cameras.items():
            if cfg.camera_type == CameraType.WRIST:
                rot = np.array(p.getMatrixFromQuaternion(ee_orn, physicsClientId=self.client_id)).reshape(3,3)
                cam_pos = ee_pos + rot @ cfg.wrist_offset
                target = ee_pos + rot @ cfg.wrist_look_offset
                up = rot @ [0,1,0]
                vm = p.computeViewMatrix(cam_pos, target, up, physicsClientId=self.client_id)
                pm = p.computeProjectionMatrixFOV(cfg.fov, cfg.width/cfg.height, cfg.near, cfg.far, physicsClientId=self.client_id)
            else:
                vm, pm = self._view_matrices[name], self._proj_matrices[name]
                
            _, _, rgb, depth, _ = p.getCameraImage(
                cfg.width, cfg.height, vm, pm, 
                renderer=p.ER_TINY_RENDERER, flags=p.ER_NO_SEGMENTATION_MASK, 
                physicsClientId=self.client_id
            )
            
            rgbs[name] = np.array(rgb, dtype=np.uint8).reshape(cfg.height, cfg.width, 4)[:,:,:3]
            if self.config.use_depth:
                d_buf = np.array(depth, dtype=np.float32).reshape(cfg.height, cfg.width)
                real_d = cfg.far * cfg.near / (cfg.far - (cfg.far - cfg.near) * d_buf)
                depths[name] = np.clip(real_d, 0, cfg.far).astype(np.float32)
                
        return VisualObservation(rgbs, depths)
    
    def randomize(self): self._update_fixed_cameras(True)
    def get_names(self): return list(self.cameras.keys())


# ==================== 4. 核心收集器 ====================

class VisualExpertDemoCollector:
    def __init__(self, data_config=None, rand_config=None, record_video=False, video_save_dir="videos"):
        self.client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        
        self.data_config = data_config or DataCollectionConfig()
        self.rand_config = rand_config or RandomizationConfig()
        
        self.record_video = record_video
        self.video_save_dir = video_save_dir
        if self.record_video: os.makedirs(self.video_save_dir, exist_ok=True)

        self.ee_index = 11
        self.gripper_indices = [9, 10]
        self.arm_joints = list(range(7))
        self.base_pos = np.array([-0.5, 0, 0.625])
        self.table_height = 0.625
        self.trajectories = []
        self.current_phase = TaskPhase.APPROACH

    def setup_scene(self, seed=None):
        if seed is not None: np.random.seed(seed)
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)
        
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        p.loadURDF("table/table.urdf", [0,0,0], useFixedBase=True, physicsClientId=self.client_id)
        self.panda_id = p.loadURDF("franka_panda/panda.urdf", self.base_pos, useFixedBase=True, physicsClientId=self.client_id)
        
        for i, val in enumerate([0, -0.3, 0, -2.4, 0, 2.1, 0.78, 0.04, 0.04]):
            p.resetJointState(self.panda_id, i, val, physicsClientId=self.client_id)
        for j in self.gripper_indices:
            p.resetJointState(self.panda_id, j, 0.04, physicsClientId=self.client_id)
            
        # [新增] 抓夹末端绝对坐标位置随机化 (在初始状态基础上叠加全向位置噪声)
        if self.rand_config.ee_pos_noise > 0:
            # 1. 获取复位后的标准末端位姿
            ee_state = p.getLinkState(self.panda_id, self.ee_index, physicsClientId=self.client_id)
            std_pos = np.array(ee_state[0])
            std_orn = np.array(ee_state[1])
            
            # 2. 生成随机噪声 dx, dy, dz
            noise = np.random.uniform(
                -self.rand_config.ee_pos_noise, 
                self.rand_config.ee_pos_noise, 
                size=3
            )
            target_pos = std_pos + noise
            # 确保不会撞到桌子（简单限制一下最小高度）
            if target_pos[2] < self.table_height + 0.05:
                target_pos[2] = self.table_height + 0.05

            # 3. IK解算并重置关节
            joint_poses = p.calculateInverseKinematics(
                self.panda_id, 
                self.ee_index, 
                target_pos, 
                std_orn, 
                maxNumIterations=50,
                physicsClientId=self.client_id
            )
            # 排除gripper joints，只取前7个关节
            for i in range(7):
                p.resetJointState(self.panda_id, i, joint_poses[i], physicsClientId=self.client_id)

        # ====================================================
        # [新增] 篓子位置随机化
        # 复制一份配置，避免修改全局配置
        current_basket_config = copy.deepcopy(self.data_config.basket_config)
        
        # 随机扰动 X 坐标 (Forward/Backward), Y 坐标 (Left/Right) 保持不变
        x_noise = np.random.uniform(-self.rand_config.basket_pos_x_noise, self.rand_config.basket_pos_x_noise)
        current_basket_config.position_relative[0] += x_noise
        
        # 创建篓子
        self.basket_ids, self.target_pos = BasketBuilder.create_basket(
            current_basket_config, self.base_pos, self.table_height, self.client_id
        )
        # ====================================================
        
        rx = np.random.uniform(*self.rand_config.cube_pos_x_range)
        ry = np.random.uniform(*self.rand_config.cube_pos_y_range)
        yaw = np.random.uniform(*self.rand_config.cube_rotation_z_range)
        scale = np.random.uniform(*self.rand_config.cube_scale_range)
        
        c_pos = self.base_pos + [rx, ry, 0]
        c_pos[2] = self.table_height + scale/2
        self.cube_id = p.loadURDF("cube.urdf", c_pos, p.getQuaternionFromEuler([0,0,yaw]), globalScaling=scale, physicsClientId=self.client_id)
        p.changeVisualShape(self.cube_id, -1, rgbaColor=[*np.random.uniform(0.2, 0.9, 3), 1], physicsClientId=self.client_id)
        p.changeDynamics(self.cube_id, -1, mass=0.1, lateralFriction=1.0, physicsClientId=self.client_id)
        
        # 相机需要基于当前的篓子位置初始化 (Side camera 等可能需要)
        # 这里 b_w_pos 使用随机化后的位置
        b_w_pos = self.base_pos + current_basket_config.position_relative
        self.camera_mgr = MultiCameraManager(
            self.data_config, self.rand_config, self.base_pos, self.table_height, b_w_pos, self.client_id
        )
        self.camera_mgr.randomize()
        for _ in range(50): p.stepSimulation(physicsClientId=self.client_id)
        self.current_phase = TaskPhase.APPROACH

    def _check_contact(self) -> bool:
        contacts_left = p.getContactPoints(bodyA=self.panda_id, bodyB=self.cube_id, linkIndexA=self.gripper_indices[0], physicsClientId=self.client_id)
        contacts_right = p.getContactPoints(bodyA=self.panda_id, bodyB=self.cube_id, linkIndexA=self.gripper_indices[1], physicsClientId=self.client_id)
        return len(contacts_left) > 0 and len(contacts_right) > 0

    def get_obs(self, phase: TaskPhase = None) -> VisualObservation:
        ee = p.getLinkState(self.panda_id, self.ee_index, computeLinkVelocity=1, physicsClientId=self.client_id)
        ee_pos = np.array(ee[0], dtype=np.float32)
        ee_orn = np.array(ee[1], dtype=np.float32)
        obs = self.camera_mgr.capture_all(ee_pos, ee_orn)
        
        if self.data_config.include_gripper_state:
            obs.gripper_state = p.getJointState(self.panda_id, self.gripper_indices[0], physicsClientId=self.client_id)[0]
        if self.data_config.include_joint_positions:
            obs.joint_positions = np.array([s[0] for s in p.getJointStates(self.panda_id, self.arm_joints, physicsClientId=self.client_id)])
        if self.data_config.include_ee_position:
            obs.ee_position = ee_pos.astype(np.float32)
        if self.data_config.include_ee_velocity:
            try: lin_vel = np.array(ee[6], dtype=np.float32)
            except: lin_vel = np.zeros(3, dtype=np.float32)
            obs.ee_velocity = lin_vel
        if self.data_config.include_gripper_width:
            j0 = p.getJointState(self.panda_id, self.gripper_indices[0], physicsClientId=self.client_id)[0]
            j1 = p.getJointState(self.panda_id, self.gripper_indices[1], physicsClientId=self.client_id)[0]
            obs.gripper_width = float(j0 + j1)
        if self.data_config.include_contact_sensor:
            obs.is_grasping = 1.0 if self._check_contact() else 0.0
        if self.data_config.include_object_relative_pos:
            cube_pos = np.array(p.getBasePositionAndOrientation(self.cube_id, physicsClientId=self.client_id)[0])
            obs.object_relative_pos = (cube_pos - ee_pos).astype(np.float32)
        
        obs.phase_label = phase.value if phase is not None else self.current_phase.value
        return obs

    def execute_action(self, action: Action, steps=10):
        target = np.array(p.getLinkState(self.panda_id, self.ee_index, physicsClientId=self.client_id)[0]) + action.delta_position
        target[0] = np.clip(target[0], self.base_pos[0]+0.2, self.base_pos[0]+0.8)
        target[1] = np.clip(target[1], -0.65, 0.4)
        target[2] = np.clip(target[2], self.table_height+0.01, self.table_height+0.5)
        
        g_val = 0.04 if action.gripper_action > 0.5 else 0.0
        orn = p.getQuaternionFromEuler([math.pi, 0, 0])
        
        for _ in range(steps):
            j_pos = p.calculateInverseKinematics(self.panda_id, self.ee_index, target, orn, maxNumIterations=20, physicsClientId=self.client_id)
            for i in range(7): p.setJointMotorControl2(self.panda_id, i, p.POSITION_CONTROL, j_pos[i], force=200, physicsClientId=self.client_id)
            for i in self.gripper_indices: p.setJointMotorControl2(self.panda_id, i, p.POSITION_CONTROL, g_val, force=50, physicsClientId=self.client_id)
            p.stepSimulation(physicsClientId=self.client_id)

    def check_cube_dropped(self) -> bool:
        ee_pos = np.array(p.getLinkState(self.panda_id, self.ee_index, physicsClientId=self.client_id)[0])
        cube_pos = np.array(p.getBasePositionAndOrientation(self.cube_id, physicsClientId=self.client_id)[0])
        gripper_state = p.getJointState(self.panda_id, self.gripper_indices[0], physicsClientId=self.client_id)[0]
        
        distance = np.linalg.norm(ee_pos - cube_pos)
        if gripper_state < 0.02 and distance > self.rand_config.drop_detection_distance: return True
        return False

    def _move_to_target(self, target, gripper_action, pos_threshold, max_frames, phys_steps, max_step, check_drop, data, recorder, phase):
        frame_count = 0
        target_noisy = target + np.random.randn(3) * 0.002
        while frame_count < max_frames:
            obs = self.get_obs(phase=phase)
            curr = np.array(p.getLinkState(self.panda_id, self.ee_index, physicsClientId=self.client_id)[0])
            delta = target_noisy - curr
            dist = np.linalg.norm(delta)
            if dist > max_step: delta = delta / dist * max_step
            
            act = Action(delta, gripper_action)
            self.execute_action(act, steps=phys_steps)
            data.append((obs, act))
            frame_count += 1
            
            if recorder:
                ee = p.getLinkState(self.panda_id, self.ee_index, physicsClientId=self.client_id)
                recorder.add_frame(np.array(ee[0]), np.array(ee[1]))
            
            new_pos = np.array(p.getLinkState(self.panda_id, self.ee_index, physicsClientId=self.client_id)[0])
            if np.linalg.norm(target_noisy - new_pos) < pos_threshold: return True, False
            if check_drop and self.check_cube_dropped(): return False, True
        return True, False

    def _execute_gripper_action(self, target_pos, gripper_action, num_frames, phys_steps, data, recorder, phase):
        for _ in range(num_frames):
            obs = self.get_obs(phase=phase)
            curr = np.array(p.getLinkState(self.panda_id, self.ee_index, physicsClientId=self.client_id)[0])
            delta = target_pos - curr
            if np.linalg.norm(delta) > 0.005: delta = delta / np.linalg.norm(delta) * 0.005
            
            act = Action(delta, gripper_action)
            self.execute_action(act, steps=phys_steps)
            data.append((obs, act))
            if recorder:
                ee = p.getLinkState(self.panda_id, self.ee_index, physicsClientId=self.client_id)
                recorder.add_frame(np.array(ee[0]), np.array(ee[1]))

    def collect_trajectory(self, seed=None) -> Optional[VisualTrajectory]:
        self.setup_scene(seed)
        data = []
        recorder = None
        if self.record_video:
            video_path = os.path.join(self.video_save_dir, f"trajectory_seed_{seed}.mp4")
            recorder = VideoRecorder(video_path, self.camera_mgr, [CameraType.FRONT_45, CameraType.WRIST], self.client_id)

        try:
            c_pos = np.array(p.getBasePositionAndOrientation(self.cube_id, physicsClientId=self.client_id)[0])
            dropped = False
            
            # =========================================================================
            # Phase 1: Approach (Simulate Imperfect Alignment)
            # =========================================================================
            # 专家不直接到正上方，而是到一个有随机 XY 偏差的位置 (Z=0.18 -> 0.15)
            # 生成随机偏差 (例如 +/- 2.5cm)
            xy_error = np.random.uniform(-self.rand_config.expert_alignment_noise, self.rand_config.expert_alignment_noise, size=2)
            imperfect_approach_pos = c_pos + [xy_error[0], xy_error[1], 0.15]
            
            # 移动到粗略位置
            waypoints_approach = [
                (c_pos + [xy_error[0], xy_error[1], 0.22], 1.0, 0.025, 40, 10, 0.030, False),
                (imperfect_approach_pos,                   1.0, 0.015, 30, 8,  0.020, False),
            ]
            for wp in waypoints_approach:
                _, dropped = self._move_to_target(*wp, data, recorder, TaskPhase.APPROACH)
                if dropped: break
            if dropped: return None

            # =========================================================================
            # Phase 2 (New): Align to Grasp (Correct the XY error)
            # =========================================================================
            # 从有偏差的位置平滑移动到完美的抓取预备位 (Z=0.15 -> Z=0.10)
            perfect_pre_grasp = c_pos + [0, 0, 0.10]
            # 这是一个关键的对齐动作
            _, dropped = self._move_to_target(perfect_pre_grasp, 1.0, 0.008, 25, 8, 0.012, False, data, recorder, TaskPhase.ALIGN_GRASP)
            if dropped: return None
            
            # 进一步下降到接触点前
            pre_grasp_low = c_pos + [0, 0, 0.005]
            _, dropped = self._move_to_target(pre_grasp_low, 1.0, 0.005, 20, 8, 0.008, False, data, recorder, TaskPhase.ALIGN_GRASP) # 也可以算作 GRASP 的前奏
            if dropped: return None

            # =========================================================================
            # Phase 3: Grasp
            # =========================================================================
            grasp_pos = c_pos + [0, 0, -0.005]
            self._execute_gripper_action(grasp_pos, 0.0, 28, 10, data, recorder, TaskPhase.GRASP)
            
            # Phase 4: Lift
            waypoints_lift = [
                (c_pos + [0, 0, 0.02], 0.0, 0.010, 20, 10, 0.015, True),
                (c_pos + [0, 0, 0.06], 0.0, 0.015, 25, 10, 0.020, True),
                (c_pos + [0, 0, 0.20], 0.0, 0.018, 25, 10, 0.025, True),
            ]
            for wp in waypoints_lift:
                _, dropped = self._move_to_target(*wp, data, recorder, TaskPhase.LIFT)
                if dropped: break
            if dropped: return None
            
            # =========================================================================
            # Phase 5: Transport (Simulate Imperfect Alignment above Basket)
            # =========================================================================
            # 目标：移动到篓子上方，但带有随机 XY 偏差
            xy_error_basket = np.random.uniform(-self.rand_config.expert_alignment_noise, self.rand_config.expert_alignment_noise, size=2)
            imperfect_basket_pos = self.target_pos + [xy_error_basket[0], xy_error_basket[1], 0.22]
            
            dist = np.linalg.norm(imperfect_basket_pos[:2] - c_pos[:2])
            frames = max(50, min(int(dist/0.04)+20, 120))
            _, dropped = self._move_to_target(imperfect_basket_pos, 0.0, 0.025, frames, 12, 0.05, True, data, recorder, TaskPhase.TRANSPORT)
            if dropped: return None
            
            # =========================================================================
            # Phase 6 (New): Align to Release
            # =========================================================================
            # 从有偏差的位置修正到正上方
            perfect_pre_release = self.target_pos + [0, 0, 0.18]
            _, dropped = self._move_to_target(perfect_pre_release, 0.0, 0.015, 30, 8, 0.020, True, data, recorder, TaskPhase.ALIGN_RELEASE)
            if dropped: return None

            # =========================================================================
            # Phase 7: Descend
            # =========================================================================
            waypoints_descend = [
                (self.target_pos + [0, 0, 0.10], 0.0, 0.012, 25, 8, 0.018, True),
                (self.target_pos + [0, 0, 0.03], 0.0, 0.008, 25, 8, 0.012, True),
            ]
            for wp in waypoints_descend:
                _, dropped = self._move_to_target(*wp, data, recorder, TaskPhase.DESCEND)
                if dropped: break
            if dropped: return None
            
            # Phase 8: Release
            release_pos = self.target_pos + [0, 0, 0.03]
            self._execute_gripper_action(release_pos, 1.0, 26, 10, data, recorder, TaskPhase.RELEASE)
            
            # Phase 9: Retreat
            waypoints_retreat = [
                (self.target_pos + [0, 0, 0.15], 1.0, 0.018, 20, 10, 0.025, False),
                (self.target_pos + [0, 0, 0.25], 1.0, 0.020, 25, 10, 0.030, False),
            ]
            for wp in waypoints_retreat:
                self._move_to_target(*wp, data, recorder, TaskPhase.RETREAT)
            
            # Success Check
            for _ in range(30): p.stepSimulation(physicsClientId=self.client_id)
            c_final, _ = p.getBasePositionAndOrientation(self.cube_id, physicsClientId=self.client_id)
            gripper = p.getJointState(self.panda_id, self.gripper_indices[0], physicsClientId=self.client_id)[0]
            
            in_xy = abs(c_final[0]-self.target_pos[0]) < 0.12/2*0.9 and abs(c_final[1]-self.target_pos[1]) < 0.12/2*0.9
            in_z = self.table_height-0.08 < c_final[2] < self.table_height
            success = in_xy and in_z and gripper > 0.03 and len(data) >= 50
            if not success: return None

            # Data Packaging
            names = self.camera_mgr.get_names()
            rgbs = np.stack([d[0].get_stacked_rgb(names) for d in data]).astype(np.uint8)
            
            depths = None
            if self.data_config.use_depth:
                depth_stack = np.stack([d[0].get_stacked_depth(names) for d in data]) # (T, N, H, W)
                depths = depth_stack[..., np.newaxis].astype(np.float16) # (T, N, H, W, 1)

            acts = np.stack([d[1].to_vector() for d in data]).astype(np.float32)
            auxs = np.stack([d[0].get_auxiliary_state() for d in data]).astype(np.float32)
            phases = np.array([d[0].phase_label for d in data], dtype=np.int8)
            
            print(f"✓ Success: {len(data)} frames")
            return VisualTrajectory(rgbs, depths, acts, auxs, names, phases, True)

        finally:
            if recorder: recorder.close()

    def close(self):
        p.disconnect(self.client_id)

    def save_dataset(self, filepath, split=0.9):
        if not self.trajectories: return
        print(f"\nSaving {len(self.trajectories)} trajectories...")
        all_acts = np.concatenate([t.actions for t in self.trajectories])
        all_aux = np.concatenate([t.auxiliary_states for t in self.trajectories])
        
        d_min, d_max = float('inf'), float('-inf')
        if self.trajectories[0].depth_images is not None:
            sample_trajs = self.trajectories[::10]
            for t in sample_trajs:
                d_min = min(d_min, t.depth_images.min())
                d_max = max(d_max, t.depth_images.max())
        
        stats = {
            'action_mean': all_acts.mean(0), 'action_std': all_acts.std(0) + 1e-6,
            'aux_mean': all_aux.mean(0), 'aux_std': all_aux.std(0) + 1e-6,
            'depth_min': d_min, 'depth_max': d_max,
            'num_phases': len(TaskPhase)
        }
        
        idxs = np.random.permutation(len(self.trajectories))
        sp = int(len(idxs) * split)
        train_idxs, val_idxs = idxs[:sp], idxs[sp:]
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with h5py.File(filepath, 'w') as f:
            meta = f.create_group('metadata')
            meta.attrs['num_trajectories'] = len(self.trajectories)
            meta.attrs['camera_names'] = json.dumps(self.trajectories[0].camera_names)
            meta.attrs['num_phases'] = len(TaskPhase)
            for k, v in stats.items(): 
                if isinstance(v, np.ndarray): meta.create_dataset(k, data=v)
                else: meta.attrs[k] = v
            meta.create_dataset('train_indices', data=train_idxs)
            meta.create_dataset('val_indices', data=val_idxs)
            
            for i, traj in enumerate(self.trajectories):
                g = f.create_group(f'trajectory_{i:04d}')
                g.create_dataset('rgb', data=traj.rgb_images, compression='gzip', compression_opts=4)
                if traj.depth_images is not None:
                    g.create_dataset('depth', data=traj.depth_images, compression='gzip', compression_opts=4)
                g.create_dataset('actions', data=traj.actions)
                g.create_dataset('aux', data=traj.auxiliary_states)
                g.create_dataset('phase_labels', data=traj.phase_labels)
                g.attrs['length'] = traj.length
        print(f"✓ Saved to {filepath}")

# ==================== 5. 并行执行 ====================

def _worker_task(seed, count, record_video=False):
    col = VisualExpertDemoCollector(record_video=record_video)
    res = []
    attempts = 0
    max_attempts = count * 10
    while len(res) < count and attempts < max_attempts:
        t = col.collect_trajectory(seed + attempts)
        if t: res.append(t)
        attempts += 1
    col.close()
    return res

def run_collection(total=1000, workers=8, record_video=False):
    workers = min(workers, os.cpu_count() or 4)
    per_w = total // workers if total >= workers else 1
    workers = total if total < workers else workers
    
    print(f"Collecting {total} demos with {workers} workers...")
    with mp.get_context("spawn").Pool(workers) as pool:
        results = pool.starmap(_worker_task, [(i*10000, per_w, record_video) for i in range(workers)])
    
    col = VisualExpertDemoCollector()
    col.trajectories = [t for r in results for t in r]
    return col

if __name__ == "__main__":
    mp.freeze_support()
    # 既然有 4090，建议把 total 设为 2000 以覆盖新增的随机性
    collector = run_collection(total=1000, workers=125, record_video=False) 
    if collector.trajectories:
        collector.save_dataset("data_enhanced/basket_demos_with_phase.h5")