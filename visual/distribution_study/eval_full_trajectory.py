# eval_full_trajectory.py

import os
import cv2
import torch
import numpy as np
import pybullet as p
from tqdm import tqdm
import multiprocessing as mp
import argparse
import time

# 必须导入训练文件中的模型类
from train_full_trajectory import FullTrajectoryPolicy, CONFIG as TRAIN_CONFIG
# 导入数据收集工具
from data_collector import (
    VisualExpertDemoCollector, 
    Action, 
    DataCollectionConfig, 
    RandomizationConfig, 
    CameraType,
    BasketConfig
)

# ==================== 1. 视频录制工具 (保持不变) ====================

class VideoRecorder:
    def __init__(self, save_dir, camera_manager, camera_types_to_record, fps=20, width=480, height=360):
        self.save_dir = save_dir
        self.camera_manager = camera_manager
        self.camera_types = camera_types_to_record
        self.fps = fps
        self.width = width
        self.height = height
        self.frames = []
        os.makedirs(save_dir, exist_ok=True)

    def capture(self, client_id, ee_pos, ee_orn):
        """同时捕获多个视角拼接"""
        images = []
        for cam_type in self.camera_types:
            cam_name = cam_type.name.lower()
            if cam_name not in self.camera_manager.cameras: continue
                
            cfg = self.camera_manager.cameras[cam_name]
            
            if cam_type == CameraType.WRIST:
                rot = np.array(p.getMatrixFromQuaternion(ee_orn, physicsClientId=client_id)).reshape(3,3)
                cam_pos = ee_pos + rot @ cfg.wrist_offset
                target = ee_pos + rot @ cfg.wrist_look_offset
                up = rot @ [0, 1, 0]
                vm = p.computeViewMatrix(cam_pos, target, up, physicsClientId=client_id)
                pm = p.computeProjectionMatrixFOV(cfg.fov, self.width/self.height, cfg.near, cfg.far, physicsClientId=client_id)
            else:
                vm = self.camera_manager._view_matrices[cam_name]
                pm = p.computeProjectionMatrixFOV(cfg.fov, self.width/self.height, cfg.near, cfg.far, physicsClientId=client_id)
            
            _, _, rgb, _, _ = p.getCameraImage(
                self.width, self.height, vm, pm, 
                renderer=p.ER_TINY_RENDERER, physicsClientId=client_id
            )
            rgb = np.array(rgb, dtype=np.uint8).reshape(self.height, self.width, 4)[:, :, :3]
            images.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            
        if images:
            self.frames.append(cv2.hconcat(images))

    def save(self, filename):
        if not self.frames: return
        path = os.path.join(self.save_dir, filename)
        H, W, _ = self.frames[0].shape
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (W, H))
        for f in self.frames: out.write(f)
        out.release()
        self.frames = []

# ==================== 2. 全轨迹智能体 ====================

class FullTrajectoryAgent:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        print(f"[Agent] Loading checkpoint: {checkpoint_path}")
        
        # weights_only=False 以支持完整的配置加载
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        self.config = ckpt['config']
        stats_np = ckpt['stats']
        
        self.stats = {}
        for k, v in stats_np.items():
            if isinstance(v, np.ndarray):
                self.stats[k] = torch.from_numpy(v).float().to(device)
            else:
                self.stats[k] = torch.tensor(v, device=device).float()
                
        aux_dim = len(self.stats['aux_mean'])
        num_phases = int(self.stats['num_phases'].item()) if 'num_phases' in self.stats else 9 # 默认为9
        
        self.model = FullTrajectoryPolicy(self.config, aux_dim, num_phases)
        
        # ========== 加载权重 (跳过 spatial softmax buffer) ==========
        state_dict = ckpt['model']
        new_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "")
            if "spatial_softmax.pos_x" in new_key or "spatial_softmax.pos_y" in new_key:
                continue
            new_dict[new_key] = v
            
        self.model.load_state_dict(new_dict, strict=False)
        self.model.to(device).eval()
        self.hidden = None
        
        # [关键修改] 更新 Phase Map 以匹配新的 9 阶段任务
        self.phase_map = {
            0: "Approach", 
            1: "AlignGrasp",   # 新增
            2: "Grasping", 
            3: "Lifting",
            4: "Transport", 
            5: "AlignRelease", # 新增
            6: "Descend", 
            7: "Releasing", 
            8: "Retreating"
        }

    def reset(self):
        self.hidden = None

    @torch.no_grad()
    def predict(self, obs):
        rgb_list, depth_list = [], []
        
        for cam_name in self.config['camera_names']:
            rgb = obs.rgb_images[cam_name].astype(np.float32) / 255.0
            rgb_list.append(rgb)
            if self.config['use_depth']:
                d = obs.depth_images[cam_name]
                d_min = self.stats['depth_min'].item()
                d_max = self.stats['depth_max'].item()
                d_norm = (np.clip(d, d_min, d_max) - d_min) / (d_max - d_min + 1e-6)
                depth_list.append(d_norm)
                
        rgb_np = np.stack(rgb_list)
        rgb_t = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).float().to(self.device)
        
        if self.config['use_depth']:
            d_np = np.stack(depth_list)
            d_t = torch.from_numpy(d_np).unsqueeze(1).float().to(self.device)
            img_t = torch.cat([rgb_t, d_t], dim=1)
        else:
            img_t = rgb_t
            
        img_in = img_t.unsqueeze(0).unsqueeze(0)
        
        aux = obs.get_auxiliary_state()
        aux_t = torch.from_numpy(aux).float().to(self.device)
        aux_in = (aux_t - self.stats['aux_mean']) / self.stats['aux_std']
        aux_in = aux_in.view(1, 1, -1)
        
        actions, phases, self.hidden = self.model(img_in, aux_in, self.hidden)
        
        pred_act = actions[0, 0]
        pred_phase = phases[0, 0]
        
        delta = (pred_act[:3] * self.stats['action_std'][:3] + self.stats['action_mean'][:3]).cpu().numpy()
        grip_prob = torch.sigmoid(pred_act[3]).item()
        grip_act = 1.0 if grip_prob > 0.5 else 0.0
        
        phase_probs = torch.softmax(pred_phase, dim=0)
        phase_idx = torch.argmax(phase_probs).item()
        
        return delta, grip_act, phase_idx

# ==================== 3. 单进程评估 Worker ====================

def eval_worker(worker_id, checkpoint_path, num_episodes, start_seed, save_video, max_steps, device, ee_pos_noise):
    """
    单个 Worker 进程
    """
    try:
        agent = FullTrajectoryAgent(checkpoint_path, device=device)
        
        cam_enums = [CameraType[c.upper()] for c in agent.config['camera_names']]
        
        # 数据配置
        data_cfg = DataCollectionConfig(
            image_width=112, image_height=112, 
            use_depth=agent.config['use_depth'],
            camera_types=tuple(cam_enums),
            basket_config=BasketConfig(),
            include_object_relative_pos=False # 坚持不作弊
        )
        
        # [关键] 随机化配置：测评环境固定为“Hard”设定，并额外开启 reset 初始末端绝对位置随机化
        rand_cfg = RandomizationConfig(
            cube_pos_x_range=(0.35, 0.55),
            cube_pos_y_range=(-0.2, 0.2),
            cube_scale_range=(0.03, 0.035),
            basket_pos_x_noise=0.08,  # 显式启用篓子随机化测试鲁棒性
            ee_pos_noise=float(ee_pos_noise),
        )
        
        # 此时 setup_scene 会调用新的 create_basket 逻辑
        collector = VisualExpertDemoCollector(data_config=data_cfg, rand_config=rand_cfg)
        
        success_list = []
        
        iter_range = range(num_episodes)
        if worker_id == 0:
            iter_range = tqdm(iter_range, desc=f"Worker {worker_id}", position=0, leave=True)
            
        for i in iter_range:
            ep_seed = start_seed + i
            agent.reset()
            collector.setup_scene(seed=ep_seed)
            
            recorder = None
            if save_video:
                recorder = VideoRecorder(f"eval_videos_batch/worker_{worker_id}", collector.camera_mgr, 
                                        [CameraType.FRONT_45, CameraType.WRIST])
            
            consecutive_success = 0
            success = False
            
            # 使用稍多的 steps 来应对新增的对齐阶段
            for step in range(max_steps + 50): 
                if recorder:
                    ee = p.getLinkState(collector.panda_id, collector.ee_index, physicsClientId=collector.client_id)
                    recorder.capture(collector.client_id, ee[0], ee[1])
                
                obs = collector.get_obs()
                delta, grip, phase_idx = agent.predict(obs)
                
                delta = np.clip(delta, -0.05, 0.05)
                action = Action(delta, grip)
                collector.execute_action(action, steps=10)
                
                # Success Check (Collector 内的 target_pos 已经是随机化后的真值)
                cube_pos = p.getBasePositionAndOrientation(collector.cube_id, physicsClientId=collector.client_id)[0]
                target = collector.target_pos
                inner = collector.data_config.basket_config.inner_size
                
                in_xy = (abs(cube_pos[0] - target[0]) < inner[0]/2 * 0.9) and \
                        (abs(cube_pos[1] - target[1]) < inner[1]/2 * 0.9)
                in_z = (collector.table_height - inner[2] < cube_pos[2] < collector.table_height)
                j_state = p.getJointState(collector.panda_id, collector.gripper_indices[0], physicsClientId=collector.client_id)[0]
                is_open = j_state > 0.03
                
                if in_xy and in_z and is_open:
                    consecutive_success += 1
                else:
                    consecutive_success = 0
                
                if consecutive_success >= 10:
                    success = True
                    break
            
            if recorder:
                status = "SUCCESS" if success else "FAIL"
                phase_name = agent.phase_map.get(phase_idx, str(phase_idx))
                recorder.save(f"ep_{ep_seed}_{status}_{phase_name}.mp4")
                
            success_list.append(success)
        
        collector.close()
        return success_list
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Worker {worker_id} Error: {e}")
        return []

# ==================== 4. 主控逻辑 ====================

def run_batch_evaluation(args):
    mp.set_start_method('spawn', force=True) 
    
    if not os.path.exists(args.ckpt):
        print(f"Checkpoint not found: {args.ckpt}")
        return

    num_workers = min(args.workers, args.total_episodes)
    episodes_per_worker = args.total_episodes // num_workers
    remainder = args.total_episodes % num_workers
    
    current_seed = args.seed
    
    print(f"\n🚀 开始大规模并行测评 (PID: {os.getpid()})")
    print(f"   Total Episodes: {args.total_episodes}")
    print(f"   Workers: {num_workers}")
    print(f"   Save Video: {args.save_video}")
    print(f"   Device: {args.device}")
    
    pool_args = []
    for i in range(num_workers):
        count = episodes_per_worker + (1 if i < remainder else 0)
        pool_args.append((
            i, 
            args.ckpt, 
            count, 
            current_seed, 
            args.save_video, 
            args.max_steps, 
            args.device,
            args.ee_pos_noise,
        ))
        current_seed += count
    
    start_time = time.time()
    
    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(eval_worker, pool_args)
    
    all_success = []
    for res in results:
        all_success.extend(res)
    
    success_count = sum(all_success)
    total_run = len(all_success)
    rate = (success_count / total_run * 100) if total_run > 0 else 0
    
    duration = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"📊 测评报告 (耗时: {duration:.1f}s)")
    print(f"   总场次: {total_run}")
    print(f"   成功数: {success_count}")
    print(f"   成功率: {rate:.2f}%")
    print(f"{'='*60}")
    
    if args.output_json:
        import json
        with open(args.output_json, 'w') as f:
            json.dump({
                "success_rate": rate,
                "total_episodes": total_run,
                "success_count": int(success_count),
                "ckpt": args.ckpt
            }, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--total_episodes', type=int, default=100, help='Total episodes to run')
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel processes')
    parser.add_argument('--save_video', action='store_true', help='Enable video recording')
    parser.add_argument('--seed', type=int, default=10000, help='Starting random seed')
    parser.add_argument('--max_steps', type=int, default=300, help='Max steps per episode') # 增加步数上限
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--output_json', type=str, default=None, help='Path to save results json')
    parser.add_argument('--ee_pos_noise', type=float, default=0.03, help='Reset initial EE absolute position noise (meters) during evaluation')
    
    args = parser.parse_args()

    
    args = parser.parse_args()
    
    if args.ckpt is None:
        args.ckpt = os.path.join(TRAIN_CONFIG['save_dir'], 'best_policy.pth')
        
    run_batch_evaluation(args)