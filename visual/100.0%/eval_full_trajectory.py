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

# ==================== 1. 视频录制工具 ====================

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
        num_phases = int(self.stats['num_phases'].item()) if 'num_phases' in self.stats else 7
        
        self.model = FullTrajectoryPolicy(self.config, aux_dim, num_phases)
        
        state_dict = ckpt['model']
        new_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "")
            new_dict[new_key] = v
            
        self.model.load_state_dict(new_dict)
        self.model.to(device).eval()
        self.hidden = None
        
        self.phase_map = {
            0: "Approaching", 1: "Grasping", 2: "Lifting",
            3: "Moving", 4: "Descending", 5: "Releasing", 6: "Retreating"
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

def eval_worker(worker_id, checkpoint_path, num_episodes, start_seed, save_video, max_steps, device):
    """
    单个 Worker 进程，负责跑 num_episodes 轮
    """
    # 每个进程需要独立的 Agent 和 Environment
    try:
        agent = FullTrajectoryAgent(checkpoint_path, device=device)
        
        cam_enums = [CameraType[c.upper()] for c in agent.config['camera_names']]
        data_cfg = DataCollectionConfig(
            image_width=64, image_height=64,
            use_depth=agent.config['use_depth'],
            camera_types=tuple(cam_enums),
            basket_config=BasketConfig()
        )
        rand_cfg = RandomizationConfig(
            cube_pos_x_range=(0.35, 0.55),
            cube_pos_y_range=(-0.2, 0.2),
            cube_scale_range=(0.03, 0.035)
        )
        
        collector = VisualExpertDemoCollector(data_config=data_cfg, rand_config=rand_cfg)
        
        success_list = []
        
        # 进度条只在 Worker 0 显示，或者保持静默
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
            
            for step in range(max_steps):
                if recorder:
                    ee = p.getLinkState(collector.panda_id, collector.ee_index, physicsClientId=collector.client_id)
                    recorder.capture(collector.client_id, ee[0], ee[1])
                
                obs = collector.get_obs()
                delta, grip, phase_idx = agent.predict(obs)
                
                delta = np.clip(delta, -0.05, 0.05)
                action = Action(delta, grip)
                collector.execute_action(action, steps=10)
                
                # Success Check
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
        print(f"Worker {worker_id} Error: {e}")
        return []

# ==================== 4. 主控逻辑 ====================

def run_batch_evaluation(args):
    mp.set_start_method('spawn', force=True) # 必须用 spawn 以兼容 CUDA
    
    if not os.path.exists(args.ckpt):
        print(f"Checkpoint not found: {args.ckpt}")
        return

    num_workers = min(args.workers, args.total_episodes)
    episodes_per_worker = args.total_episodes // num_workers
    
    # 剩余的任务分配给前几个worker
    remainder = args.total_episodes % num_workers
    
    tasks = []
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
            args.device
        ))
        current_seed += count
    
    start_time = time.time()
    
    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(eval_worker, pool_args)
    
    # 汇总结果
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--total_episodes', type=int, default=500, help='Total episodes to run')
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel processes')
    parser.add_argument('--save_video', action='store_true', help='Enable video recording')
    parser.add_argument('--seed', type=int, default=10000, help='Starting random seed')
    parser.add_argument('--max_steps', type=int, default=200, help='Max steps per episode')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    
    args = parser.parse_args()
    
    # 自动寻找默认模型
    if args.ckpt is None:
        args.ckpt = os.path.join(TRAIN_CONFIG['save_dir'], 'best_policy.pth')
        
    run_batch_evaluation(args)

# python eval_full_trajectory.py --total_episodes 500 --workers 100