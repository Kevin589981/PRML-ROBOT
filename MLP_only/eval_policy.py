import torch
import numpy as np
import time
import pybullet as p
from data_collector import ExpertDemoCollector, Action, NoiseType
from train_bc import ActorNetwork

def check_object_settled(collector, velocity_threshold=0.05):
    """
    检查物体是否已经稳定（速度足够小且在桌面上）
    """
    cube_pos, _ = p.getBasePositionAndOrientation(collector.cube_id)
    cube_vel, _ = p.getBaseVelocity(collector.cube_id)
    
    linear_vel = np.linalg.norm(cube_vel)
    on_table = collector.table_height - 0.02 < cube_pos[2] < collector.table_height + 0.08
    
    return linear_vel < velocity_threshold and on_table

def wait_for_object_settle(collector, max_wait_steps=120, render=True):
    """
    等待物体落地并稳定
    """
    for _ in range(max_wait_steps):
        p.stepSimulation()
        if render:
            time.sleep(1./240.)
        
        if check_object_settled(collector):
            return True
    
    return check_object_settled(collector)

def run_evaluation(num_episodes=20, render=True):
    CHECKPOINT_PATH = "data/policy_checkpoint.pth"
    
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 {CHECKPOINT_PATH}")
        return

    stats = checkpoint['stats']
    model_state = checkpoint['model_state_dict']
    
    device = torch.device("cpu")
    model = ActorNetwork(stats['obs_dim'], stats['action_dim']).to(device)
    model.load_state_dict(model_state)
    model.eval()
    
    print(f"动作均值: {stats['action_mean']}")
    print(f"动作标准差: {stats['action_std']}")

    collector = ExpertDemoCollector(gui=render, noise_type=NoiseType.NONE)
    
    success_count = 0
    total_steps_list = []
    
    print(f"\n开始评估 {num_episodes} 条轨迹...\n")
    
    for i in range(num_episodes):
        config = collector.setup_scene(randomize=True, seed=i*100)
        
        if config is None:
            print(f"Episode {i+1}: 场景初始化失败，跳过")
            continue
            
        print(f"Episode {i+1}/{num_episodes}: 物体 {np.round(config['cube_initial_pos'], 2)}, 目标 {np.round(config['target_pos'], 2)}")
        
        obs = collector.get_observation()
        step = 0
        max_steps = 500
        
        # ====== 跟踪夹爪状态变化 ======
        gripper_was_closed = False  # 夹爪是否曾经闭合过
        gripper_opened_after_close = False  # 闭合后是否又张开了（表示放置动作）
        release_step = -1  # 记录松开夹爪的步数
        
        while step < max_steps:
            obs_vec = obs.to_vector()
            
            # 归一化
            obs_norm = (obs_vec - stats['obs_mean']) / stats['obs_std']
            obs_tensor = torch.FloatTensor(obs_norm).unsqueeze(0).to(device)
            
            # 推理
            with torch.no_grad():
                pred_act_norm = model(obs_tensor).numpy()[0]
            
            # 反归一化
            pred_act_real = pred_act_norm * stats['action_std'] + stats['action_mean']
            
            # 限制动作范围
            delta_pos = np.clip(pred_act_real[:3], -0.03, 0.03)
            gripper_raw = pred_act_real[3]
            gripper_action = 1.0 if gripper_raw > 0.5 else 0.0
            
            # 调试输出
            if step < 3 or step % 50 == 0:
                print(f"  Step {step}: delta={delta_pos.round(4)}, gripper={gripper_action:.0f}")
            
            # ====== 跟踪夹爪状态 ======
            if gripper_action < 0.5:  # 夹爪闭合
                gripper_was_closed = True
            elif gripper_was_closed and gripper_action > 0.5:  # 之前闭合过，现在张开
                if not gripper_opened_after_close:
                    gripper_opened_after_close = True
                    release_step = step
                    print(f"  Step {step}: 检测到夹爪释放，等待物体稳定...")
            
            action = Action(
                delta_position=delta_pos,
                gripper_action=gripper_action
            )
            
            obs = collector.execute_action(action, steps=5)
            
            if render:
                time.sleep(0.01)
            
            # ====== 只有在夹爪释放后才检查成功 ======
            if gripper_opened_after_close and step > release_step + 10:
                # 等待物体稳定
                if check_object_settled(collector):
                    # 再等一下确保完全稳定
                    print(f"  Step {step}: 物体似乎稳定了，等待确认...")
                    wait_for_object_settle(collector, max_wait_steps=60, render=render)
                    
                    # 最终判断
                    cube_pos, _ = p.getBasePositionAndOrientation(collector.cube_id)
                    cube_pos = np.array(cube_pos)
                    distance = np.linalg.norm(cube_pos[:2] - collector.target_pos[:2])
                    
                    print(f"  最终位置: {cube_pos.round(3)}, 距目标: {distance:.3f}m")
                    
                    if distance < 0.06:
                        print(f"  >>> 成功! (步数: {step})")
                        success_count += 1
                        total_steps_list.append(step)
                    else:
                        print(f"  --- 放置位置偏差过大")
                    break
            
            step += 1
        
        if step >= max_steps:
            print(f"  --- 失败 (超时)")
            
        time.sleep(0.3)

    print("\n" + "="*40)
    print(f"评估完成")
    print(f"成功率: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    if success_count > 0:
        print(f"平均成功步数: {np.mean(total_steps_list):.1f}")
    print("="*40)
    
    collector.close()

if __name__ == "__main__":
    run_evaluation(num_episodes=500, render=False)