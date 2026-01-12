# diagnose_training.py
"""
Training Diagnosis Tool
训练问题诊断工具
"""

import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib
import os

# 设置支持中文的字体
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def diagnose_dataset(h5_path):
    """诊断数据集统计信息"""
    print("="*60)
    print("数据集诊断")
    print("="*60)
    
    with h5py.File(h5_path, 'r') as f:
        # 读取统计信息
        action_mean = f['metadata']['action_mean'][:]
        action_std = f['metadata']['action_std'][:]
        
        print("\n1. 动作归一化参数:")
        print(f"   Mean: {action_mean}")
        print(f"   Std:  {action_std}")
        
        # 检查是否有异常小的std（会导致归一化后梯度爆炸）
        if np.any(action_std < 0.001):
            print("   ⚠️  警告: Std过小！会导致归一化后数值过大")
        
        # 收集所有动作
        train_indices = f['metadata']['train_indices'][:]
        all_actions = []
        all_grippers = []
        
        for idx in train_indices[:100]:  # 采样100条分析
            actions = f[f'trajectory_{idx:04d}']['actions'][:]
            all_actions.append(actions[:, :3])  # xyz
            all_grippers.append(actions[:, 3])   # gripper
        
        all_actions = np.concatenate(all_actions)
        all_grippers = np.concatenate(all_grippers)
        
        print("\n2. 动作统计 (原始值):")
        print(f"   X: [{all_actions[:, 0].min():.4f}, {all_actions[:, 0].max():.4f}] "
              f"mean={all_actions[:, 0].mean():.4f} std={all_actions[:, 0].std():.4f}")
        print(f"   Y: [{all_actions[:, 1].min():.4f}, {all_actions[:, 1].max():.4f}] "
              f"mean={all_actions[:, 1].mean():.4f} std={all_actions[:, 1].std():.4f}")
        print(f"   Z: [{all_actions[:, 2].min():.4f}, {all_actions[:, 2].max():.4f}] "
              f"mean={all_actions[:, 2].mean():.4f} std={all_actions[:, 2].std():.4f}")
        
        print("\n3. Gripper 分布:")
        unique, counts = np.unique(all_grippers, return_counts=True)
        for val, cnt in zip(unique, counts):
            print(f"   值 {val:.2f}: {cnt} 次 ({cnt/len(all_grippers)*100:.1f}%)")
        
        # 检查gripper是否只有0/1
        if len(unique) > 10:
            print(f"   ✓ Gripper是连续值 (0→1平滑过渡)")
        else:
            print(f"   ⚠️  Gripper只有{len(unique)}个离散值")
        
        # 归一化后的分布
        actions_norm = (all_actions - action_mean[:3]) / action_std[:3]
        print("\n4. 归一化后的动作范围:")
        print(f"   X_norm: [{actions_norm[:, 0].min():.2f}, {actions_norm[:, 0].max():.2f}]")
        print(f"   Y_norm: [{actions_norm[:, 1].min():.2f}, {actions_norm[:, 1].max():.2f}]")
        print(f"   Z_norm: [{actions_norm[:, 2].min():.2f}, {actions_norm[:, 2].max():.2f}]")
        
        if np.abs(actions_norm).max() > 10:
            print("   ⚠️  归一化后数值过大！会导致训练不稳定")
        
        # 可视化 - 使用英文标签
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 原始动作分布
        axes[0, 0].hist(all_actions[:, 0], bins=50, alpha=0.7, label='X')
        axes[0, 0].hist(all_actions[:, 1], bins=50, alpha=0.7, label='Y')
        axes[0, 0].hist(all_actions[:, 2], bins=50, alpha=0.7, label='Z')
        axes[0, 0].set_title('Original Action Distribution')
        axes[0, 0].set_xlabel('Action Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # 归一化后分布
        axes[0, 1].hist(actions_norm[:, 0], bins=50, alpha=0.7, label='X_norm')
        axes[0, 1].hist(actions_norm[:, 1], bins=50, alpha=0.7, label='Y_norm')
        axes[0, 1].hist(actions_norm[:, 2], bins=50, alpha=0.7, label='Z_norm')
        axes[0, 1].set_title('Normalized Action Distribution')
        axes[0, 1].set_xlabel('Normalized Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Gripper分布
        axes[1, 0].hist(all_grippers, bins=30)
        axes[1, 0].set_title('Gripper Action Distribution')
        axes[1, 0].set_xlabel('Gripper Value')
        axes[1, 0].set_ylabel('Frequency')
        
        # 动作幅度随时间
        sample_traj = f['trajectory_0000']['actions'][:]
        axes[1, 1].plot(np.linalg.norm(sample_traj[:, :3], axis=1), label='Position Delta')
        axes[1, 1].plot(sample_traj[:, 3], label='Gripper', alpha=0.7)
        axes[1, 1].set_title('Sample Trajectory: Action Magnitude')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('diagnosis_data.png', dpi=150)
        print(f"\n✓ 可视化已保存: diagnosis_data.png")


def estimate_loss_scale(h5_path):
    """估算不同loss项的数值范围"""
    print("\n" + "="*60)
    print("Loss Scale 估算")
    print("="*60)
    
    with h5py.File(h5_path, 'r') as f:
        action_mean = f['metadata']['action_mean'][:]
        action_std = f['metadata']['action_std'][:]
        
        train_indices = f['metadata']['train_indices'][:]
        sample_actions = []
        
        for idx in train_indices[:50]:
            actions = f[f'trajectory_{idx:04d}']['actions'][:]
            sample_actions.append(actions)
        
        sample_actions = np.concatenate(sample_actions)
        
        # 模拟归一化
        pos_raw = sample_actions[:, :3]
        pos_norm = (pos_raw - action_mean[:3]) / action_std[:3]
        
        # 模拟MSE loss (假设预测完全随机)
        random_pred = np.random.randn(*pos_norm.shape)
        mse_pos = np.mean((pos_norm - random_pred) ** 2)
        mse_z = np.mean((pos_norm[:, 2] - random_pred[:, 2]) ** 2)
        
        # 模拟BCE loss (gripper)
        gripper_raw = sample_actions[:, 3]
        gripper_binary = (gripper_raw > 0.5).astype(float)
        random_logits = np.random.randn(len(gripper_binary)) * 2
        bce_gripper = -np.mean(
            gripper_binary * np.log(1 / (1 + np.exp(-random_logits)) + 1e-8) +
            (1 - gripper_binary) * np.log(1 - 1 / (1 + np.exp(-random_logits)) + 1e-8)
        )
        
        print(f"\n随机初始化时的预期Loss:")
        print(f"  MSE(position):  {mse_pos:.4f}")
        print(f"  MSE(z-axis):    {mse_z:.4f}")
        print(f"  BCE(gripper):   {bce_gripper:.4f}")
        
        # 当前配置的总loss
        total = 1.0 * mse_pos + 2.0 * mse_z + 1.0 * bce_gripper
        print(f"\n当前权重配置 (pos=1.0, z=2.0, grip=1.0):")
        print(f"  预期总Loss: {total:.4f}")
        print(f"  各项占比: pos={1.0*mse_pos/total*100:.1f}%, "
              f"z={2.0*mse_z/total*100:.1f}%, grip={1.0*bce_gripper/total*100:.1f}%")
        
        if 2.0 * mse_z / total > 0.6:
            print(f"\n  ⚠️  Z轴loss占比过高！可能导致过度关注Z轴")
        
        # 建议权重
        pos_weight = 1.0
        z_weight = mse_pos / (mse_z + 1e-8) * 0.5  # Z轴占总position loss的50%
        grip_weight = mse_pos / (bce_gripper + 1e-8) * 0.3  # Gripper占30%
        
        print(f"\n建议权重配置:")
        print(f"  'pos_loss_weight': {pos_weight:.2f}")
        print(f"  'z_axis_loss_weight': {z_weight:.2f}")
        print(f"  'gripper_loss_weight': {grip_weight:.2f}")


def check_gradient_flow(model_path):
    """检查已训练模型的梯度流"""
    if not os.path.exists(model_path):
        print(f"\n模型文件不存在: {model_path}")
        return
    
    print("\n" + "="*60)
    print("模型检查")
    print("="*60)
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    print(f"\n训练轮数: {checkpoint.get('epoch', 'unknown')}")
    
    # 检查权重分布
    state_dict = checkpoint['model']
    
    print("\n关键层权重统计:")
    for name, param in state_dict.items():
        if 'weight' in name and len(param.shape) >= 2:
            mean = param.abs().mean().item()
            std = param.std().item()
            print(f"  {name:40s}: mean={mean:.6f}, std={std:.6f}")
            
            if mean < 1e-4:
                print(f"    ⚠️  权重接近0，可能梯度消失")
            if std > 10:
                print(f"    ⚠️  权重方差过大，可能梯度爆炸")


if __name__ == "__main__":
    import sys
    
    # 数据集路径
    h5_path = "data/basket_demos_dense_temporal.h5"
    model_path = "checkpoints_temporal_gru/best_policy.pth"
    
    if not os.path.exists(h5_path):
        print(f"错误: 数据集不存在 {h5_path}")
        sys.exit(1)
    
    # 运行诊断
    diagnose_dataset(h5_path)
    estimate_loss_scale(h5_path)
    check_gradient_flow(model_path)
    
    print("\n" + "="*60)
    print("诊断完成！请查看上述输出和 diagnosis_data.png")
    print("="*60)