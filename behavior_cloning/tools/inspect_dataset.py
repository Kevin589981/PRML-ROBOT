# visualize_dataset.py
"""
数据集可视化工具：将第一条轨迹转为时间序列图像
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import os
import json

def visualize_trajectory(h5_path, traj_idx=0, output_dir='visualization'):
    """
    将指定轨迹的所有帧可视化为图像序列
    
    参数:
        h5_path: 数据集路径
        traj_idx: 要可视化的轨迹索引
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(h5_path, 'r') as f:
        # 读取元数据
        camera_names = json.loads(f['metadata'].attrs['camera_names'])
        
        # 读取第一条轨迹
        traj_key = f'trajectory_{traj_idx:04d}'
        if traj_key not in f:
            print(f"错误: {traj_key} 不存在")
            return
            
        grp = f[traj_key]
        rgb = grp['rgb'][:]           # (T, N_cam, H, W, 3)
        actions = grp['actions'][:]   # (T, 4)
        length = grp.attrs['length']
        
        print(f"{'='*60}")
        print(f"可视化轨迹: {traj_key}")
        print(f"  长度: {length} 帧")
        print(f"  相机: {camera_names}")
        print(f"  图像尺寸: {rgb.shape[2]}x{rgb.shape[3]}")
        print(f"{'='*60}\n")
        
        # 为每一帧生成可视化
        for t in range(length):
            fig = plt.figure(figsize=(16, 10))
            gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
            
            # 顶部：四个相机视图
            for i, cam_name in enumerate(camera_names):
                ax = fig.add_subplot(gs[0:2, i])
                img = rgb[t, i]  # (H, W, 3)
                ax.imshow(img)
                ax.set_title(f'{cam_name.upper()}\nFrame {t}/{length-1}', 
                           fontsize=10, fontweight='bold')
                ax.axis('off')
                
                # 添加边框指示关键阶段
                color = get_stage_color(t, length)
                rect = patches.Rectangle((0, 0), img.shape[1]-1, img.shape[0]-1,
                                        linewidth=3, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
            
            # 底部：动作信息
            ax_info = fig.add_subplot(gs[2, :])
            ax_info.axis('off')
            
            action = actions[t]
            dx, dy, dz, gripper = action
            
            stage_name = get_stage_name(t, length)
            
            info_text = f"""
STAGE: {stage_name}
            
Action Vector:
  Δx = {dx:+.4f} m
  Δy = {dy:+.4f} m  
  Δz = {dz:+.4f} m
  Gripper = {gripper:.2f} ({'OPEN' if gripper > 0.5 else 'CLOSED'})
  
Movement Magnitude: {np.linalg.norm([dx, dy, dz]):.4f} m
            """
            
            ax_info.text(0.05, 0.5, info_text, 
                        fontsize=11, family='monospace',
                        verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # 绘制动作向量可视化
            ax_vec = fig.add_subplot(gs[2, 2:], projection='3d')
            ax_vec.quiver(0, 0, 0, dx, dy, dz, color='red', arrow_length_ratio=0.3, linewidth=2)
            ax_vec.set_xlim([-0.05, 0.05])
            ax_vec.set_ylim([-0.05, 0.05])
            ax_vec.set_zlim([-0.05, 0.05])
            ax_vec.set_xlabel('X')
            ax_vec.set_ylabel('Y')
            ax_vec.set_zlabel('Z')
            ax_vec.set_title('Action Vector', fontsize=10)
            
            # 保存
            filename = f'frame_{t:03d}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
            
            if (t+1) % 10 == 0:
                print(f"  已生成: {t+1}/{length} 帧")
        
        print(f"\n✓ 完成! 图像已保存到: {output_dir}/")
        print(f"  总计: {length} 张图片")
        
        # 生成一个大的拼接图（显示关键帧）
        create_summary_grid(rgb, actions, camera_names, length, output_dir)


def get_stage_name(frame, total):
    """根据帧数推断当前阶段"""
    progress = frame / total
    if progress < 0.20:
        return "🔵 APPROACH"
    elif progress < 0.35:
        return "🟡 DESCEND"
    elif progress < 0.50:
        return "🟢 GRASP"
    elif progress < 0.65:
        return "🟣 LIFT"
    elif progress < 0.75:
        return "🟠 TRANSFER"
    elif progress < 0.88:
        return "🔴 PLACE"
    else:
        return "⚪ RETREAT"


def get_stage_color(frame, total):
    """为不同阶段返回颜色"""
    progress = frame / total
    if progress < 0.20:
        return 'blue'
    elif progress < 0.35:
        return 'yellow'
    elif progress < 0.50:
        return 'green'
    elif progress < 0.65:
        return 'purple'
    elif progress < 0.75:
        return 'orange'
    elif progress < 0.88:
        return 'red'
    else:
        return 'gray'


def create_summary_grid(rgb, actions, camera_names, length, output_dir):
    """生成关键帧汇总图"""
    # 选择8个关键帧
    key_frames = np.linspace(0, length-1, 8, dtype=int)
    
    fig, axes = plt.subplots(len(camera_names), len(key_frames), 
                            figsize=(20, 10))
    
    for row, cam_name in enumerate(camera_names):
        for col, frame_idx in enumerate(key_frames):
            ax = axes[row, col]
            img = rgb[frame_idx, row]
            ax.imshow(img)
            
            if row == 0:
                stage = get_stage_name(frame_idx, length)
                ax.set_title(f'F{frame_idx}\n{stage}', fontsize=8)
            
            if col == 0:
                ax.set_ylabel(cam_name.upper(), fontsize=10, fontweight='bold')
            
            ax.axis('off')
    
    plt.suptitle(f'Trajectory Summary: {length} Frames', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_keyframes.png'), dpi=150)
    plt.close()
    
    print(f"  ✓ 关键帧汇总图: summary_keyframes.png")


def create_video_from_frames(output_dir, fps=10):
    """
    可选：使用 ffmpeg 将图像序列转为视频
    需要安装: pip install imageio[ffmpeg]
    """
    try:
        import imageio
        images = []
        frame_files = sorted([f for f in os.listdir(output_dir) if f.startswith('frame_')])
        
        print(f"\n生成视频...")
        for filename in frame_files:
            images.append(imageio.imread(os.path.join(output_dir, filename)))
        
        video_path = os.path.join(output_dir, 'trajectory_animation.mp4')
        imageio.mimsave(video_path, images, fps=fps)
        print(f"✓ 视频已保存: {video_path}")
        
    except ImportError:
        print("\n提示: 安装 imageio 可生成视频:")
        print("  pip install imageio[ffmpeg]")


if __name__ == "__main__":
    import sys
    
    # 使用方法
    if len(sys.argv) > 1:
        h5_path = sys.argv[1]
    else:
        h5_path = "data/basket_demos_dense_temporal.h5"
    
    if not os.path.exists(h5_path):
        print(f"错误: 文件不存在 {h5_path}")
        print("用法: python visualize_dataset.py <数据集路径>")
        sys.exit(1)
    
    # 可视化第一条轨迹
    visualize_trajectory(h5_path, traj_idx=0, output_dir='visualization/traj_000')
    
    # 尝试生成视频
    create_video_from_frames('visualization/traj_000', fps=10)
    
    print("\n" + "="*60)
    print("可视化完成！")
    print("  查看单帧: visualization/traj_000/frame_XXX.png")
    print("  查看汇总: visualization/traj_000/summary_keyframes.png")
    print("="*60)