# ==========================================
# 将此代码块放在你提供的 train_full_trajectory.py 的末尾
# 或者单独保存，但在同一目录下确保能引用到 FullTrajectoryPolicy 类
# ==========================================
from train_full_trajectory import FullTrajectoryPolicy, CONFIG
import torch
import os
def inspect_checkpoint(ckpt_path, current_model_class=FullTrajectoryPolicy):
    print(f"\n{'='*20} 检测 Checkpoint: {os.path.basename(ckpt_path)} {'='*20}")
    
    if not os.path.exists(ckpt_path):
        print(f"❌ 错误: 文件不存在 -> {ckpt_path}")
        return

    # 1. 加载 Checkpoint
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu',weights_only=False)
    except Exception as e:
        print(f"❌ 无法加载文件: {e}")
        return

    # 2. 分析保存的 Config (如果有)
    saved_config = checkpoint.get('config', None)
    saved_stats = checkpoint.get('stats', None)
    
    print(f"📅 存档对应的 Epoch: {checkpoint.get('epoch', 'Unknown')}")
    
    if saved_config:
        
        print("\n[INFO] 存档时的关键参数 (Saved Config):")
        print(f"  - Hidden Size: {saved_config.get('hidden_size')}")
        print(f"  - Num Layers:  {saved_config.get('num_layers')}")
        print(f"  - Img Size:    {saved_config.get('img_size')}")
        print(f"  - Cameras:     {saved_config.get('camera_names')}")
        print(f"  - Use Depth:   {saved_config.get('use_depth')}")
        for k,v in saved_config.items():
            if k not in ['hidden_size', 'num_layers', 'img_size', 'camera_names', 'use_depth']:
                print(f"  - {k}: {v}")
    else:
        print("\n[WARN] Checkpoint 中没有保存 config 字典。")

    # 3. 准备实例化当前代码定义的模型
    # 我们需要推断 aux_dim 和 num_phases 来初始化模型
    # 如果存档里有 stats，用存档的；否则用假数据，主要为了看架构
    if saved_stats:
        aux_dim = len(saved_stats['aux_mean'])
        num_phases = saved_stats.get('num_phases', 7)
    else:
        print("[WARN] 缺少 stats，尝试使用默认值 aux_dim=6 (猜测), num_phases=7")
        aux_dim = 6 
        num_phases = 7

    # 使用当前的全局 CONFIG 初始化模型
    print("\n[INFO] 正在尝试用当前代码定义实例化模型...")
    try:
        # 注意：这里使用的是你当前脚本顶部的 CONFIG 变量
        current_model = current_model_class(CONFIG, aux_dim, num_phases)
    except Exception as e:
        print(f"❌ 实例化当前模型失败 (可能 CONFIG 字段缺失): {e}")
        return

    current_state = current_model.state_dict()
    saved_state = checkpoint['model']

    # 4. 核心对比逻辑
    mismatches = []
    shape_mismatches = []
    
    current_keys = set(current_state.keys())
    saved_keys = set(saved_state.keys())
    
    # 处理 torch.compile 可能产生的前缀 (如果保存时带有 _orig_mod.)
    saved_keys_clean = {k.replace('_orig_mod.', ''): k for k in saved_keys}
    saved_state_clean = {k.replace('_orig_mod.', ''): v for k, v in saved_state.items()}
    saved_keys = set(saved_state_clean.keys())

    # 4.1 检查层名称是否匹配
    missing_in_saved = current_keys - saved_keys
    extra_in_saved = saved_keys - current_keys
    
    if missing_in_saved:
        mismatches.append(f"❌ 当前代码多出了这些层 (存档里没有): {list(missing_in_saved)[:5]}...")
    if extra_in_saved:
        mismatches.append(f"❌ 存档里多出了这些层 (当前代码没有): {list(extra_in_saved)[:5]}...")

    # 4.2 检查形状是否匹配
    common_keys = current_keys.intersection(saved_keys)
    for key in common_keys:
        s_shape = saved_state_clean[key].shape
        c_shape = current_state[key].shape
        if s_shape != c_shape:
            shape_mismatches.append(f"   - {key}: 存档 {s_shape} vs 当前代码 {c_shape}")

    # 5. 输出结果
    print("\n[RESULT] 对比结果:")
    if not mismatches and not shape_mismatches:
        print("✅ 完美匹配！该 Checkpoint 可以被当前代码加载。")
    else:
        print("⚠️  结构不匹配！详情如下：")
        for m in mismatches:
            print(m)
        if shape_mismatches:
            print("❌ 形状参数不一致 (这通常意味着 hidden_size 或 layer 层数变了):")
            for m in shape_mismatches:
                print(m)
        
        # 尝试给出具体修改建议
        if shape_mismatches:
            print("\n💡 分析建议:")
            for m in shape_mismatches:
                if "encoder.projection" in m:
                    print(f"  -> 视觉特征维度不同。检查 ResNetVisualEncoder 的 feature_dim 参数。")
                if "lstm.weight" in m:
                    print(f"  -> LSTM 维度不同。检查 CONFIG['hidden_size'] 或 ['num_layers']。")
                if "action_head" in m:
                    print(f"  -> 输出头维度不同。可能 hidden_size 变了。")
                if "conv1" in m:
                    print(f"  -> 输入通道不同。检查 ['use_depth'] 或 ['camera_names'] 数量。")
                break # 只打印第一条建议

# ==========================================
# 使用示例：
# ==========================================

if __name__ == "__main__":
    # 假设你想检测这个文件
    target_ckpt = r"./84.8%严苛（训练时无强dropout）checkpoints_full_traj_4090/best_policy.pth" # 修改为你的实际路径
    
    # 如果你有多个文件想批量检测：
    # import glob
    # for f in glob.glob("checkpoints_full_traj_4090/*.pth"):
    #     inspect_checkpoint(f)
    
    inspect_checkpoint(target_ckpt)