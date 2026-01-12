# train_full_trajectory.py

import os
import h5py
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime

# ==================== 1. 全局配置 (RTX 4090 顶配版) ====================
CONFIG = {
    # 数据路径 (请修改为您实际生成的h5文件路径)
    'dataset_path': 'data_enhanced/basket_demos_with_phase.h5', 
    'save_dir': 'checkpoints_full_traj_4090',
    
    # 训练参数
    'batch_size': 20,     # 全轨迹训练显存消耗大，16是一个平衡点，24G显存可尝试32
    'num_epochs': 200,    # LSTM需要更多epoch收敛
    'lr': 2e-4,           # LSTM对高LR敏感，保持较低
    'weight_decay': 1e-4,
    'seed': 2025,
    
    # 视觉参数
    'img_size': 64,
    'use_depth': True,
    'camera_names': ['front_45', 'wrist'],
    
    # 损失权重
    'pos_loss_weight': 1.0,
    'gripper_loss_weight': 0.5,
    'phase_loss_weight': 0.2, # 辅助任务
    
    # 模型架构
    'hidden_size': 512,   # 强记忆力核心
    'num_layers': 2,
    'dropout': 0.2,
    
    # 数据增强
    'aug_pad': 4,
    'aug_brightness': 0.05,
    'aug_contrast': 0.05,
    
    # 统计修正
    'min_action_std': 0.01, # 防止除以0，保留微小动作
}

# ==================== 2. 核心网络模块 ====================

class SpatialSoftmax(nn.Module):
    """
    将卷积特征图转换为关键点坐标 (B, C*2)
    """
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        # 生成归一化坐标网格 [-1, 1]
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.height),
            np.linspace(-1., 1., self.width)
        )
        self.register_buffer('pos_x', torch.from_numpy(pos_x.reshape(height*width)).float())
        self.register_buffer('pos_y', torch.from_numpy(pos_y.reshape(height*width)).float())

    def forward(self, feature):
        # feature: (B, C, H, W)
        N, C, H, W = feature.shape
        feature_flat = feature.view(N, C, -1)
        attention = F.softmax(feature_flat, dim=-1) # (N, C, H*W)
        
        expected_x = torch.sum(self.pos_x * attention, dim=-1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=-1, keepdim=True)
        
        # (N, C, 2) -> (N, C*2)
        keypoints = torch.cat([expected_x, expected_y], dim=-1).view(N, -1)
        return keypoints


class VisualEncoder(nn.Module):
    """
    基于卷积的视觉编码器，包含SpatialSoftmax
    """
    def __init__(self, input_channels, feature_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Mish(),
            nn.MaxPool2d(2), # 64 -> 32
            
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Mish(),
            nn.MaxPool2d(2), # 32 -> 16
            
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Mish(),
            nn.MaxPool2d(2), # 16 -> 8
            
            # 1x1 Conv 降维
            nn.Conv2d(128, feature_dim, kernel_size=1)
        )
        # 8x8 feature map for Spatial Softmax
        self.spatial_softmax = SpatialSoftmax(8, 8)
        self.output_dim = feature_dim * 2 # x,y coords

    def forward(self, x):
        features = self.net(x)
        keypoints = self.spatial_softmax(features)
        return keypoints


class FullTrajectoryPolicy(nn.Module):
    """
    全轨迹 LSTM 策略网络
    """
    def __init__(self, config, aux_dim, num_phases=7):
        super().__init__()
        self.cameras = config['camera_names']
        self.hidden_size = config['hidden_size']
        
        # 增强参数
        self.aug_pad = config['aug_pad']
        self.aug_bright = config['aug_brightness']
        self.aug_contrast = config['aug_contrast']
        
        # 视觉编码器
        # 输入通道 = (RGB(3) + Depth(1)?) * FrameStack(1)
        in_ch = 4 if config['use_depth'] else 3
        
        # 这里的 feature_dim 指的是每个 channel spatial softmax 之前的通道数
        self.encoder = VisualEncoder(input_channels=in_ch, feature_dim=64)
        
        # 计算 RNN 输入总维度
        # num_cameras * (feature_dim * 2) + aux_dim
        rnn_input_dim = len(self.cameras) * self.encoder.output_dim + aux_dim
        
        # LSTM 核心
        self.lstm = nn.LSTM(
            input_size=rnn_input_dim,
            hidden_size=self.hidden_size,
            num_layers=config['num_layers'],
            batch_first=True,
            dropout=config['dropout'] if config['num_layers'] > 1 else 0
        )
        
        # 动作输出头 (delta_pos:3 + gripper:1)
        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(256, 4) 
        )
        
        # 阶段分类头 (辅助任务)
        self.phase_head = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.LayerNorm(64),
            nn.Mish(),
            nn.Linear(64, num_phases)
        )

    def apply_augmentation(self, x):
        """
        对 Batch*Time 维度统一应用增强
        x: (N_total, C, H, W)
        """
        if not self.training:
            return x
            
        N, C, H, W = x.shape
        
        # 1. Random Shift (Pad & Crop)
        if self.aug_pad > 0:
            x_pad = F.pad(x, (self.aug_pad, self.aug_pad, self.aug_pad, self.aug_pad), mode='replicate')
            # 整个 Batch 随机取一个 crop 位置 (高效且能保持 batch 内一致性，防止过大的显存开销)
            # 如果想更强的增强，可以用 grid_sample，但这里简单切片足够
            eps_x = torch.randint(0, 2 * self.aug_pad + 1, (1,)).item()
            eps_y = torch.randint(0, 2 * self.aug_pad + 1, (1,)).item()
            x = x_pad[:, :, eps_y:eps_y+H, eps_x:eps_x+W]
        
        # 2. Color Jitter (仅对 RGB 通道)
        if self.aug_bright > 0 or self.aug_contrast > 0:
            # 假设前3通道是RGB
            rgb = x[:, :3, :, :]
            rem = x[:, 3:, :, :] if C > 3 else None
            
            # Brightness
            if self.aug_bright > 0:
                noise = torch.rand(N, 1, 1, 1, device=x.device) * 2 * self.aug_bright - self.aug_bright + 1.0
                rgb = rgb * noise
                
            # Contrast
            if self.aug_contrast > 0:
                mean = rgb.mean(dim=(2, 3), keepdim=True)
                noise = torch.rand(N, 1, 1, 1, device=x.device) * 2 * self.aug_contrast - self.aug_contrast + 1.0
                rgb = (rgb - mean) * noise + mean
            
            rgb = torch.clamp(rgb, 0, 1)
            x = torch.cat([rgb, rem], dim=1) if rem is not None else rgb
            
        return x

    def forward(self, images, aux, hidden=None):
        """
        images: (B, T, N_cam, C, H, W)
        aux: (B, T, Aux_dim)
        hidden: (h, c) or None
        """
        B, T, N_cam, C, H, W = images.shape
        
        # 1. 扁平化以进行高效 CNN 处理
        # (B*T*N_cam, C, H, W)
        img_flat = images.view(B * T * N_cam, C, H, W)
        
        # 2. 图像增强
        img_aug = self.apply_augmentation(img_flat)
        
        # 3. 视觉编码
        visual_feat = self.encoder(img_aug) # (B*T*N_cam, D_vis)
        
        # 4. 还原维度并拼接多相机特征
        # (B, T, N_cam * D_vis)
        visual_feat = visual_feat.view(B, T, N_cam * self.encoder.output_dim)
        
        # 5. 拼接辅助状态
        # (B, T, Total_In_Dim)
        rnn_input = torch.cat([visual_feat, aux], dim=-1)
        
        # 6. LSTM 前向传播
        # rnn_out: (B, T, Hidden)
        # new_hidden: (num_layers, B, Hidden) - 包含 h 和 c
        rnn_out, new_hidden = self.lstm(rnn_input, hidden)
        
        # 7. 解码输出
        actions = self.action_head(rnn_out) # (B, T, 4)
        phases = self.phase_head(rnn_out)   # (B, T, Num_Phases)
        
        return actions, phases, new_hidden


# ==================== 3. 强健的数据集加载器 ====================

class WholeTrajectoryDataset(Dataset):
    def __init__(self, h5_path, split='train', camera_names=None):
        super().__init__()
        self.camera_names = camera_names
        
        print(f"[{split.upper()}] Loading dataset from {h5_path}...")
        with h5py.File(h5_path, 'r') as f:
            meta = f['metadata']
            
            # 读取并处理统计数据
            self.stats = {
                'action_mean': meta['action_mean'][:],
                'action_std': np.maximum(meta['action_std'][:], CONFIG['min_action_std']),
                'aux_mean': meta['aux_mean'][:],
                'aux_std': np.maximum(meta['aux_std'][:], 1e-3),
                'depth_min': float(meta.attrs.get('depth_min', 0.0) if 'depth_min' in meta.attrs else meta['depth_min'][()]),
                'depth_max': float(meta.attrs.get('depth_max', 1.0) if 'depth_max' in meta.attrs else meta['depth_max'][()]),
                'num_phases': int(meta.attrs.get('num_phases', 7) if 'num_phases' in meta.attrs else 7)
            }
            
            # 解析相机索引
            all_cams_str = meta.attrs.get('camera_names', '[]')
            if isinstance(all_cams_str, bytes): all_cams_str = all_cams_str.decode('utf-8')
            all_cams = json.loads(all_cams_str)
            
            self.cam_idxs = []
            for c in camera_names:
                if c not in all_cams:
                    raise ValueError(f"Camera {c} not found in dataset {all_cams}")
                self.cam_idxs.append(all_cams.index(c))
            
            # 预加载所有数据到 RAM
            indices_key = 'train_indices' if split == 'train' else 'val_indices'
            indices = meta[indices_key][:]
            self.trajectories = []
            
            for idx in tqdm(indices, desc=f"Loading RAM ({split})"):
                grp = f[f'trajectory_{idx:04d}']
                
                # 动作和辅助
                actions = grp['actions'][:]
                aux = grp['aux'][:]
                
                # 相机数据 (T, N_all, H, W, 3) -> 选择相机 -> (T, N_sel, H, W, 3)
                rgb = grp['rgb'][:][:, self.cam_idxs]
                depth = grp['depth'][:][:, self.cam_idxs] if 'depth' in grp else None
                
                # 阶段标签 (如果不存在则造假，防止代码崩)
                if 'phase_labels' in grp:
                    phase = grp['phase_labels'][:]
                else:
                    phase = np.zeros(len(actions), dtype=np.int64) 
                
                self.trajectories.append({
                    'rgb': rgb,
                    'depth': depth,
                    'actions': actions,
                    'aux': aux,
                    'phase': phase
                })
                
        print(f"[{split.upper()}] Loaded {len(self.trajectories)} trajectories.")

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]


def collate_fn_pad(batch):
    """
    处理变长序列的 Padding 和 Masking
    """
    # 1. 确定当前 batch 的最大长度
    lengths = [b['actions'].shape[0] for b in batch]
    max_len = max(lengths)
    
    # 容器
    batch_rgb = []
    batch_depth = []
    batch_actions = []
    batch_aux = []
    batch_phase = []
    batch_mask = []
    
    for b in batch:
        cur_len = b['actions'].shape[0]
        pad_len = max_len - cur_len
        
        # === Create Mask (1=Valid, 0=Pad) ===
        mask = torch.ones(cur_len, dtype=torch.float32)
        if pad_len > 0:
            mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.float32)])
        batch_mask.append(mask)
        
        # === RGB: (T, N, H, W, 3) ===
        # 注意: 这里先保持 numpy/tensor 转换，padding 0
        rgb = torch.from_numpy(b['rgb'])
        if pad_len > 0:
            # shape: (Pad, N, H, W, 3)
            rgb_pad = torch.zeros((pad_len, *rgb.shape[1:]), dtype=rgb.dtype)
            rgb = torch.cat([rgb, rgb_pad], dim=0)
        batch_rgb.append(rgb)
        
        # === Depth ===
        if b['depth'] is not None:
            d = torch.from_numpy(b['depth'])
            if pad_len > 0:
                d_pad = torch.zeros((pad_len, *d.shape[1:]), dtype=d.dtype)
                d = torch.cat([d, d_pad], dim=0)
            batch_depth.append(d)
            
        # === Actions: (T, 4) ===
        act = torch.from_numpy(b['actions'])
        if pad_len > 0:
            act = torch.cat([act, torch.zeros((pad_len, 4), dtype=act.dtype)])
        batch_actions.append(act)
        
        # === Aux: (T, D) ===
        aux = torch.from_numpy(b['aux'])
        if pad_len > 0:
            aux = torch.cat([aux, torch.zeros((pad_len, aux.shape[1]), dtype=aux.dtype)])
        batch_aux.append(aux)
        
        # === Phase: (T,) ===
        ph = torch.from_numpy(b['phase']).long()
        if pad_len > 0:
            # 填充 -100，这是 CrossEntropyLoss 的默认 ignore_index
            ph = torch.cat([ph, torch.full((pad_len,), -100, dtype=torch.long)])
        batch_phase.append(ph)
        
    # Stack output
    out = {
        'mask': torch.stack(batch_mask), # (B, T)
        'rgb': torch.stack(batch_rgb),   # (B, T, N, H, W, 3)
        'actions': torch.stack(batch_actions),
        'aux': torch.stack(batch_aux),
        'phase': torch.stack(batch_phase)
    }
    if batch_depth:
        out['depth'] = torch.stack(batch_depth)
    else:
        out['depth'] = None
        
    return out


# ==================== 4. 训练主循环 (不省略任何细节) ====================

def train():
    # 0. 环境设置
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True # 加速卷积
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    # 1. 准备数据
    train_ds = WholeTrajectoryDataset(CONFIG['dataset_path'], 'train', CONFIG['camera_names'])
    val_ds = WholeTrajectoryDataset(CONFIG['dataset_path'], 'val', CONFIG['camera_names'])
    
    # 将统计数据转为 Tensor 并存入 GPU 以加速归一化
    stats = {
        k: torch.from_numpy(v).float().to(device) if isinstance(v, np.ndarray) else v 
        for k, v in train_ds.stats.items()
    }
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, 
                              num_workers=2, collate_fn=collate_fn_pad, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, 
                            num_workers=4, collate_fn=collate_fn_pad, pin_memory=True)
    
    # 2. 初始化模型
    model = FullTrajectoryPolicy(CONFIG, aux_dim=len(stats['aux_mean']), num_phases=stats['num_phases']).to(device)
    
    # 编译模型 (PyTorch 2.0+, 4090 必开)
    try:
        model = torch.compile(model)
        print("Model compiled with torch.compile()!")
    except Exception as e:
        print(f"Warning: torch.compile failed: {e}")

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scaler = torch.amp.GradScaler('cuda') # 混合精度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'], eta_min=1e-5)
    
    # Loss Functions
    huber_loss = nn.SmoothL1Loss(beta=0.1, reduction='none') # 不规约，为了手动乘 mask
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    print(f"START TRAINING: {len(train_ds)} trajs, {CONFIG['num_epochs']} epochs, Batch={CONFIG['batch_size']}")
    
    for epoch in range(CONFIG['num_epochs']):
        # === Training ===
        model.train()
        train_loss_acc = 0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{CONFIG['num_epochs']}")
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            
            # --- Data Transfer & Norm ---
            mask = batch['mask'].to(device) # (B, T)
            valid_tokens = mask.sum()
            
            # Prepare Images: (B, T, N, H, W, 3) -> (B, T, N, C, H, W)
            rgb = batch['rgb'].to(device).permute(0, 1, 2, 5, 3, 4).float() / 255.0
            
            if batch['depth'] is not None:
                d = batch['depth'].to(device).permute(0, 1, 2, 5, 3, 4).float()
                # Depth Norm
                d = (d - stats['depth_min']) / (stats['depth_max'] - stats['depth_min'] + 1e-6)
                d = torch.clamp(d, 0, 1)
                images = torch.cat([rgb, d], dim=3) # Concat on Channel dim
            else:
                images = rgb
            
            # Aux Norm
            aux = (batch['aux'].to(device) - stats['aux_mean']) / stats['aux_std']
            
            # Targets
            gt_act = batch['actions'].to(device)
            target_act = (gt_act - stats['action_mean']) / stats['action_std']
            target_phase = batch['phase'].to(device)
            
            # --- Forward ---
            with torch.amp.autocast('cuda'):
                # hidden=None: LSTM starts with zero state for every new trajectory in batch
                pred_act, pred_phase, _ = model(images, aux, hidden=None)
                
                # --- Loss Calculation ---
                # 1. Action Position (Huber)
                # mask shape (B, T) -> expand to (B, T, 3)
                loss_pos_elem = huber_loss(pred_act[..., :3], target_act[..., :3])
                loss_pos = (loss_pos_elem * mask.unsqueeze(-1)).sum() / (valid_tokens * 3 + 1e-6)
                
                # 2. Gripper (BCE)
                # gt is 0/1, pred is logit
                gt_grip = (gt_act[..., 3] > 0.5).float()
                loss_grip_elem = bce_loss(pred_act[..., 3], gt_grip)
                loss_grip = (loss_grip_elem * mask).sum() / (valid_tokens + 1e-6)
                
                # 3. Phase (CE)
                # Flatten (B*T, NumClasses) vs (B*T)
                # ce_loss handles ignore_index=-100 inside, but reduction='none' returns (B*T)
                # We need to manually average over valid tokens to be safe
                loss_phase_elem = ce_loss(pred_phase.reshape(-1, stats['num_phases']), target_phase.view(-1))
                # Note: target_phase has -100 for pads. loss_phase_elem will be 0 at those indices if ignore_index works
                # But explicitly:
                valid_mask_flat = mask.view(-1)
                loss_phase = (loss_phase_elem * valid_mask_flat).sum() / (valid_mask_flat.sum() + 1e-6)
                
                total_loss = (CONFIG['pos_loss_weight'] * loss_pos + 
                              CONFIG['gripper_loss_weight'] * loss_grip + 
                              CONFIG['phase_loss_weight'] * loss_phase)
            
            # --- Backward ---
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient Clipping
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_acc += total_loss.item()
            train_batches += 1
            pbar.set_postfix({'L': f"{total_loss.item():.4f}", 'P': f"{loss_phase.item():.3f}"})
            
        avg_train_loss = train_loss_acc / train_batches
        history['train_loss'].append(avg_train_loss)
        
        # === Validation ===
        model.eval()
        val_loss_acc = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                mask = batch['mask'].to(device)
                valid_tokens = mask.sum()
                
                # Preprocess (Same as train but no aug)
                rgb = batch['rgb'].to(device).permute(0, 1, 2, 5, 3, 4).float() / 255.0
                if batch['depth'] is not None:
                    d = batch['depth'].to(device).permute(0, 1, 2, 5, 3, 4).float()
                    d = (d - stats['depth_min']) / (stats['depth_max'] - stats['depth_min'] + 1e-6)
                    images = torch.cat([rgb, d], dim=3)
                else: images = rgb
                aux = (batch['aux'].to(device) - stats['aux_mean']) / stats['aux_std']
                gt_act = batch['actions'].to(device)
                target_act = (gt_act - stats['action_mean']) / stats['action_std']
                target_phase = batch['phase'].to(device)
                
                with torch.amp.autocast('cuda'):
                    pred_act, pred_phase, _ = model(images, aux, hidden=None)
                    
                    # Losses
                    l_pos = (huber_loss(pred_act[..., :3], target_act[..., :3]) * mask.unsqueeze(-1)).sum() / (valid_tokens*3+1e-6)
                    l_grip = (bce_loss(pred_act[..., 3], (gt_act[..., 3]>0.5).float()) * mask).sum() / (valid_tokens+1e-6)
                    l_ph_elem = ce_loss(pred_phase.reshape(-1, stats['num_phases']), target_phase.view(-1))
                    l_ph = (l_ph_elem * mask.view(-1)).sum() / (valid_tokens+1e-6)
                    
                    val_loss = (CONFIG['pos_loss_weight'] * l_pos + 
                                CONFIG['gripper_loss_weight'] * l_grip + 
                                CONFIG['phase_loss_weight'] * l_ph)
                
                val_loss_acc += val_loss.item()
                val_batches += 1
                
        avg_val_loss = val_loss_acc / val_batches
        history['val_loss'].append(avg_val_loss)
        scheduler.step()
        
        print(f"Ep {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
        
        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(CONFIG['save_dir'], 'best_policy.pth')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'config': CONFIG,
                'stats': {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in stats.items()}
            }, save_path)
            print(f"  --> Saved Best Model: {save_path}")
            
        # Periodic Save
        if (epoch + 1) % 10 == 0:
            torch.save({
                'model': model.state_dict(),
                'config': CONFIG,
                'stats': {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in stats.items()}
            }, os.path.join(CONFIG['save_dir'], f'policy_ep{epoch+1}.pth'))

    # Final Plot
    plt.figure()
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title("Full Trajectory Training Loss")
    plt.legend()
    plt.savefig(os.path.join(CONFIG['save_dir'], 'loss_curve.png'))
    print("Training Complete.")

if __name__ == "__main__":
    train()