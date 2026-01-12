import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pickle
import os
import time

# ================= 配置优化 =================
CONFIG = {
    'data_path': 'data/expert_demos.pkl',
    'save_path': 'data/policy_checkpoint.pth',
    
    # [GPU优化 1] 增大 Batch Size
    # GPU 显存很大，对于这种小向量数据，直接开到 512 或 1024 甚至 2048
    # 这能极大提升 GPU 利用率
    'batch_size': 1024,      
    
    'lr': 3e-6,
    'epochs': 400,
    'hidden_dim': 512,      # 加宽网络
    'seed': 42,
    
    # [GPU优化 2] DataLoader 配置
    'num_workers': 4,       # 使用4个CPU核心加载数据 (Windows如果报错改成0)
    'pin_memory': True      # 加速 CPU -> GPU 传输
}

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # 针对 CUDNN 的确定性设置
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True # 开启自动寻找最优算法

# --- 1. 网络结构 (保持不变) ---
class ResBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.activation = nn.Mish()
        
    def forward(self, x):
        return self.activation(x + self.block(x))

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super(ActorNetwork, self).__init__()
        self.input_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish()
        )
        self.res_blocks = nn.Sequential(
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim) 
        )
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.res_blocks(x)
        return self.output_net(x)

# --- 2. 数据集 (保持不变) ---
class RobotDataset(Dataset):
    def __init__(self, obs, acts):
        self.obs = torch.FloatTensor(obs)
        self.acts = torch.FloatTensor(acts)
        
    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, idx):
        return self.obs[idx], self.acts[idx]

# --- 3. 训练流程 (深度优化) ---
def train():
    set_seed(CONFIG['seed'])
    
    # 检查 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Memory Usage: {torch.cuda.memory_allocated(0)/1024**2:.1f}MB")

    # 1. 加载数据
    if not os.path.exists(CONFIG['data_path']):
        print(f"错误: 找不到数据 {CONFIG['data_path']}")
        return

    print("加载数据...")
    with open(CONFIG['data_path'], 'rb') as f:
        data = pickle.load(f)
    
    # 预处理
    stats = {
        'obs_mean': data['obs_mean'], 'obs_std': data['obs_std'],
        'action_mean': data['action_mean'], 'action_std': data['action_std'],
        'obs_dim': data['obs_dim'], 'action_dim': data['action_dim']
    }
    
    obs_norm = (data['observations'] - stats['obs_mean']) / stats['obs_std']
    act_norm = (data['actions'] - stats['action_mean']) / stats['action_std']
    
    # 2. DataLoader 优化
    full_dataset = RobotDataset(obs_norm, act_norm)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # [GPU优化] num_workers 和 pin_memory
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True,
        num_workers=CONFIG['num_workers'], 
        pin_memory=CONFIG['pin_memory'],
        persistent_workers=True if CONFIG['num_workers'] > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False,
        num_workers=CONFIG['num_workers'], 
        pin_memory=CONFIG['pin_memory']
    )
    
    # 3. 模型与编译
    model = ActorNetwork(stats['obs_dim'], stats['action_dim'], CONFIG['hidden_dim']).to(device)
    


    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4) # AdamW通常比Adam好
    criterion = nn.MSELoss()
    
    # [GPU优化 4] 混合精度训练 Scaler
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else torch.amp.GradScaler(enabled=False)

    print(f"开始训练 {CONFIG['epochs']} Epochs...")
    start_time = time.time()
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        # --- 训练 ---
        model.train()
        train_loss = 0.0
        train_count = 0
        
        for batch_obs, batch_act in train_loader:
            # pin_memory=True 后，non_blocking=True 可以实现异步传输
            batch_obs = batch_obs.to(device, non_blocking=True)
            batch_act = batch_act.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # [GPU优化 4] 混合精度前向传播
            # 自动将部分 float32 运算转为 float16，利用 Tensor Cores 加速
            with torch.amp.autocast('cuda'):
                pred_act = model(batch_obs)
                loss = criterion(pred_act, batch_act)
            
            # 混合精度反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            batch_size = batch_obs.size(0)
            train_loss += loss.item() * batch_size
            train_count += batch_size
        
        avg_train_loss = train_loss / max(1, train_count)
        
        # --- 验证 ---
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_obs, batch_act in val_loader:
                batch_obs = batch_obs.to(device, non_blocking=True)
                batch_act = batch_act.to(device, non_blocking=True)
                
                # 验证时不需要Scaler，普通预测即可，也可以加 autocast
                with torch.amp.autocast('cuda'):
                    pred_act = model(batch_obs)
                    loss = criterion(pred_act, batch_act)
                    
                batch_size = batch_obs.size(0)
                val_loss += loss.item() * batch_size
                val_count += batch_size
        
        avg_val_loss = val_loss / max(1, val_count)
        
        # --- 记录与保存 ---
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
                  f"Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | "
                  f"Time: {elapsed:.1f}s")
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 如果用了 torch.compile，保存时可能需要取 .orig_mod 或者直接存 state_dict
            # 这里简单处理，直接存 state_dict 通常兼容
            save_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            
            checkpoint = {
                'model_state_dict': save_model.state_dict(),
                'stats': stats,
                'config': CONFIG
            }
            torch.save(checkpoint, CONFIG['save_path'])

    print(f"\n训练结束! 最佳验证 Loss: {best_val_loss:.6f}")
    print(f"总耗时: {time.time() - start_time:.2f} 秒")

if __name__ == "__main__":
    
    train()