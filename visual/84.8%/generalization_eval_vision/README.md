# 84.8% 视觉策略：泛化测试（课程要求版）

目标：以 `84.8%严苛（训练时无强dropout）checkpoints_full_traj_4090/best_policy.pth` 为样板，在“基础成功率”之外，补充一组**简单但可控**的泛化测试，满足课程对“基本成功率实验 + 简单泛化测试”的要求。

本目录会产出：

- 每个条件一个 `*.jsonl`（逐回合记录：seed、success、steps、final distance 等）
- `summary.csv`（按条件汇总：成功率、平均步数、平均距离）
- `plots/success_curves.png`（按“梯度条件”生成折线图）

## 运行方式

从仓库根目录运行（推荐）：

```bash
export PATH="/workspace/xzp/robot/.venv/bin:$PATH"
source .venv/bin/activate

python 84.8%/84.8%train/generalization_eval_vision/run_all.py \
  --checkpoint "84.8%/84.8%严苛（训练时无强dropout）checkpoints_full_traj_4090/best_policy.pth" \
  --episodes 100 \
  --workers 10 \
  --device cuda \
  --plot
```

输出默认在：`84.8%/84.8%train/generalization_eval_vision/runs/<run_id>/`。

折线图默认在：`.../runs/<run_id>/plots/success_curves.png`。

## 条件设计（梯度泛化，更像你原来的 MLP 泛化测试）

- **baseline（基础成功率）**：与 84.8% 测评脚本一致的分布（含 `basket_pos_x_noise=0.08`）。
- **basket_xnoise_***：篓子 X 随机化梯度（0.00/0.04/0.08/0.12/0.16）。
- **cam_pos_noise_***：相机位姿噪声梯度（0.00/0.01/0.02/0.03/0.05）。
- **action_noise_***：动作噪声梯度（0.000/0.001/0.002/0.003/0.005）。
- **cube_friction_***：方块摩擦梯度（0.50/0.75/1.00/1.50/2.00）。
- **cube_range_k***：方块初始分布范围倍数（x1.00/x1.25/x1.50/x1.75/x2.00）。

这些条件足以支撑课程报告中的讨论：哪些变化下性能稳定，哪些变化下失败明显。
