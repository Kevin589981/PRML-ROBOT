# Generalization Evaluation

本目录用于“控制变量”的泛化测评（每次只改一个变量，默认每个条件 100 轮）。

## 目录结构

- `generalization_eval/shapes/`：初始目标形状泛化（球体/长方体/圆柱/三棱柱等）
- `generalization_eval/heights/`：初始点海拔 / 目标点海拔 泛化
- `generalization_eval/base/`：机械臂基座初始位置泛化（基座 x/y/z 坐标偏移）
- `generalization_eval/extra/`：其他可行的控制变量泛化（姿态/质量/摩擦/弹性/重力/仿真参数/噪声延迟/外力/目标偏移/障碍物）
- `generalization_eval/analysis/`：结果汇总
- `generalization_eval/assets/`：mesh 资源（如三棱柱）

## 运行方式

从仓库根目录执行（也就是 `data_collector.py` 所在目录）：

```bash
conda activate robot

python generalization_eval/shapes/eval_shapes.py --num_trials 100
python generalization_eval/heights/eval_initial_elevation.py --num_trials 100
python generalization_eval/heights/eval_target_elevation.py --num_trials 100
python generalization_eval/base/eval_base_x.py --num_trials 100
python generalization_eval/base/eval_base_y.py --num_trials 100
python generalization_eval/base/eval_base_z.py --num_trials 100

# extra
python generalization_eval/extra/pose/eval_initial_pose.py --num_trials 100
python generalization_eval/extra/physics/eval_mass.py --num_trials 100
python generalization_eval/extra/physics/eval_friction.py --num_trials 100
python generalization_eval/extra/physics/eval_restitution.py --num_trials 100
python generalization_eval/extra/physics/eval_object_scale.py --num_trials 100
python generalization_eval/extra/sim/eval_gravity.py --num_trials 100
python generalization_eval/extra/sim/eval_solver_iterations.py --num_trials 100
python generalization_eval/extra/sim/eval_timestep.py --num_trials 100
python generalization_eval/extra/noise/eval_obs_noise.py --num_trials 100
python generalization_eval/extra/noise/eval_obs_delay_drop.py --num_trials 100
python generalization_eval/extra/noise/eval_action_noise_delay.py --num_trials 100
python generalization_eval/extra/disturbance/eval_external_push.py --num_trials 100
python generalization_eval/extra/target/eval_target_xy_offset.py --num_trials 100
python generalization_eval/extra/obstacles/eval_obstacle_box.py --num_trials 100
```

说明：`base_z` 默认只测 0/1/2/5cm（更稳定），如需测试 10/20cm：

```bash
python generalization_eval/base/eval_base_z.py --num_trials 100 --include_large
```

如果你不想在当前 shell 里 `conda activate`，也可以用：

```bash
conda run -n robot python generalization_eval/shapes/eval_shapes.py --num_trials 100
conda run -n robot python generalization_eval/heights/eval_initial_elevation.py --num_trials 100
conda run -n robot python generalization_eval/heights/eval_target_elevation.py --num_trials 100
conda run -n robot python generalization_eval/base/eval_base_x.py --num_trials 100
conda run -n robot python generalization_eval/base/eval_base_y.py --num_trials 100
conda run -n robot python generalization_eval/base/eval_base_z.py --num_trials 100
```

默认读取模型：`data/policy_checkpoint.pth`，输出在 `generalization_eval/runs/<timestamp>/...`。

## 汇总

把某次 run 的所有 jsonl 汇总成 `summary.csv`：

```bash
python generalization_eval/analysis/summarize.py --run_dir generalization_eval/runs/<timestamp>
```
