### 环境配置
1. 安装uv
```sh
pip install uv
```
2. 安装环境
```sh
uv sync
```
3. 激活环境
```sh
.venv/bin/activate
```

### 如何运行
#### 1. MLP_only
```sh
cd MLP_only
# 收集数据集
python data_collector.py
# 运行训练
python train_bc.py
# 运行测评
python eval_policy.py

# 运行泛化测试
cd generalization_eval
python run_all.py
```

#### 2. 视觉
1. 进入你需要测试的代码版本（文件夹名以成功率命名）

| 成功率（文件夹名） | 分辨率 | 特权参数 | basket位置随机化 | Dropout |
| :--- | :--- | :--- | :--- | :--- |
| 100.0% | 64px | 方块相对抓夹末端向量 | 无 | dropout 0.3 |
| 47.4% | 64px | 无 | 无 | dropout 0.3 |
| 96.0% | 112px | 无 | 无 | dropout 0.3 |
| 84.8% | 112px | 无 | 有 | dropout 0.3 |
| 61.8% | 112px | 无 | 有 | dropout 0.3+2D dropout 0.2 |
| 78.6% | 112px | 无 | 有 | dropout 0.2+2D dropout 0.2 |
```sh
# 以进入47.4%成功率的代码版本为例
cd visual/47.4%
# 运行测试
chmod +x ../target.sh
../target.sh
```
特别地，如果有特别需要，可以在target.sh中最后一行`python eval_full_trajectory.py` 修改参数
* --total_episodes <需要测评的次数，默认为500> 
* --workers <测评开启的进程数，默认为10>
* --ckpt <需要测评的模型权重路径，默认自动读取训练代码中的模型权重路径>
* --save_video <测评同时录制mp4，默认不开启>
* --max_steps <设置仿真最大步数，默认为200>
* --device <设置cuda或cpu，设置仿真的设备，默认为cuda>

2. 运行在84.8%成功率基线下所训练出的模型的泛化能力
```sh
cd visual
chmod +x run_generalization_eval.sh
./run_generalization_eval.sh
```
然后在 `/visual/84.8%/generalization_eval_vision/runs` 下查看测评结果

3. 研究初始数据集分布对模型泛化能力的影响
```sh
cd /visual/distribution_study
python run_study.py
# 绘制图表
python plot_study_summary.py
```

#### 3. 工具代码
- dataset_sanity_check.py
检查数据集数据分布是否合理，有无异常值
- diagnose.py
检查数据集结构和权重输入结构是否匹配
- inspect_dataset.py
将收集到的数据集转为图像序列导出
- inspect_structure.py
移动到对应的训练代码目录，运行后检查模型权重结构与训练代码是否一致
