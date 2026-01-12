# PRML-ROBOT: Generalization Challenges in Imitation Learning for Robotic Manipulation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyBullet](https://img.shields.io/badge/PyBullet-3.2.5-orange.svg)](https://pybullet.org/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.5-purple.svg)](https://developer.nvidia.com/isaac-sim)

> **A Case Study on Generalization Challenges in Imitation Learning: Architecture Choices, Distribution Design, and Data Scaling for Robust Robotic Manipulation**
> 
> Authors: Qiming Qiu (23307110278), Zhipeng Xu (23307110122)

## 📖 Overview

This repository contains the complete implementation and experimental framework for our research on **imitation learning generalization** in robotic manipulation tasks. We systematically investigate how architecture choices, training data distribution, and data augmentation techniques affect policy robustness under distribution shift.

### Key Findings

- **Privileged State Policy** (MLP-Base): Achieves **98.6%** success with perfect state information, but catastrophically fails under temporal/actuation noise
- **Vision-Based Policy** (Vision-Final): Achieves **84.8%** success in fully randomized environments using only RGB-D inputs
- **Data Augmentation**: MimicGen expands 10 real demonstrations to 2000 trajectories, improving success rate from 60% to **97%**
- **Visual Generalization**: Cosmos Transfer enables photorealistic domain randomization for sim-to-real transfer

### Main Contributions

1. **Comprehensive Generalization Framework** for Franka Panda in PyBullet
   - High-resolution visual representations (112×112 RGB-D)
   - Spatial Softmax for geometric feature extraction
   - Dense 9-phase supervision for perceptual disambiguation
   - Systematic evaluation across 12 perturbation dimensions

2. **Scalable Data Enhancement Pipeline** in Isaac Sim
   - MimicGen for trajectory interpolation-based augmentation
   - Cosmos Transfer for visual domain randomization
   - Teleoperation via Apple Vision Pro for high-quality demonstrations

## 🎯 Tasks

### Task 1: Pick-and-Place (PyBullet)
- **Environment**: 7-DoF Franka Panda with parallel-jaw gripper
- **Goal**: Grasp a randomized cube and place it into a basket
- **Challenges**: Full randomization of object poses, lighting, friction, sensor noise


### Task 2: Three-Cube Stacking (Isaac Sim)
- **Environment**: Franka Panda in Isaac Sim with MimicGen integration
- **Goal**: Stack three cubes vertically in sequence
- **Challenges**: Multi-step planning, precise alignment, contact stability

## 🏗️ Project Structure

```
PRML-ROBOT/
├── behavior_cloning/          # Main BC implementation (PyBullet)
│   ├── MLP_only/              # Privileged state baseline
│   ├── visual/                # Vision-based policies
│   └── tools/                 # Dataset inspection utilities
├── IsaacLab/                  # Isaac Sim environment
│   ├── Controllers/           # Teleoperation & retargeting
│   └── scripts/imitation_learning/  # MimicGen integration
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- NVIDIA GPU (RTX 4090 recommended for Cosmos Transfer)
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/PRML-ROBOT.git
cd PRML-ROBOT
```

2. **Set up PyBullet environment**
```bash
cd behavior_cloning
pip install uv
uv sync
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Set up Isaac Sim environment** (for advanced tasks)
```bash
cd ../IsaacLab
./isaaclab.sh --install  # Follow Isaac Lab installation guide
```

### Usage

#### 1. MLP Baseline (Privileged State)

```bash
cd behavior_cloning/MLP_only

# Collect expert demonstrations (1000 trajectories)
python data_collector.py

# Train BC policy
python train_bc.py

# Evaluate on test set
python eval_policy.py

# Run comprehensive generalization tests (12 dimensions)
cd generalization_eval
python run_all.py
```

#### 2. Vision-Based Policy

Navigate to the desired configuration folder (named by success rate):

| Folder | Resolution | Privileged Info | Basket Randomization | Dropout | Success Rate |
|:-------|:-----------|:----------------|:---------------------|:--------|:-------------|
| 100.0% | 64px | ✅ Cube relative pose | ❌ | 0.3 | 100.0% |
| 47.4%  | 64px | ❌ | ❌ | 0.3 | 47.4% |
| 96.0%  | 112px | ❌ | ❌ | 0.3 | 96.0% |
| **84.8%** | **112px** | ❌ | ✅ | **0.3** | **84.8%** |
| 61.8%  | 112px | ❌ | ✅ | 0.3 + 2D-0.2 | 61.8% |

```bash
# Example: Run Vision-Final (84.8%)
cd behavior_cloning/visual/84.8%

# Collect visual demonstrations
python data_collector.py

# Train visual BC policy
python train_full_trajectory.py

# Evaluate (default: 500 episodes)
python eval_full_trajectory.py --total_episodes 500

# Run generalization tests
cd generalization_eval_vision
python run_all.py
```

#### 3. Isaac Sim + MimicGen (Advanced)

```bash
cd IsaacLab

# Collect 10 human teleoperation demonstrations via Vision Pro
# (Follow Controllers/LocomanipulationAssets documentation)

# Generate 2000 augmented trajectories using MimicGen
python scripts/imitation_learning/mimicgen_augment.py \
    --source_demos 10 \
    --output_demos 2000 \
    --subtasks 3

# Train stacking policy
python scripts/imitation_learning/train_stacking.py

# Evaluate
python scripts/imitation_learning/eval_stacking.py
```

## 📊 Experimental Results

### PyBullet Experiments

#### Baseline Performance
| Model | Architecture | Input | Success Rate |
|:------|:-------------|:------|:-------------|
| MLP-Base | Residual MLP | Privileged state ($\mathbf{x}_{obj}^{ee}$) | **98.6%** |
| Vision-Final | ResNet-18 + LSTM | RGB-D (112×112) + Proprio | **84.8%** |

#### Generalization Robustness

**Vision-Final Performance Under Perturbations:**
- ✅ Camera pose noise (±2cm): ~86% → ~65% (graceful degradation)
- ✅ Basket position noise (±16cm): >80%
- ✅ Friction variation (μ=0.5-5.0): ~70%-88%
- ⚠️ Spatial extrapolation (2× training range): 50%
- ❌ Action noise (σ=0.005): ~10% (critical vulnerability)

**MLP-Base Failure Modes:**
- ❌ Simulation timestep change (240Hz→480Hz): 100% → 0%
- ❌ Action noise (σ=0.01): 0%
- ❌ Height offset (+10cm): 0%

### Isaac Sim Experiments

#### MimicGen Data Augmentation
| Training Dataset | # Trajectories | Success Rate |
|:-----------------|:---------------|:-------------|
| Human Teleop Only | 10 | ~40% |
| MimicGen 1k | 1000 | 60.0% |
| **MimicGen 2k** | **2000** | **97.0%** |

#### Distribution Study (100 demos budget)
| Randomization Factor | Optimal Level | Test Success |
|:---------------------|:--------------|:-------------|
| Cube Range | 100% (full coverage) | 71.0% |
| Basket Noise | 25% (focused) | 58.0% |
| EE Init Noise | 20mm (sweet spot) | **71.0%** |

### Key Insights

1. **Privileged ≠ Robust**: MLP-Base achieves near-perfect success but is extremely brittle to temporal/actuation changes
2. **Resolution Matters**: 64×64 → 112×112 doubles success rate (47.4% → 96.0%)
3. **Spatial Softmax > Pooling**: Explicit keypoint extraction crucial for geometric reasoning
4. **Phase Supervision Helps**: 9-phase classification improves temporal coherence
5. **Data Scaling Works**: MimicGen demonstrates strong positive scaling (60% → 97%)

## 🛠️ Advanced Features

### Teleoperation System
- **Hardware**: Apple Vision Pro with OpenXR hand tracking
- **Retargeting**: 26-DoF hand pose → 7-DoF arm + 2-DoF gripper
- **IK Solver**: Real-time inverse kinematics for Franka Panda

### Data Augmentation Pipeline
1. **Geometric Augmentation** (MimicGen):
   - Key-frame extraction from human demos
   - Cubic spline interpolation with Gaussian noise
   - IK re-solving for kinematic validity
   
2. **Visual Augmentation** (Cosmos Transfer):
   - Multimodal conditioning (RGB, Depth, Segmentation)
   - Adaptive spatiotemporal control map
   - Photorealistic style transfer (lighting, texture, background)

### Network Architecture
```
Vision-Final Architecture:
┌─────────────────────────────────────────┐
│ ResNet-18 (modified for RGBD input)     │
│   ├─ Remove Layer4 (preserve resolution)│
│   └─ Project: 256 → 64 channels         │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│ Spatial Softmax (K=64 keypoints)        │
│   Output: (2 cameras, 64, 2D coords)    │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│ 2-Layer LSTM (hidden=512)               │
│   + Proprioception (joint angles, etc.) │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
  Action Head          Phase Head
  (Δpos + gripper)     (9 classes)
```

## 📈 Reproducibility

All experiments use fixed random seeds for reproducibility:
- MLP-Base: 500 evaluation runs
- Vision-Final: 500 evaluation runs
- Ablations: 200 evaluation runs each

Training hyperparameters:
- Optimizer: AdamW (lr=2e-4, weight_decay=1e-3)
- Loss weights: λ_pos=1.0, λ_grip=0.5, λ_phase=0.2
- Batch size: 256
- Epochs: 200 (with early stopping)


## 🔗 Related Work

- **MimicGen**: [Automated Data Generation for Dexterous Manipulation](https://mimicgen.github.io/)
- **Cosmos Transfer**: [NVIDIA World Foundation Models](https://www.nvidia.com/en-us/ai/cosmos/)
- **Isaac Lab**: [Robot Learning Framework](https://isaac-sim.github.io/IsaacLab/)
- **PyBullet**: [Physics Simulation](https://pybullet.org/)

## 📧 Contact

- Qiming Qiu: 23307110278@m.fudan.edu.cn
- Zhipeng Xu: 23307110122@m.fudan.edu.cn

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NVIDIA for Isaac Sim and Cosmos Transfer
- Stanford PAIR lab for MimicGen framework
- Fudan University PRML course for project support

---

**Note**: This is a research project. The code is provided as-is for educational and research purposes. For production deployment, additional safety measures and validation are required.
