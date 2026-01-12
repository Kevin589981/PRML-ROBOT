#!/usr/bin/env bash
set -euo pipefail

# 84.8% 视觉策略：课程要求版泛化测试
# 输出：84.8%/generalization_eval_vision/runs/<run_id>/

# ====== 1) 可调参数（你可以直接改这里） ======
CHECKPOINT=${CHECKPOINT:-"84.8%/checkpoints_full_traj_4090/best_policy.pth"}
EPISODES=${EPISODES:-100}
WORKERS=${WORKERS:-34}
DEVICE=${DEVICE:-cuda}    # cuda 或 cpu
SEED=${SEED:-10000}
RUN_ID=${RUN_ID:-"run_$(date +%Y%m%d_%H%M%S)"}

echo "[run_84_8_generalization_eval] checkpoint: ${CHECKPOINT}"
echo "[run_84_8_generalization_eval] episodes: ${EPISODES}, workers: ${WORKERS}, device: ${DEVICE}, seed: ${SEED}"
echo "[run_84_8_generalization_eval] run_id: ${RUN_ID}"

# ====== 2) 运行 ======
python -u "84.8%/generalization_eval_vision/run_all.py" \
  --checkpoint "${CHECKPOINT}" \
  --episodes "${EPISODES}" \
  --workers "${WORKERS}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --run_id "${RUN_ID}" \
  --plot

echo
echo "[run_84_8_generalization_eval] DONE"
echo "查看结果: 84.8%/generalization_eval_vision/runs/${RUN_ID}/summary.csv"
