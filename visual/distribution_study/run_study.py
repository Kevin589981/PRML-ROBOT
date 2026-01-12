import subprocess
import os
import json
import sys
import shutil


def _run(cmd):
    subprocess.check_call(cmd)


# ==================== Study definition ====================
# Baseline (84.8% default)
BASE = {
    # Gripper-end-effector offset distribution in demos (expert's XY misalignment)
    "expert_alignment_noise": 0.03,

    # Basket initial position distribution
    "basket_pos_x_noise": 0.08,

    # Cube initial position distribution widths (keep center fixed)
    "cube_x_width": 0.40,
    "cube_y_width": 0.75,
    "cube_x_center": 0.45,
    "cube_y_center": 0.025,

    # Reset initial end-effector absolute position noise (world frame, meters)
    # Note: this is NOT enabled in the original 84.8% baseline (default 0.0).
    # For this study, baseline includes reset EE noise (so evaluation also enables it).
    "ee_pos_noise": 0.04,
}

# 3 gradients per factor (baseline is included as one of the EE_POS_LEVELS entries).
SCALES = [0.75, 0.50, 0.25]

# For sweeping reset EE init distribution, pick explicit levels (meters).
# Note: baseline is 0.04, so we need distinct values
EE_POS_LEVELS = [0.03, 0.02, 0.01]  

# Quick-run setting: collecting 100 demos per experiment (you can increase later).
DEMOS_PER_EXPERIMENT = 100

# Evaluation env should always include reset EE noise too.
EVAL_EE_POS_NOISE = BASE["ee_pos_noise"]

# If baseline artifacts already exist from a previous run, force re-run it under
# the current quick-run settings (e.g., 100 demos).
FORCE_RERUN_BASELINE = True


def make_experiments():
    exps = []
    
    # 1) Baseline (always first)
    exps.append(
        {
            "name": "baseline",
            "expert_alignment_noise": BASE["expert_alignment_noise"],
            "basket_pos_x_noise": BASE["basket_pos_x_noise"],
            "cube_x_width": BASE["cube_x_width"],
            "cube_y_width": BASE["cube_y_width"],
            "cube_x_center": BASE["cube_x_center"],
            "cube_y_center": BASE["cube_y_center"],
            "ee_pos_noise": BASE["ee_pos_noise"],
        }
    )

    # 2) Reset initial end-effector absolute position distribution (ee_pos_noise)
    # Keep expert_alignment_noise (demo misalignment) fixed at baseline.
    for level in EE_POS_LEVELS:
        exps.append(
            {
                "name": f"ee_init_{int(level*1000)}mm",
                "expert_alignment_noise": BASE["expert_alignment_noise"],
                "basket_pos_x_noise": BASE["basket_pos_x_noise"],
                "cube_x_width": BASE["cube_x_width"],
                "cube_y_width": BASE["cube_y_width"],
                "cube_x_center": BASE["cube_x_center"],
                "cube_y_center": BASE["cube_y_center"],
                "ee_pos_noise": float(level),
            }
        )

    # 3) Basket distribution
    for s in SCALES:
        exps.append(
            {
                "name": f"basket_{int(s*100)}",
                "expert_alignment_noise": BASE["expert_alignment_noise"],
                "basket_pos_x_noise": BASE["basket_pos_x_noise"] * s,
                "cube_x_width": BASE["cube_x_width"],
                "cube_y_width": BASE["cube_y_width"],
                "cube_x_center": BASE["cube_x_center"],
                "cube_y_center": BASE["cube_y_center"],
                "ee_pos_noise": BASE["ee_pos_noise"],
            }
        )

    # 4) Cube distribution
    for s in SCALES:
        exps.append(
            {
                "name": f"cube_{int(s*100)}",
                "expert_alignment_noise": BASE["expert_alignment_noise"],
                "basket_pos_x_noise": BASE["basket_pos_x_noise"],
                "cube_x_width": BASE["cube_x_width"] * s,
                "cube_y_width": BASE["cube_y_width"] * s,
                "cube_x_center": BASE["cube_x_center"],
                "cube_y_center": BASE["cube_y_center"],
                "ee_pos_noise": BASE["ee_pos_noise"],
            }
        )

    return exps


EXPERIMENTS = make_experiments()

# Baseline will be trained/evaluated in this pipeline.
SKIP_BASELINE = False

results = {}

print("STARTING DISTRIBUTION STUDY")
print(f"Total experiments: {len(EXPERIMENTS)} (baseline skipped={SKIP_BASELINE})")
print("Order:")
for e in EXPERIMENTS:
    print(" -", e["name"])

for exp in EXPERIMENTS:
    name = exp["name"]
    print(f"\n{'#'*40}")
    print(f"Running Experiment: {name}")
    print(f"{'#'*40}")

    exp_dir = f"experiment_{name}"
    # Force re-run for baseline if needed
    # if name == "baseline" and FORCE_RERUN_BASELINE and os.path.exists(exp_dir):
    #     print("Forcing re-run for baseline: removing existing artifacts...")
    #     shutil.rmtree(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)

    data_path = os.path.join(exp_dir, "dataset.h5")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    res_json = os.path.join(exp_dir, "result_2.json")

    # 1) Generate dataset (distribution changes ONLY)
    if not os.path.exists(data_path):
        print(f"Generating Data for {name}...")
        cmd = [
            sys.executable,
            "data_collector.py",
            "--save_path",
            data_path,
            "--total_episodes",
            str(DEMOS_PER_EXPERIMENT),
            "--workers",
            "100",
            "--expert_alignment_noise",
            str(exp["expert_alignment_noise"]),
            "--basket_pos_x_noise",
            str(exp["basket_pos_x_noise"]),
            "--cube_x_width",
            str(exp["cube_x_width"]),
            "--cube_y_width",
            str(exp["cube_y_width"]),
            "--cube_x_center",
            str(exp["cube_x_center"]),
            "--cube_y_center",
            str(exp["cube_y_center"]),
            "--ee_pos_noise",
            str(exp.get("ee_pos_noise", 0.0)),
        ]
        try:
            _run(cmd)
        except subprocess.CalledProcessError as e:
            print(f"Error generating data for {name}: {e}")
            continue
    else:
        print(f"Data for {name} already exists.")

    # 2) Train (from scratch for each distribution)
    best_policy = os.path.join(ckpt_dir, "best_policy.pth")
    last_policy = os.path.join(ckpt_dir, "policy_ep200.pth")
    if not os.path.exists(best_policy):
        print(f"Training Model for {name}...")
        cmd = [
            sys.executable,
            "train_full_trajectory.py",
            "--dataset_path",
            data_path,
            "--save_dir",
            ckpt_dir,
            "--epochs",
            "200",
        ]
        try:
            _run(cmd)
        except subprocess.CalledProcessError as e:
            print(f"Error training for {name}: {e}")
            continue
    else:
        print(f"Model for {name} already trained.")

    # 3) Eval (evaluation code/config stays fixed)
    if not os.path.exists(res_json):
        print(f"Evaluating Model for {name}...")
        cmd = [
            sys.executable,
            "eval_full_trajectory.py",
            "--ckpt",
            # best_policy,
            last_policy,
            "--total_episodes",
            "100",
            "--workers",
            "25",
            "--output_json",
            res_json,
            "--ee_pos_noise",
            str(EVAL_EE_POS_NOISE),
        ]
        try:
            _run(cmd)
        except subprocess.CalledProcessError as e:
            print(f"Error evaluating for {name}: {e}")
            continue
    else:
        print(f"Evaluation for {name} already done.")

    if os.path.exists(res_json):
        with open(res_json, "r") as f:
            res = json.load(f)
        results[name] = res.get("success_rate")
        print(f"Result {name}: {results[name]}%")

print("\n\n" + "=" * 40)
print("FINAL RESULTS SUMMARY")
print("=" * 40)
print(json.dumps(results, indent=2, ensure_ascii=False))

with open("study_summary_2.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
