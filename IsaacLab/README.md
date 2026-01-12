### Environment Setup
``` bash
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab
pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
sudo apt install cmake build-essential
./isaaclab.sh --install
```

### Data Augmentation

* step a: create folder for datasets
```bash
mkdir -p datasets
```
* step b: collect data with a selected teleoperation device. Replace <teleop_device> with your preferred input device.
  - Available options: spacemouse, keyboard, handtracking
```bash
./isaaclab.sh -p scripts/tools/record_demos.py --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --device cpu --teleop_device <teleop_device> --dataset_file ./datasets/dataset.hdf5 --num_demos 10
```
* step a: replay the collected dataset
```bash
./isaaclab.sh -p scripts/tools/replay_demos.py --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --device cpu --dataset_file ./datasets/dataset.hdf5
```
* annotate the subtasks in the recording
```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
--device cpu --enable_cameras --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0 --auto \
--input_file ./datasets/dataset.hdf5 --output_file ./datasets/annotated_dataset.hdf5
```
* Then, use Isaac Lab Mimic to generate some additional demonstrations:
``` bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
--device cpu --enable_cameras --headless --num_envs 10 --generation_num_trials 1000 \
--input_file ./datasets/annotated_dataset.hdf5 --output_file ./datasets/mimic_dataset_1k.hdf5 \
--task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Mimic-v0 \
```
