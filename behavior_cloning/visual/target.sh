#!/bin/bash

python data_collector.py
python train_full_trajectory.py
python eval_full_trajectory.py --total_episodes 500 --workers 20