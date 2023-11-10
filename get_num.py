from pathlib import Path
import os
import json

base_dir = Path('/home/yuqunwu2/large_scale_nerf/data/vision-21/eth3d_neuralangelo')

num_list = []
for scene in sorted(base_dir.iterdir()):
    trans = scene / 'transforms_train.json'
    with open(trans, 'r') as f:
        train_trans = json.load(f)
    frames = train_trans['frames']
    num_list.append(len(frames))
print(num_list)