import os
import sys
sys.path.append('.')
sys.path.append('..')

from pathlib import Path
import json
from PIL import Image
from lpips import LPIPS
import torchvision.transforms.functional as F
import torch
import numpy as np
from tqdm import tqdm
import math

eth3d_evaluation_bin = Path('/home/yuqunwu2/large_scale_nerf/multi-view-evaluation/build/ETH3DMultiViewEvaluation')

scenes = ['courtyard', 'delivery_area', 'electro', 'facade', 'kicker', 'meadow', 'office', 'pipes', 'playground', 'relief', 'relief_2', 'terrace', 'terrains']

def evaluate_3d(pc_file, gt_path):
    gt_file = gt_path / "dslr_scan_eval" / "scan_alignment.mlp"
    exe_str = f'{str(eth3d_evaluation_bin)} --reconstruction_ply_path {pc_file} --ground_truth_mlp_path {gt_file} --tolerances 0.02,0.05'
    output = os.popen(exe_str).read()
    lines = output.split('\n')
    tolerances = [0.02, 0.05]
    print(exe_str)
    com_index = [i for i, line in enumerate(lines) if line.find('Completeness') == 0][0]
    com_line = lines[com_index]
    acc_line = lines[com_index+1]
    f1_line = lines[com_index+2]
    com_words = com_line.split()
    acc_words = acc_line.split()
    f1_words = f1_line.split()
    ress = {}
    for i, tol in enumerate(tolerances):
        res ={}
        res[f'completeness'] = float(com_words[i + 1])
        res[f'accuracy'] = float(acc_words[i + 1])
        res[f'f1'] = float(f1_words[i + 1])
        ress[f'tol_{tol}'] = res

    return ress

def main(args):
    # first perform fusion
    output_path = Path(args.output_path)
    gt_path = Path(args.gt_path)
    (output_path / 'results').mkdir(exist_ok=True, parents=True)
    pc_file =  Path(args.pc_file)

    # evaluate 3d 
    evals_3d = evaluate_3d(pc_file, gt_path)

    # write evaluation results to file
    evals = {**evals_3d}

    with open(output_path, 'w') as f:
        json.dump(evals, f)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--pc_file', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--gt_path', type=str)

    args = parser.parse_args()
    main(args)
