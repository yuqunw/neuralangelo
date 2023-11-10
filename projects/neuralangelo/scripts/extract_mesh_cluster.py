'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import argparse
import json
import os
import sys
import numpy as np
from functools import partial
import torch
import multiprocessing as mp
from tqdm import tqdm
# import trimesh
# import open3d as o3d
import pandas as pd
from pyntcloud import PyntCloud
from pathlib import Path

sys.path.append(os.getcwd())
from imaginaire.config import Config, recursive_update_strict, parse_cmdline_arguments  # noqa: E402
from imaginaire.utils.distributed import init_dist, get_world_size, is_master, master_only_print as print  # noqa: E402
from imaginaire.utils.gpu_affinity import set_affinity  # noqa: E402
from imaginaire.trainers.utils.get_trainer import get_trainer  # noqa: E402
from projects.neuralangelo.utils.mesh import extract_mesh, extract_texture  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config", required=True, help="Path to the training config file.")
    parser.add_argument("--checkpoint_txt", default="", help="Checkpoint txt containing path.")
    parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0))
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument("--resolution", default=2048, type=int, help="Marching cubes resolution")
    parser.add_argument("--block_res", default=64, type=int, help="Block-wise resolution for marching cubes")
    parser.add_argument("--output_file", default="mesh.ply", type=str, help="Output file name")
    parser.add_argument("--textured", action="store_true", help="Export mesh with texture")
    parser.add_argument("--keep_lcc", action="store_true",
                        help="Keep only largest connected component. May remove thin structures.")
    args, cfg_cmd = parser.parse_known_args()
    return args, cfg_cmd

def generate_pc_from_mesh(mesh_path, pc_path):
    mp.freeze_support()
    data_mesh = o3d.io.read_triangle_mesh(mesh_path)

    sampled_pc = data_mesh.sample_points_poisson_disk(number_of_points = 2000000, )
    np_points = np.asarray(sampled_pc.points).astype(np.float32)

    pointdata = pd.DataFrame({
        'x': np_points[:, 0],
        'y': np_points[:, 1],
        'z': np_points[:, 2]
    })
    pointcloud = PyntCloud(pointdata)
    pointcloud.to_file(str(pc_path))

def main():
    args, cfg_cmd = parse_args()
    set_affinity(args.local_rank)
    cfg = Config(args.config)

    cfg_cmd = parse_cmdline_arguments(cfg_cmd)
    recursive_update_strict(cfg, cfg_cmd)

    # If args.single_gpu is set to True, we will disable distributed data parallel.
    if not args.single_gpu:
        # this disables nccl timeout
        os.environ["NCLL_BLOCKING_WAIT"] = "0"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        cfg.local_rank = args.local_rank
        init_dist(cfg.local_rank, rank=-1, world_size=-1)
    print(f"Running mesh extraction with {get_world_size()} GPUs.")

    cfg.logdir = ''

    # Initialize data loaders and models.
    trainer = get_trainer(cfg, is_inference=True, seed=0)
    # Load checkpoint.
    with open(args.checkpoint_txt, 'r') as f:
        checkpoint_name = f.readline()[:-1]
    checkpoint_path = str(Path(args.checkpoint_txt).parents[0] / checkpoint_name)
    trainer.checkpointer.load(checkpoint_path, load_opt=False, load_sch=False)
    trainer.model.eval()

    # Set the coarse-to-fine levels.
    trainer.current_iteration = trainer.checkpointer.eval_iteration
    if cfg.model.object.sdf.encoding.coarse2fine.enabled:
        trainer.model_module.neural_sdf.set_active_levels(trainer.current_iteration)
        if cfg.model.object.sdf.gradient.mode == "numerical":
            trainer.model_module.neural_sdf.set_normal_epsilon()

    meta_fname = f"{cfg.data.root}/transforms.json"
    with open(meta_fname) as file:
        meta = json.load(file)

    if "aabb_range" in meta:
        bounds = (np.array(meta["aabb_range"]) - np.array(meta["sphere_center"])[..., None]) / meta["sphere_radius"]
    else:
        bounds = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])

    sdf_func = lambda x: -trainer.model_module.neural_sdf.sdf(x)  # noqa: E731
    texture_func = partial(extract_texture, neural_sdf=trainer.model_module.neural_sdf,
                           neural_rgb=trainer.model_module.neural_rgb,
                           appear_embed=trainer.model_module.appear_embed) if args.textured else None
    mesh = extract_mesh(sdf_func=sdf_func, bounds=bounds, intv=(2.0 / args.resolution),
                        block_res=args.block_res, texture_func=texture_func, filter_lcc=args.keep_lcc)


    transform_path = os.path.join(cfg.data.root, 'transforms.json')
    # Transform to world coordWinates
    # First load true scale matrix
    with open(transform_path, 'r') as f:
        transforms = json.load(f)
    center = np.array(transforms['pose_offset']).reshape(3,)
    scale = transforms['pose_scale']      
    true_scale_mat = np.eye(4).astype(np.float32)
    true_scale_mat[:3, 3] = -center
    true_scale_mat[:3 ] /= scale 
    # Get the inverse of the true scale matrix, following monosdf definition
    inv_true_scale_mat = np.linalg.inv(true_scale_mat)

    # Apply the inverse of the true scale matrix to the mesh
    mesh.apply_transform(inv_true_scale_mat)

    if is_master():
        print(f"vertices: {len(mesh.vertices)}")
        print(f"faces: {len(mesh.faces)}")
        if args.textured:
            print(f"colors: {len(mesh.visual.vertex_colors)}")
        # center and scale
        # mesh.vertices = mesh.vertices * meta["sphere_radius"] + np.array(meta["sphere_center"])
        # mesh.update_faces(mesh.nondegenerate_faces())
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        mesh.export(args.output_file)
        pc_file = args.output_file.replace('mesh', 'fused')
        # generate_pc_from_mesh(args.output_file, pc_file)

if __name__ == "__main__":
    main()
