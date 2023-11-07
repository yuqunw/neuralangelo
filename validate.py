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
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import cv2
import torch.nn.functional as NF
import matplotlib.pyplot as plt

import imaginaire.config
from imaginaire.config import Config, recursive_update_strict, parse_cmdline_arguments
from imaginaire.utils.cudnn import init_cudnn
from imaginaire.utils.distributed import init_dist, get_world_size, master_only_print as print, is_master
from imaginaire.utils.gpu_affinity import set_affinity
from imaginaire.trainers.utils.logging import init_logging
from imaginaire.trainers.utils.get_trainer import get_trainer
from imaginaire.utils.set_random_seed import set_random_seed

def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', help='Path to the training config file.', required=True)
    parser.add_argument('--logdir', help='Dir for saving logs and models.', default=None)
    parser.add_argument("--checkpoint_txt", default="", help="Checkpoint txt containing path.")
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0))
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--show_pbar', action='store_true')
    parser.add_argument('--output_dir', type=str, help="Output directory for images", default='output')
    args, cfg_cmd = parser.parse_known_args()
    return args, cfg_cmd


def main():
    args, cfg_cmd = parse_args()
    set_affinity(args.local_rank)
    cfg = Config(args.config)

    cfg_cmd = parse_cmdline_arguments(cfg_cmd)
    recursive_update_strict(cfg, cfg_cmd)

    # Initialize the validation image size and number of images.
    cfg.data.val.image_size = [640, 960]
    cfg.data.val.subset = cfg.data.num_images
    # cfg.data.val.subset = 2

    # If args.single_gpu is set to True, we will disable distributed data parallel.
    if not args.single_gpu:
        # this disables nccl timeout
        os.environ["NCLL_BLOCKING_WAIT"] = "0"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        cfg.local_rank = args.local_rank
        init_dist(cfg.local_rank, rank=-1, world_size=-1)
    print(f"Training with {get_world_size()} GPUs.")

    # set random seed by rank
    set_random_seed(args.seed, by_rank=True)

    # Global arguments.
    imaginaire.config.DEBUG = args.debug

    # Create log directory for storing training results.
    cfg.logdir = init_logging(args.config, args.logdir, makedir=True)

    # Print and save final config
    if is_master():
        cfg.print_config()
        cfg.save_config(cfg.logdir)

    # Initialize cudnn.
    init_cudnn(cfg.cudnn.deterministic, cfg.cudnn.benchmark)

    # Initialize data loaders and models.
    trainer = get_trainer(cfg, is_inference=False, seed=args.seed)

    # trainer.set_data_loader(cfg, split="train")
    trainer.set_data_loader(cfg, split="val")

    with open(args.checkpoint_txt, 'r') as f:
        checkpoint_name = f.readline()[:-1]
    checkpoint_path = str(Path(args.checkpoint_txt).parents[0] / checkpoint_name)
    trainer.checkpointer.load(checkpoint_path, resume=True, load_sch=True, load_opt=True)
    trainer.model.module.eval()

    trainer.current_iteration = trainer.checkpointer.eval_iteration
    if cfg.model.object.sdf.encoding.coarse2fine.enabled:
        trainer.model_module.neural_sdf.set_active_levels(trainer.current_iteration)
        if cfg.model.object.sdf.gradient.mode == "numerical":
            trainer.model_module.neural_sdf.set_normal_epsilon()    
    # Start training.
    output = trainer.test(trainer.eval_data_loader,
                 mode = 'val',
                  show_pbar=args.show_pbar,
                   only_visualize = True) # save files

    # output_path = Path(args.output_path)
    # output_path.mkdir(parents=True, exist_ok=True)
    # (output_path / 'images').mkdir(parents=True, exist_ok=True)
    # (output_path / 'depths').mkdir(parents=True, exist_ok=True)

    # shutil.copy(Path(args.data_path) / 'transforms.json', output_path / 'transforms.json')


    # datamodule.setup_test()
    # dataset = datamodule.dataset

    # H, W = datamodule.height, datamodule.width
    # batch_size = 1024
    # fp_16 = False
    
    # # iterate and write to depth
    # print('Evaluating!')
    # for i, batch in enumerate(dataset):
    #     infs = render_image_raw(batch, model, grid, (H, W), batch_size, fp_16)
    #     rgb = infs['rgb']
    #     depth = infs['d']
    #     scale = batch['ray_s'].view(*depth.shape)
    #     scaled_depth = depth / scale
    #     np_image = (rgb.view(H, W, 3).cpu().numpy() * 255).astype(np.uint8)
    #     Image.fromarray(np_image).save(output_path / 'images' / f'{i:06d}.rgb.png')
    #     cv2.imwrite(str(output_path / 'depths' / f'{i:06d}.depth.tiff'), scaled_depth.cpu().numpy())
    # print('Evaluating Done!')

    image_dir = Path(args.output_dir) / 'images'
    depth_dir = Path(args.output_dir) / 'depths'
    normal_dir = Path(args.output_dir) / 'normals'
    depth_vis_dir = Path(args.output_dir) / 'depth_vis'
    image_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)
    depth_vis_dir.mkdir(parents=True, exist_ok=True)


    images = output['rgb_map']
    depths = output['depth_map']
    normals = output['normal_map']
    indexes = output['idx']

    for index, image, depth, normal in zip(indexes, images, depths, normals):
        image_name = f'{index:06d}.rgb.png'
        depth_name = f'{index:06d}.depth.tiff'
        normal_name = f'{index:06d}.normal.png'
        depth_vis_name = f'{index:06d}.depth_vis.png'

        image_path = str(image_dir / image_name)
        depth_path = str(depth_dir / depth_name)
        normal_path = str(normal_dir / normal_name)
        depth_vis_path = str(depth_vis_dir / depth_vis_name)

        np_image = (image.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(np_image).save(image_path)
        cv2.imwrite(depth_path, depth.cpu().numpy()[0])
        normal = NF.normalize(normal, dim=0, p=2)
        np_normal = ((normal * 0.5 + 0.5).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(np_normal).save(normal_path)
        plt.imsave(depth_vis_path, np.log(depth[0].cpu().numpy() ** 2), cmap='inferno')




if __name__ == "__main__":
    main()
