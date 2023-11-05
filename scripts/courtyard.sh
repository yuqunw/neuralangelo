#!bin/bash


EXPERIMENT=eth3d_courtyard
GROUP=eth3d
NAME=courtyard
CONFIG=projects/neuralangelo/configs/${EXPERIMENT}.yaml
GPUS=1  # use >1 for multi-GPU training!
# torchrun --nproc_per_node=${GPUS} train.py \
#     --logdir=logs/${GROUP}/${NAME} \
#     --config=${CONFIG} \
#     --wandb \
#     --wandb_name eth3d_courtyard


python train.py \
    --logdir=logs/${GROUP}/${NAME} \
    --config=${CONFIG} \
    --wandb \
    --wandb_name eth3d_courtyard \
    --single_gpu

# python train.py --logdir logs/nerual_angelo_meetingroom_na_preprocess \
#                 --config projects/neuralangelo/configs/tnt_meetingroom.yaml \
#                 --show_pbar \
#                 --wandb \
#                 --wandb_name eth3d_rescale_large_batch
