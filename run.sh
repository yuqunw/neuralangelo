#!bin/bash


EXPERIMENT=tnt_meetingroom
GROUP=eth3d_rescale
NAME=courtyard_large_batch
CONFIG=projects/neuralangelo/configs/${EXPERIMENT}.yaml
GPUS=3  # use >1 for multi-GPU training!
torchrun --nproc_per_node=${GPUS} train.py \
    --logdir=logs/${GROUP}/${NAME} \
    --config=${CONFIG} \
    --wandb \
    --wandb_name eth3d_rescale_courtyard_large_batch 


# python train.py --logdir logs/nerual_angelo_meetingroom_na_preprocess \
#                 --config projects/neuralangelo/configs/tnt_meetingroom.yaml \
#                 --show_pbar \
#                 --wandb \
#                 --wandb_name tnt_meetingroom_neuralangelo_preprocessed_large_batch
