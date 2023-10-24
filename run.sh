#!bin/bash
python train.py --logdir logs/nerual_angelo_meetingroom_na_preprocess \
                --config projects/neuralangelo/configs/tnt_meetingroom.yaml \
                --show_pbar \
                --single_gpu \
                --wandb \
                --wandb_name tnt_meetingroom_neuralangelo_preprocessed
