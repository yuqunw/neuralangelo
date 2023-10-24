
# python train.py --logdir logs/nerual_angelo_kicker \
#                 --config projects/neuralangelo/configs/eth3d_kicker.yaml \
#                 --show_pbar \
#                 --single_gpu \
#                 --wandb \
#                 --wandb_name neural_angelo_kicker

# python train.py --logdir logs/nerual_angelo_meetingroom_ours \
#                 --config projects/neuralangelo/configs/tnt_meetingroom_ours.yaml \
#                 --show_pbar \
#                 --single_gpu \
#                 --wandb \
#                 --wandb_name neural_angelo_meetingroom_ours


python train.py --logdir logs/neural_angelo_eth3d_rescale \
                --config projects/neuralangelo/configs/eth3d_courtyard.yaml \
                --show_pbar \
                --single_gpu \
                --wandb \
                --wandb_name neural_angelo_eth3d_rescale
