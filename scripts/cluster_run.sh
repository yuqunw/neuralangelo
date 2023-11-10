#!/bin/bash
run_name="neuralangelo_eth3d"
echo "Args provided: $1 $2 $3 $4"
scene=$1
if [[ $2 == "--with_full" ]]; then
    with_full=True
    run_name+="_full"
    show_full="_full"
else
    with_full=False
    show_full="unfull"
fi
img_count=$3
full_img_count=$4



echo "Running with run name ${run_name}"
MOUNT_DIR="/home/yuqunwu2/large_scale_nerf/data/$(hostname -s)"
DATA_DIR="${MOUNT_DIR}/eth3d_neuralangelo"
GT_DIR="${MOUNT_DIR}/eth3d_ground_truths"
CHECKPOINT_DIR="${MOUNT_DIR}/${run_name}"
mkdir -p $CHECKPOINT_DIR
rm -r "${CHECKPOINT_DIR}/${scene}"

cd /home/yuqunwu2/large_scale_nerf/neuralangelo

EXPERIMENT="eth3d_courtyard"
CONFIG=projects/neuralangelo/configs/${EXPERIMENT}.yaml

echo "python train.py --data.root=${DATA_DIR}/${scene} \
                --data.num_images=${img_count} \
                --data.full=${with_full} \
                --max_iter=200000 \
                --config ${CONFIG} \
                --logdir=${CHECKPOINT_DIR}/${scene} \
                --single_gpu \
                --wandb \
                --wandb_name eth3d_neuralangelo_${scene}${show_full} \
                --checkpoint.save_iter=100000 \
                --model.appear_embed.enabled=False"

python train.py --data.root="${DATA_DIR}/${scene}" \
                --data.num_images=${img_count} \
                --data.full=${with_full} \
                --max_iter=200000 \
                --config ${CONFIG} \
                --logdir="${CHECKPOINT_DIR}/${scene}" \
                --single_gpu \
                --wandb \
                --wandb_name "eth3d_neuralangelo_${scene}${show_full}" \
                --checkpoint.save_iter=100000 \
                --model.appear_embed.enabled=False


echo "python projects/neuralangelo/scripts/extract_mesh_cluster.py \
                            --checkpoint_txt ${CHECKPOINT_DIR}/${scene}/latest_checkpoint.txt \
                            --config=${CONFIG} \
                            --data.root=${DATA_DIR}/${scene} \
                            --data.num_images=${full_img_count} \
                            --single_gpu \
                            --output_file ${CHECKPOINT_DIR}/${scene}/output/results/mesh.ply \
                            --textured \
                            --model.appear_embed.enabled=False"

python projects/neuralangelo/scripts/extract_mesh_cluster.py \
                            --checkpoint_txt "${CHECKPOINT_DIR}/${scene}/latest_checkpoint.txt" \
                            --config=${CONFIG} \
                            --data.root="${DATA_DIR}/${scene}" \
                            --data.num_images=${full_img_count} \
                            --single_gpu \
                            --output_file "${CHECKPOINT_DIR}/${scene}/output/results/mesh.ply" \
                            --textured \
                            --model.appear_embed.enabled=False

echo "python validate.py --data.root=${DATA_DIR}/${scene} \
                   --data.num_images=${full_img_count} \
                   --checkpoint_txt ${CHECKPOINT_DIR}/${scene}/latest_checkpoint.txt \
                   --config ${CONFIG} \
                   --single_gpu \
                   --output_dir ${CHECKPOINT_DIR}/${scene}/output/ \
                   --show_pbar \
                   --model.appear_embed.enabled=False"

python validate.py --data.root="${DATA_DIR}/${scene}" \
                   --data.num_images=${full_img_count} \
                   --checkpoint_txt "${CHECKPOINT_DIR}/${scene}/latest_checkpoint.txt" \
                   --config ${CONFIG} \
                   --single_gpu \
                   --output_dir "${CHECKPOINT_DIR}/${scene}/output/" \
                   --show_pbar \
                   --model.appear_embed.enabled=False


# # python scripts/report.py --input_path "${DATA_DIR}/${scene}" \
# #                          --output_path "${CHECKPOINT_DIR}/${scene}/output" \
# #                          --gt_path "${GT_DIR}/${scene}" 

cd ${CHECKPOINT_DIR}/${scene}/
zip -r "${scene}.zip" "output"

result_dirname="ablations"
ssh yuqun "mkdir -p /mnt/data/cluster_results/${result_dirname}/${run_name}"
ssh yuqun "mkdir -p /mnt/data/eth3d_outputs/${result_dirname}/${run_name}/${scene}"
scp "${scene}.zip" yuqun:/mnt/data/cluster_results/${result_dirname}/${run_name}/
ssh yuqun "cd /mnt/data/cluster_results/${result_dirname}/${run_name}/ && unzip ${scene}.zip -d /mnt/data/eth3d_outputs/${result_dirname}/${run_name}/${scene}"

