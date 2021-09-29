export PYTHONPATH=/home/ubuntu/Projects/SIW/CBNetV2:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0,1,2,3

TIME=$(date +"%m-%d-%Y-%H-%M-%S")
LOG=$TIME.txt

#!/usr/bin/env bash

CONFIG=../../configs/cbnet/cascade_mask_rcnn_cbv2_swin_large_patch4_window7_mstrain_400-1400_adamw_3x_object356.py
GPUS=4
PORT=${PORT:-29500}

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    ../../tools/train.py $CONFIG --launcher pytorch ${@:3} $1 2>&1 | tee "LOG"

# if you want to resume training, add the following parts
# --load_ckpt <the path of the checkpoint>.pth \
# --resume


# Note: Try the classification? 
# Change config files
# --cfg_file lib/configs/resnet50_stride32_multidepth_classificationc100 \
# Add classification loss
# --loss_mode _class-wshift_

