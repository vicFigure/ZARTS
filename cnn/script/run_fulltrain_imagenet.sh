#!/bin/sh
DATA=/data/wangxiaoxing/imagenet/

ID=20220617-155357
TASK=1
BASE_DIR=ckpt/search-EXP-$ID-task$TASK
LOG_DIR=test_logs_imagenet

BATCHSIZE=1024

GPUS=2
PORT=${PORT:-6666}

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    --use_env train_imagenet_distributed.py --data $DATA --batch_size $BATCHSIZE --auxiliary --base_path $BASE_DIR --genotype_name 49 --mode train --float16 > $LOG_DIR/$ID-task$TASK.log 2>&1 &


