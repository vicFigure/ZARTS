#!/bin/sh
# RDARTS Evaluation
DATASET='cifar10'
CONFIG="--layers 20 --init_channels 36"
#CONFIG=""
ID=20220615-124607
SEARCH_DIR=ckpt/search-EXP-$ID-task
LOG_DIR=test_logs

BATCHSIZE=64

let j=0
for i in $(seq 0 0);do
    let SLURM_ARRAY_TASK_ID=$i
    BASE_DIR="$SEARCH_DIR$i"
    echo $BASE_DIR $j
    python train.py --gpu $j $CONFIG --batch_size $BATCHSIZE --cutout --auxiliary --dataset $DATASET --base_dir $BASE_DIR > $LOG_DIR/$ID-task$i.log  2>&1 &
    let j=$j+1
done


