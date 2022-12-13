#!/bin/sh
SLURM_ARRAY_JOB_ID=0
SPACE='s6'
DATASET='cifar10'
LOGDIR=logs
EPOCHS=50
gpu=0

WARMUP=10
ITERW=10
SAMPLE=4
STD=0.025
STD_DECAY_TYPE='cosine'
TAU=0.01

DROP_PROB=0.0

for i in $(seq 4 7);do
    let SLURM_ARRAY_TASK_ID=$i
    echo $i $gpu
    python train_search.py --space $SPACE --dataset $DATASET --epochs $EPOCHS --gpu $gpu --seed -1 --task $SLURM_ARRAY_TASK_ID --drop_path_prob $DROP_PROB --warmup $WARMUP --iter_w $ITERW --num_sample $SAMPLE --std $STD --std_decay_type $STD_DECAY_TYPE --tau $TAU --use_merge > $LOGDIR/mergeZARTS-$DATASET-$SLURM_ARRAY_TASK_ID.log 2>&1 &
    let gpu=($gpu+1)%8
done
