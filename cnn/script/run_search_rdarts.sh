#!/bin/sh
SLURM_ARRAY_JOB_ID=0
SPACE='s4'
DATASET='svhn'
#LOGDIR=logs_darts
LOGDIR=logs_rdarts
EPOCHS=50
gpu=0

WARMUP=10
ITERW=10
SAMPLE=4
STD=0.2 # 0.2
STD_DECAY_TYPE='cosine' # 'stable'
TAU=0.01
LR=0.025 # 0.025

DROP_PROB=0.0

for i in $(seq 0 3);do
    let SLURM_ARRAY_TASK_ID=$i
    echo $i $gpu
    # don't compute hessian
    python train_search.py --save RDARTS --learning_rate $LR --space $SPACE --dataset $DATASET --epochs $EPOCHS --gpu $gpu --seed -1 --task $SLURM_ARRAY_TASK_ID --drop_path_prob $DROP_PROB --warmup $WARMUP --iter_w $ITERW --num_sample $SAMPLE --std $STD --std_decay_type $STD_DECAY_TYPE --tau $TAU > $LOGDIR/test22-$DATASET-$SPACE-$SLURM_ARRAY_TASK_ID.log 2>&1 &
    let gpu=($gpu+1)%2
done

