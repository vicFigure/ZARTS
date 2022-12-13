#!/bin/sh
SLURM_ARRAY_JOB_ID=0
SPACE='s6' # DARTS w/o None
#DATA_ROOT='../data/reduced_imagenet/train'
DATA_ROOT='/data/wangxiaoxing/imagenet/train'
DATASET='imagenet'
EPOCHS=50
LOGDIR=logs

WARMUP=20
ITERW=10
SAMPLE=4
STD=0.025
STD_DECAY_TYPE='cosine'
TAU=0.01

DROP_PROB=0.0

SLURM_ARRAY_TASK_ID=0
python train_search.py --data $DATA_ROOT --batch_size 256 --num_workers 8 --space $SPACE --dataset $DATASET --epochs $EPOCHS --seed -1 --task $SLURM_ARRAY_TASK_ID --drop_path_prob $DROP_PROB --warmup $WARMUP --iter_w $ITERW --num_sample $SAMPLE --std $STD --std_decay_type $STD_DECAY_TYPE --tau $TAU --use_merge --train_portion 0.5 > $LOGDIR/mergeZARTS-$DATASET-$SLURM_ARRAY_TASK_ID.log 2>&1 &



