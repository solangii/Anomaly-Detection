#! /bin/sh
nohup python -u main.py \
    --gpu 1 \
    --dataset 'capsule' \
    --dataroot 'datasets/' \
    --mode 'train' \
    --seed 1 \
    --epochs 5000 \
    --batch_size 3 \
    --lr 1e-4 \
    > result/nohup/capsule.out &