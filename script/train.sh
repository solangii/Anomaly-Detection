#! /bin/sh
nohup python -u ../main.py \
    --dataset 'toothbrush' \
    --dataroot '../datasets/' \
    --mode 'train' \
    --seed 1 \
    --epochs 1500 \
    --constant 1e-9 \
    > ../result/nohup/train.out &