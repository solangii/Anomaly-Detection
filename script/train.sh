#! /bin/sh
nohup python -u ../main.py \
    --gpu 0 \
    --dataset 'toothbrush' \
    --dataroot '../datasets/' \
    --mode 'train' \
    --seed 1 \
    --epochs 1500 \
    --batch_size 1 \
    > ../result/nohup/train.out &