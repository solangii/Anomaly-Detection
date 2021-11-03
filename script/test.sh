#! /bin/sh
nohup python -u ../main.py \
    --gpu 0 \
    --dataset 'toothbrush' \
    --mode 'test/defective' \
    --weight_path '../result/toothbrush/param/epo-1500_lr-1e-05_bs-1_const-1e-05.pth' \
    > ../result/nohup/test.out &