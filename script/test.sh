#! /bin/sh
nohup python -u ../main.py \
    --dataset 'toothbrush' \
    --mode 'test/defective' \
    --weight_path '../result/toothbrush/param/epo-1500_lr-0.0001_bs-1_const-1e-09.pth' \
    --memo '' \
    > ../result/nohup/test.out &