#!usr/bin/env python
# -*- coding:utf-8 -*-
# author: caojia

CUDA_VISIBLE_DEVICES=0 python3 eval_KVNet.py \
        --exp_name kitti_rawsize\
        --sigma_soft_max 10 \
        --frame_interv 5 \
        --t_win 2 \
        --d_min 1 \
        --d_max 60 \
        --feature_dim 64 \
        --ndepth 64 \
        --dataset kitti \
        --dataset_path /data4/kitti_raw/ \
        --split_file ./mdataloader/kitti_split/testing.txt \
        --model_path ./saved_models/kvnet_kitti.tar \
        --change_aspect_ratio False
