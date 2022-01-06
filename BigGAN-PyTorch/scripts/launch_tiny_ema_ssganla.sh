#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train.py \
--shuffle --batch_size 100 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 \
--num_D_steps 4 --G_lr 1e-4 --D_lr 4e-4 \
--dataset TINY \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_nl inplace_relu --D_nl inplace_relu \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 20000 \
--test_every 1000 --save_every 1000 --num_best_copies 1 --num_save_copies 0 --seed 0 \
--loss ssgan_la --multi_hinge --all_T \
--num_epochs 400 --experiment_name tiny_ssganla_n4_mh_allT
