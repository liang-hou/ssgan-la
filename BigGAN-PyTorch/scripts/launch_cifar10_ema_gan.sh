#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train.py \
--shuffle --batch_size 100 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 10000 \
--test_every 1000 --save_every 1000 --num_best_copies 1 --num_save_copies 0 --seed 0 \
--loss gan \
--num_epochs 400 --experiment_name c10_gan_n4
