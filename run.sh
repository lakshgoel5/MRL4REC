#!/bin/bash

nohup python main_mul.py \
  --dataset btp_10k --K 5 --loss mrl --ns dns \
  --n_negs 32 --dim 768 --context_hops 2 \
  --batch_size 4096 --test_batch_size 1024 --nesting_list "[64, 128, 256, 512, 768]"\
  --epoch 1000 --early_stop False \
  --save True --gpu_id 0 > output.log 2>&1 &
