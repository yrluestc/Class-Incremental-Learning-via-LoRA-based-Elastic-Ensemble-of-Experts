#!/bin/bash

#dataset seq-imagenet-r, seq-imagenet-a, seq-cifar100-224, seq-domainnet, seq-cub200

python main.py --dataset seq-imagenet-r --model eee --lr 0.03 --batch_size 32 --n_epochs 10 --num_workers 0 --backbone eee --task_per_exp 1\
      --full_expansion False --importance_metric mag\
      --reg_his True --reg_cur True --if_fallback True\
      --if_gradproj True --if_GAPP True --if_partition True \
      --if_freeze_old_classes True \
      --device 0 \
      > EEE_test2_INR10.out 2>&1 &

