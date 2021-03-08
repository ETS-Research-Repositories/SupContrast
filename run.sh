#!/bin/bash
set -e pipefail
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

supcon_bs=1024
lr_bs=512
ce_bs=1024

mkdir -p tmp/ce
python main_ce.py --batch_size ${ce_bs} \
  --learning_rate 0.8 \
  --cosine --save_dir tmp/ce \
  --epoch 1 >tmp/ce/result.txt

mkdir -p tmp/supcon
python main_supcon.py --batch_size ${supcon_bs} \
  --learning_rate 0.5 \
  --temp 0.1 \
  --cosine \
  --save_dir tmp/supcon \
  --epoch 1 >tmp/supcon/pretrain.txt

python main_linear.py --batch_size ${lr_bs} \
  --learning_rate 5 \
  --ckpt tmp/supcon/last.pth \
  --epoch 1 >tmp/supcon/result.txt

# desupcon
mkdir -p tmp/dsupcon
python main_supcon.py --batch_size ${supcon_bs} \
  --learning_rate 0.5 \
  --temp 0.1 \
  --cosine \
  --save_dir tmp/dsupcon \
  --epoch 1 \
  --method dSupCon >tmp/dsupcon/pretrain.txt

python main_linear.py --batch_size ${lr_bs} \
  --learning_rate 5 \
  --ckpt tmp/dsupcon/last.pth \
  --epoch 1 \
  --method dSupCon >tmp/dsupcon/result.txt
