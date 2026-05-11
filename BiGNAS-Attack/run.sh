#!/bin/bash

for ratio in 50
do
    echo "======== Running popular_top_k=${ratio} ========"
    python search.py --T-max=10 --conv-lr=0.01 --descent-step=30 --dropout=0.8 --hpo-lr=0.001 --lr=0.01 --meta-hidden-dim=16 --meta-interval=20 --meta-op=sage --num-layers=2 --weight-decay=0.00001 --top_k=60 --device cuda:0 --categories CD Kitchen --target Kitchen   --use-hard-user-augment --hard-top-ratio 0.01 --popular_top_k ${ratio} --cold-item-id 31529 --use-source --use-meta --patience 20

done