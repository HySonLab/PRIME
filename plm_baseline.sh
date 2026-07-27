#!/bin/bash

DATA_CONFIG="./config/data_config.yaml"
CUDA_DEVICE=2
SEED=1

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# for PLM in  esm2 esmc prott5 saprot; do
for PLM in esmc; do
    echo "===================================="
    echo "PLM: $PLM | Seed: $SEED"
    echo "===================================="

    python plm_baseline.py \
        --data_config $DATA_CONFIG \
        --plm $PLM \
        --cache_dir ./plm_cache \
        --batch_size 64 \
        --epochs 100 \
        --lr 1e-3 \
        --hidden_dim 512 \
        --num_layers 3 \
        --dropout 0.3 \
        --seed $SEED
done

echo "All done."