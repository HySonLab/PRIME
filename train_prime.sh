#!/bin/bash

# ==========================================
# Training Script
# ==========================================

# -----------------------------
# Config
# -----------------------------
DATA_CONFIG="/home/dvnguye2/PRL/config/data_config.yaml"
MODEL_CONFIG="/home/dvnguye2/PRL/config/model_config.yaml"
TASK="FoldClassification"   # Options: FoldClassification, ECReaction, GeneOntology
BATCH_SIZE=32
EPOCHS=150
LR=1e-3

# GPU Selection
DEVICE_ID=0   # Change this to the GPU you want (0,1,2,...)

# Optional (only used for GeneOntology)
GO_BRANCH="MF"   # Options: MF, BP, CC

echo "===================================="
echo "Training Task: $TASK"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "LR: $LR"
echo "Using GPU: $DEVICE_ID"
echo "===================================="

# -----------------------------
# Set CUDA visibility
# -----------------------------
export CUDA_VISIBLE_DEVICES=$DEVICE_ID

# -----------------------------
# Run training
# -----------------------------

if [ "$TASK" == "GeneOntology" ]; then

    echo "GO Branch: $GO_BRANCH"

    python train_prime.py \
        --data_config $DATA_CONFIG \
        --model_config $MODEL_CONFIG \
        --task $TASK \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --go_branch $GO_BRANCH

else

    python train_prime.py \
        --data_config $DATA_CONFIG \
        --model_config $MODEL_CONFIG \
        --task $TASK \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR

fi