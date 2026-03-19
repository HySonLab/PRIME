#!/bin/bash

# ==========================================
# PRIME Training Script
# ==========================================

# -----------------------------
# Config
# -----------------------------
DATA_CONFIG="/home/dvnguye2/PRL/config/data_config.yaml"
MODEL_CONFIG="/home/dvnguye2/PRL/config/model_config.yaml"

TASK="FoldClassification"   # FoldClassification | ECReaction | GeneOntology

BATCH_SIZE=32
EPOCHS=150
LR=1e-4

# -----------------------------
# GPU Selection
# -----------------------------
DEVICE_ID=3

# -----------------------------
# Optional (for GeneOntology)
# -----------------------------
GO_BRANCH="MF"   # MF | BP | CC

# -----------------------------
# Hierarchy Ablation
# -----------------------------

ACTIVE_LEVELS=("surface" "atom" "residue" "sse" "protein")
READOUT_LEVEL="protein"

# Example ablations:
# ACTIVE_LEVELS=("residue")
# READOUT_LEVEL="residue"

# ACTIVE_LEVELS=("atom" "residue")
# READOUT_LEVEL="residue"

echo "===================================="
echo "Training PRIME"
echo "Task: $TASK"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "LR: $LR"
echo "GPU: $DEVICE_ID"
echo "Active Levels: ${ACTIVE_LEVELS[@]}"
echo "Readout Level: $READOUT_LEVEL"
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
        --go_branch $GO_BRANCH \
        --active_levels ${ACTIVE_LEVELS[@]} \
        --readout_level $READOUT_LEVEL

else

    python train_prime.py \
        --data_config $DATA_CONFIG \
        --model_config $MODEL_CONFIG \
        --task $TASK \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --active_levels ${ACTIVE_LEVELS[@]} \
        --readout_level $READOUT_LEVEL

fi