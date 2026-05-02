#!/bin/bash

# ==========================================
# PRIME Training Script
# ==========================================

# -----------------------------
# Config
# -----------------------------
DATA_CONFIG="./config/data_config.yaml"
MODEL_CONFIG="./config/model_config.yaml"

TASK="BindingSite"   # FoldClassification | ECReaction | GeneOntology | BindingSite

BATCH_SIZE=32
EPOCHS=200
LR=1e-4

# -----------------------------
# GPU Selection
# -----------------------------
DEVICE_ID=3

# -----------------------------
# Optional (for GeneOntology)
# -----------------------------
GO_BRANCH="CC"   # MF | BP | CC

# -----------------------------
# Hierarchy Ablation
# -----------------------------
ACTIVE_LEVELS=("surface" "atom" "residue" "sse" "protein")
READOUT_LEVEL="residue"

# -----------------------------
# Cross-Attention Option
# -----------------------------
CROSS_ATTENTION="false"  # true or false

# -----------------------------
# Resume Option
# Set to checkpoint path to resume, or empty to train from scratch
# -----------------------------
RESUME=""

echo "===================================="
echo "Training PRIME"
echo "Task:            $TASK"
echo "Batch Size:      $BATCH_SIZE"
echo "Epochs:          $EPOCHS"
echo "LR:              $LR"
echo "GPU:             $DEVICE_ID"
echo "Active Levels:   ${ACTIVE_LEVELS[@]}"
echo "Readout Level:   $READOUT_LEVEL"
echo "Cross Attention: $CROSS_ATTENTION"
echo "Resume:          ${RESUME:-none}"
echo "===================================="

# -----------------------------
# Set CUDA visibility
# -----------------------------
export CUDA_VISIBLE_DEVICES=$DEVICE_ID

# -----------------------------
# Build base command
# -----------------------------
CMD="python train_prime.py \
    --data_config $DATA_CONFIG \
    --model_config $MODEL_CONFIG \
    --task $TASK \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --active_levels ${ACTIVE_LEVELS[@]} \
    --readout_level $READOUT_LEVEL"

# add cross_attention flag if enabled
if [ "$CROSS_ATTENTION" == "true" ]; then
    CMD="$CMD --cross_attention"
fi

# add go_branch if GeneOntology
if [ "$TASK" == "GeneOntology" ]; then
    echo "GO Branch: $GO_BRANCH"
    CMD="$CMD --go_branch $GO_BRANCH"
fi

# add resume if set
if [ -n "$RESUME" ]; then
    echo "Resuming from: $RESUME"
    CMD="$CMD --resume \"$RESUME\""
fi

# -----------------------------
# Run
# -----------------------------
eval $CMD