#!/bin/bash

# ==========================================
# PRIME Training Script
# ==========================================

# -----------------------------
# Config
# -----------------------------
DATA_CONFIG="./config/data_config.yaml"
MODEL_CONFIG="./config/model_config.yaml"

TASK="FoldClassification"   # FoldClassification | ECReaction | GeneOntology | BindingSite

BATCH_SIZE=32
EPOCHS=200
LR=1e-3

# -----------------------------
# GPU Selection
# -----------------------------
DEVICE_ID=0

# -----------------------------
# Optional (for GeneOntology)
# -----------------------------
GO_BRANCH="BP"   # MF | BP | CC

# -----------------------------
# Hierarchy Ablation
# -----------------------------
ACTIVE_LEVELS=("surface" "atom" "residue" "sse" "protein")
READOUT_LEVEL="residue"

# -----------------------------
# Cross-Attention Option
# -----------------------------
CROSS_ATTENTION="false"   # true or false

# -----------------------------
# Direction — for ablation study
# bidirectional | bottom_up_only | top_down_only
# -----------------------------
DIRECTION="bidirectional"

# -----------------------------
# Seed
# -----------------------------
SEED=1

# -----------------------------
# Resume Option
# -----------------------------
RESUME="false"

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
echo "Direction:       $DIRECTION"
echo "Seed:            $SEED"
echo "Resume:          $RESUME"
echo "===================================="

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
    --readout_level $READOUT_LEVEL \
    --direction $DIRECTION \
    --seed $SEED"

if [ "$CROSS_ATTENTION" == "true" ]; then
    CMD="$CMD --cross_attention"
fi

if [ "$TASK" == "GeneOntology" ]; then
    echo "GO Branch: $GO_BRANCH"
    CMD="$CMD --go_branch $GO_BRANCH"
fi

if [ "$RESUME" == "true" ]; then
    LEVEL_TAG="${ACTIVE_LEVELS[*]}"
    LEVEL_TAG="${LEVEL_TAG// /_}"
    MODEL_TAG="prime"
    if [ "$CROSS_ATTENTION" == "true" ]; then
        MODEL_TAG="prime_ca"
    fi

    # add direction suffix only if not bidirectional
    DIRECTION_TAG=""
    if [ "$DIRECTION" != "bidirectional" ]; then
        DIRECTION_TAG="_${DIRECTION}"
    fi

    if [ "$TASK" == "GeneOntology" ]; then
        RESUME_PATH="./ckpts/best_${MODEL_TAG}_${TASK}_${GO_BRANCH}_${LEVEL_TAG}${DIRECTION_TAG}_seed${SEED}.pt"
    else
        RESUME_PATH="./ckpts/best_${MODEL_TAG}_${TASK}_${LEVEL_TAG}${DIRECTION_TAG}_seed${SEED}.pt"
    fi

    if [ -f "$RESUME_PATH" ]; then
        echo "Resuming from: $RESUME_PATH"
        CMD="$CMD --resume \"$RESUME_PATH\""
    else
        echo "Warning: checkpoint not found at $RESUME_PATH — training from scratch"
    fi
fi

# -----------------------------
# Run
# -----------------------------
eval $CMD