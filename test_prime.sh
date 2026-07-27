#!/bin/bash

# ==========================================
# Testing Script (PRIME)
# ==========================================

# -----------------------------
# Config
# -----------------------------
DATA_CONFIG="./config/data_config.yaml"
MODEL_CONFIG="./config/model_config.yaml"

TASK="GeneOntology"   # FoldClassification | ECReaction | GeneOntology | BindingSite
BATCH_SIZE=32
CUDA_DEVICE=2

# -----------------------------
# Hierarchy Ablation
# -----------------------------
ACTIVE_LEVELS=("surface" "atom" "residue" "sse" "protein")
READOUT_LEVEL="residue"

# -----------------------------
# Cross-Attention Option
# -----------------------------
CROSS_ATTENTION="false"   # true | false

# -----------------------------
# Seed — must match training seed
# -----------------------------
SEED=3

# -----------------------------
# Optional for GO
# -----------------------------
GO_BRANCH="BP"   # MF | BP | CC

# -----------------------------
# Optional for FoldClassification
# -----------------------------
TEST_SET_SPLIT="family"   # family | superfamily | fold

echo "===================================="
echo "Testing PRIME"
echo "Task:            $TASK"
echo "Batch Size:      $BATCH_SIZE"
echo "CUDA device:     $CUDA_DEVICE"
echo "Active Levels:   ${ACTIVE_LEVELS[@]}"
echo "Readout Level:   $READOUT_LEVEL"
echo "Cross Attention: $CROSS_ATTENTION"
echo "Seed:            $SEED"
echo "===================================="

# -----------------------------
# Build base command
# -----------------------------
CMD="CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python test_prime.py \
    --data_config $DATA_CONFIG \
    --model_config $MODEL_CONFIG \
    --task $TASK \
    --batch_size $BATCH_SIZE \
    --active_levels ${ACTIVE_LEVELS[@]} \
    --readout_level $READOUT_LEVEL \
    --seed $SEED"

if [ "$CROSS_ATTENTION" == "true" ]; then
    CMD="$CMD --cross_attention"
fi

if [ "$TASK" == "GeneOntology" ]; then
    echo "GO Branch: $GO_BRANCH"
    CMD="$CMD --go_branch $GO_BRANCH"
elif [ "$TASK" == "FoldClassification" ]; then
    echo "Test Split: $TEST_SET_SPLIT"
    CMD="$CMD --test_set_split $TEST_SET_SPLIT"
fi

# -----------------------------
# Run
# -----------------------------
eval $CMD
