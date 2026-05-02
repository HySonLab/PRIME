#!/bin/bash

# ==========================================
# Attention Visualization Script (PRIME Cross-Attention)
# ==========================================

# -----------------------------
# Config
# -----------------------------
DATA_CONFIG="/home/dvnguye2/PRL/config/data_config.yaml"
MODEL_CONFIG="/home/dvnguye2/PRL/config/model_config.yaml"

# -----------------------------
# GPU Selection
# -----------------------------
CUDA_DEVICE=0

# -----------------------------
# Settings
# -----------------------------
BATCH_SIZE=32
OUTPUT_DIR="/home/dvnguye2/PRL/plots"

# -----------------------------
# Task settings
# FoldClassification | ECReaction | GeneOntology | BindingSite
# -----------------------------
TASK="ECReaction"
GO_BRANCH="MF"   # MF | BP | CC — only used for GeneOntology

# -----------------------------
# Hierarchy
# -----------------------------
ACTIVE_LEVELS=("surface" "atom" "residue" "sse" "protein")

echo "===================================="
echo "Visualizing Attention Weights"
echo "Model:        PRIME_CrossAttention"
echo "Task:         $TASK"
echo "CUDA device:  $CUDA_DEVICE"
echo "Batch Size:   $BATCH_SIZE"
echo "Active Levels: ${ACTIVE_LEVELS[@]}"
echo "Output dir:   $OUTPUT_DIR"
echo "===================================="

# -----------------------------
# Build command
# -----------------------------
CMD="CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python visualize_attention.py \
    --data_config $DATA_CONFIG \
    --model_config $MODEL_CONFIG \
    --task $TASK \
    --batch_size $BATCH_SIZE \
    --output_dir $OUTPUT_DIR \
    --active_levels ${ACTIVE_LEVELS[@]}"

# add go_branch if GeneOntology
if [ "$TASK" == "GeneOntology" ]; then
    echo "GO Branch: $GO_BRANCH"
    CMD="$CMD --go_branch $GO_BRANCH"
fi

# -----------------------------
# Run
# -----------------------------
eval $CMD

echo "Done."