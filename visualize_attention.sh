#!/bin/bash

# ==========================================
# Attention Visualization Script (PRIME Cross-Attention)
# Collects attention for all tasks -> one combined figure
# ==========================================

set -e

# -----------------------------
# Config
# -----------------------------
DATA_CONFIG="./config/data_config.yaml"
MODEL_CONFIG="./config/model_config.yaml"
CKPT_DIR="./ckpts"

# -----------------------------
# GPU Selection
# -----------------------------
CUDA_DEVICE=0

# -----------------------------
# Settings
# -----------------------------
BATCH_SIZE=32
OUTPUT_DIR="./plots"

# -----------------------------
# Runs (= panels) to compute
# FOLD_family FOLD_superfamily FOLD_fold EC GO_MF GO_BP GO_CC
# Runs with an existing cache are skipped unless OVERWRITE=true
# -----------------------------
RUNS=("FOLD_family" "FOLD_superfamily" "FOLD_fold" "EC" "GO_MF" "GO_BP" "GO_CC")

OVERWRITE=false    # true -> recompute RUNS even if cached
PLOT_ONLY=false    # true -> skip evaluation, redraw from cached .npy

# -----------------------------
# Hierarchy
# -----------------------------
ACTIVE_LEVELS=("surface" "atom" "residue" "sse" "protein")

echo "===================================="
echo "Visualizing Attention Weights"
echo "Model:         PRIME_CrossAttention"
echo "Runs:          ${RUNS[*]}"
echo "CUDA device:   $CUDA_DEVICE"
echo "Batch Size:    $BATCH_SIZE"
echo "Active Levels: ${ACTIVE_LEVELS[*]}"
echo "Checkpoints:   $CKPT_DIR"
echo "Output dir:    $OUTPUT_DIR"
echo "Overwrite:     $OVERWRITE"
echo "Plot only:     $PLOT_ONLY"
echo "===================================="

# -----------------------------
# Build command
# -----------------------------
CMD=(python visualize_attention.py
    --data_config  "$DATA_CONFIG"
    --model_config "$MODEL_CONFIG"
    --ckpt_dir     "$CKPT_DIR"
    --batch_size   "$BATCH_SIZE"
    --output_dir   "$OUTPUT_DIR"
    --active_levels "${ACTIVE_LEVELS[@]}"
    --runs "${RUNS[@]}")

[ "$OVERWRITE" = true ] && CMD+=(--overwrite)
[ "$PLOT_ONLY" = true ] && CMD+=(--plot_only)

# -----------------------------
# Run
# -----------------------------
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE "${CMD[@]}"

echo "Done."