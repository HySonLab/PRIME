#!/bin/bash

# ==========================================
# Hierarchical Graph Builder (Multi-Task)
# ==========================================

# ---- CHANGE ONLY THIS ----
# TASK="FoldClassification"
# TASK="ECReaction"
# TASK="GeneOntology"
TASK="BindingSite"

# ---- GPU device ----
CUDA_DEVICE=1

# ---- Base directories ----
BASE_DIR="/home/dvnguye2/PRL/data/downstream_task_data"

# ---- Auto paths ----
PT_DIR="${BASE_DIR}/${TASK}/processed"
OUTPUT_DIR="${BASE_DIR}/${TASK}/graphs"

# ---- Pretrained encoder paths (OPTIONAL) ----
ATOM_ENCODER_PATH="/home/dvnguye2/PRL/ckpts/atom_egnn_encoder.pt"
SURFACE_ENCODER_PATH="/home/dvnguye2/PRL/ckpts/surface_emnn_encoder.pt"

# If you want to disable, set to empty:
# ATOM_ENCODER_PATH=""
# SURFACE_ENCODER_PATH=""

# ---- Create output directory ----
mkdir -p "$OUTPUT_DIR"

echo "===================================="
echo "Task:        $TASK"
echo "CUDA device: $CUDA_DEVICE"
echo "Input:       $PT_DIR"
echo "Output:      $OUTPUT_DIR"
echo "===================================="

# ---- Build command ----
CMD="CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python ./utils/hierarchical_graph.py \
    --pt_dir \"$PT_DIR\" \
    --output_dir \"$OUTPUT_DIR\" \
    --task \"$TASK\""          

# ---- Conditionally add encoders ----
if [ -n "$ATOM_ENCODER_PATH" ]; then
    CMD="$CMD --atom_encoder_path \"$ATOM_ENCODER_PATH\""
    echo "Using atom encoder: $ATOM_ENCODER_PATH"
else
    echo "No atom encoder"
fi

if [ -n "$SURFACE_ENCODER_PATH" ]; then
    CMD="$CMD --surface_encoder_path \"$SURFACE_ENCODER_PATH\""
    echo "Using surface encoder: $SURFACE_ENCODER_PATH"
else
    echo "No surface encoder"
fi

# ---- Run ----
eval $CMD

echo "Finished building graphs for $TASK"