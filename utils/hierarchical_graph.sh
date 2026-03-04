#!/bin/bash

# ==========================================
# Hierarchical Graph Builder (Multi-Task)
# ==========================================

# ---- CHANGE ONLY THIS ----
TASK="FoldClassification"
# TASK="ECReaction"
# TASK="GeneOntology"
# TASK="AntibodyDevelopability"

# ---- Base directories ----
BASE_DIR="/home/dvnguye2/PRL/data/downstream_task_data"

# ---- Auto paths ----
PT_DIR="${BASE_DIR}/${TASK}/processed"
OUTPUT_DIR="${BASE_DIR}/${TASK}/graphs"

# ---- Create output directory automatically ----
mkdir -p "$OUTPUT_DIR"

echo "===================================="
echo "Task: $TASK"
echo "Input PT directory: $PT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "===================================="

# ---- Run Python ----
python ./utils/hierarchical_graph.py \
    --pt_dir "$PT_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "Finished building graphs for $TASK"