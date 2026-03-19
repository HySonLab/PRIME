#!/bin/bash

# ==========================================
# Testing Script (PRIME)
# ==========================================

# -----------------------------
# Config
# -----------------------------
DATA_CONFIG="/home/dvnguye2/PRL/config/data_config.yaml"
MODEL_CONFIG="/home/dvnguye2/PRL/config/model_config.yaml"

TASK="FoldClassification"   # FoldClassification | ECReaction | GeneOntology
BATCH_SIZE=32

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

# -----------------------------
# Optional for GO
# -----------------------------
GO_BRANCH="CC"   # MF | BP | CC

# -----------------------------
# Optional for FoldClassification
# -----------------------------
TEST_SET_SPLIT="superfamily"   # family | superfamily | fold

echo "===================================="
echo "Testing PRIME"
echo "Task: $TASK"
echo "Batch Size: $BATCH_SIZE"
echo "Active Levels: ${ACTIVE_LEVELS[@]}"
echo "Readout Level: $READOUT_LEVEL"
echo "===================================="

# -----------------------------
# Run testing
# -----------------------------

if [ "$TASK" == "GeneOntology" ]; then

    echo "GO Branch: $GO_BRANCH"

    python test_prime.py \
        --data_config $DATA_CONFIG \
        --model_config $MODEL_CONFIG \
        --task $TASK \
        --go_branch $GO_BRANCH \
        --batch_size $BATCH_SIZE \
        --active_levels ${ACTIVE_LEVELS[@]} \
        --readout_level $READOUT_LEVEL

elif [ "$TASK" == "FoldClassification" ]; then

    echo "Test Split: $TEST_SET_SPLIT"

    python test_prime.py \
        --data_config $DATA_CONFIG \
        --model_config $MODEL_CONFIG \
        --task $TASK \
        --test_set_split $TEST_SET_SPLIT \
        --batch_size $BATCH_SIZE \
        --active_levels ${ACTIVE_LEVELS[@]} \
        --readout_level $READOUT_LEVEL

else

    python test_prime.py \
        --data_config $DATA_CONFIG \
        --model_config $MODEL_CONFIG \
        --task $TASK \
        --batch_size $BATCH_SIZE \
        --active_levels ${ACTIVE_LEVELS[@]} \
        --readout_level $READOUT_LEVEL

fi