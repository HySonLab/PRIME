#!/bin/bash

# ==========================================
# Testing Script
# ==========================================

# -----------------------------
# Config
# -----------------------------
CONFIG="/home/dvnguye2/PRL/config/data_config.yaml"
TASK="FoldClassification"   # Option: FoldClassification, ECReaction, GeneOntology
BATCH_SIZE=32

# Optional for GO
GO_BRANCH="CC"   # Option: MF, BP, CC

# Optional for FoldClassification
TEST_SET_SPLIT="fold"   # Example: family / superfamily / fold

echo "===================================="
echo "Testing Task: $TASK"
echo "Batch Size: $BATCH_SIZE"
echo "===================================="

# -----------------------------
# Run testing
# -----------------------------

if [ "$TASK" == "GeneOntology" ]; then

    echo "GO Branch: $GO_BRANCH"

    python test.py \
        --config $CONFIG \
        --task $TASK \
        --go_branch $GO_BRANCH \
        --batch_size $BATCH_SIZE

elif [ "$TASK" == "FoldClassification" ]; then

    echo "Test Split: $TEST_SET_SPLIT"

    python test.py \
        --config $CONFIG \
        --task $TASK \
        --test_set_split $TEST_SET_SPLIT \
        --batch_size $BATCH_SIZE

else

    python test.py \
        --config $CONFIG \
        --task $TASK \
        --batch_size $BATCH_SIZE

fi