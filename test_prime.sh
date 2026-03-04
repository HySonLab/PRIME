#!/bin/bash

# ==========================================
# Testing Script (PRIME)
# ==========================================

# -----------------------------
# Config
# -----------------------------
DATA_CONFIG="/home/dvnguye2/PRL/config/data_config.yaml"
MODEL_CONFIG="/home/dvnguye2/PRL/config/model_config.yaml"
TASK="FoldClassification"   # Option: FoldClassification, ECReaction, GeneOntology
BATCH_SIZE=32

# Optional for GO
GO_BRANCH="CC"   # Option: MF, BP, CC

# Optional for FoldClassification
TEST_SET_SPLIT="family"   # Example: family / superfamily / fold

echo "===================================="
echo "Testing PRIME"
echo "Task: $TASK"
echo "Batch Size: $BATCH_SIZE"
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
        --batch_size $BATCH_SIZE

elif [ "$TASK" == "FoldClassification" ]; then

    echo "Test Split: $TEST_SET_SPLIT"

    python test_prime.py \
        --data_config $DATA_CONFIG \
        --model_config $MODEL_CONFIG \
        --task $TASK \
        --test_set_split $TEST_SET_SPLIT \
        --batch_size $BATCH_SIZE

else

    python test_prime.py \
        --data_config $DATA_CONFIG \
        --model_config $MODEL_CONFIG \
        --task $TASK \
        --batch_size $BATCH_SIZE

fi