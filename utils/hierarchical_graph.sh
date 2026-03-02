#!/bin/bash

# =====================================
# Config
# =====================================
CONFIG="/home/dvnguye2/PRL/config/model_config.yaml"
PDB_DIR="/home/dvnguye2/PRL/data/pretrain_data/pdb_data"
PDB_PATH="/home/dvnguye2/PRL/data/pretrain_data/pdb_data/AF-A0A009IHW8-F1-model_v6.pdb"
DEVICE=0

# =====================================
# Run
# =====================================
CUDA_VISIBLE_DEVICES=$DEVICE python hierarchical_graph.py \
    --config $CONFIG \
    --pdb_path $PDB_PATH \
    --use_pretrained_atom \
    --use_pretrained_surface \
    # --pdb_dir $PDB_DIR