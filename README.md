# PRIME: Protein Representation via Physics-Informed Multiscale Equivariant Hierarchies

PRIME is a hierarchical graph representation learning framework that models proteins as a nested family of five physically grounded structural graphs spanning surface, atomic, residue, secondary-structure, and protein levels.

## Overview

![PRIME Framework](./figures/PRIME.png)

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

**Step 1: Download processed data from ProteinWorkshop**

Download the preprocessed datasets and standard splits from the [ProteinWorkshop repository](https://github.com/a-r-j/ProteinWorkshop). Follow their instructions to download the datasets for the tasks you wish to evaluate:
- Fold Classification
- Reaction Class Prediction
- Gene Ontology Prediction
- PPI Site Prediction

**Step 2: Build hierarchical graphs**

Open `utils/hierarchical_graph.sh`, set `TASK` to your desired task and update the paths, then run:

```bash
bash utils/hierarchical_graph.sh
```

This script processes the raw protein structures and builds the five-level hierarchical graph representation for each protein in the dataset.

## Training

## Training

Open `train_prime.sh` and configure the following fields to match your setup:

```bash
TASK="FoldClassification"   # FoldClassification | ECReaction | GeneOntology | BindingSite
SEED=1                      # use 1, 2, 3 for 3-seed reporting
CROSS_ATTENTION="false"     # set to "true" for PRIME (w/ CA) variant
GO_BRANCH="MF"              # only for GeneOntology: MF | BP | CC

# Hierarchy ablation — controls which levels are active
ACTIVE_LEVELS=("surface" "atom" "residue" "sse" "protein")

# Readout level — which level is used for prediction
READOUT_LEVEL="residue"     # surface | atom | residue | sse | protein

# Direction ablation — controls message passing direction
DIRECTION="bidirectional"   # bidirectional | bottom_up_only | top_down_only

# Resume training from an existing checkpoint
RESUME="false"              # set to "true" to auto-resume from latest checkpoint
```

Then run:

```bash
bash train_prime.sh
```

## Testing

Open `test_prime.sh` and configure the task and seed to match your training run, then run:

```bash
bash test_prime.sh
```

The script automatically resolves the checkpoint path from the training configuration.

## Expected Outputs

Training logs and checkpoints are saved automatically to:

```
./logs/training_log_prime_{task}_{level_tag}_seed{N}.txt
./ckpts/best_prime_{task}_{level_tag}_seed{N}.pt
```

where `{level_tag}` is `surface_atom_residue_sse_protein` by default and `{N}` is the random seed.

## Configuration

All model and training hyperparameters are managed through the configuration files in the `config/` directory:
- `config/data_config.yaml` — dataset paths and split files
- `config/model_config.yaml` — model architecture hyperparameters

> **Important:** Update the `data_root` field in `config/data_config.yaml` to point to your local data directory before running any scripts.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.