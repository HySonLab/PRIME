import torch
import torch.nn as nn
from models.models import PRIME
import yaml
from torch.utils.data import DataLoader
from utils.dataset import build_graph_dataloaders
import argparse
from tqdm import tqdm
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.models.esmc import ESMC

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.hierarchical_graph import *
from torchmetrics.classification import MulticlassAccuracy, MultilabelF1Score

# --------------------------------------------------
# Load YAML config
# --------------------------------------------------

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# --------------------------------------------------
# GO label filtering
# --------------------------------------------------

def filter_go_labels(y, num_classes):

    offsets = {
        489: (0, 489),
        1943: (489, 489 + 1943),
        320: (489 + 1943, 489 + 1943 + 320)
    }

    start, end = offsets[num_classes]

    return [
        cls - start
        for cls in y
        if start <= cls < end
    ]
    
@torch.no_grad()
def test_model(
    model,
    loader,
    task_type,
    num_classes,
    device
):
    model.eval()

    if task_type == "multilabel_classification":
        metric = MultilabelF1Score(
            num_labels=num_classes,
            average="macro"
        ).to(device)
    else:
        metric = MulticlassAccuracy(
            num_classes=num_classes
        ).to(device)

    pbar = tqdm(loader, desc="Testing", leave=False)

    for batch in pbar:

        # ---------------------------------------
        # Prepare labels
        # ---------------------------------------
        if task_type == "multilabel_classification":

            labels = torch.zeros(len(batch), num_classes, device=device)

            for i, sample in enumerate(batch):
                y = sample["label"]

                if isinstance(y, torch.Tensor):
                    y = y.tolist()

                y = filter_go_labels(y, num_classes)

                for cls in y:
                    labels[i, cls] = 1.0

        else:
            labels = torch.tensor(
                [sample["label"] for sample in batch],
                dtype=torch.long,
                device=device
            )

        # ---------------------------------------
        # Forward pass
        # ---------------------------------------
        logits_list = []

        for sample in batch:
            logits = model(sample["graph"])
            logits_list.append(logits.squeeze(0))

        logits = torch.stack(logits_list, dim=0)

        # ---------------------------------------
        # Update metric
        # ---------------------------------------
        if task_type == "multilabel_classification":
            metric.update(torch.sigmoid(logits), labels)
        else:
            metric.update(logits, labels)

    score = metric.compute().item()
    return score

# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", type=str, default="config/data_config.yaml")
    parser.add_argument("--model_config", type=str, default="config/model_config.yaml")
    parser.add_argument("--task", type=str, default="FoldClassification")
    parser.add_argument("--go_branch", type=str, default=None)
    parser.add_argument("--test_set_split", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # Load configs
    # --------------------------------------------------

    data_config = load_config(args.data_config)
    task_cfg = data_config["tasks"][args.task]
    task_type = task_cfg["task_type"]

    model_config = load_config(args.model_config)

    # --------------------------------------------------
    # Determine num_classes + checkpoint
    # --------------------------------------------------

    if args.task == "GeneOntology":

        if args.go_branch is None:
            raise ValueError("Must specify --go_branch for GeneOntology")

        num_classes = task_cfg["num_classes"][args.go_branch]
        ckpt_path = f"/home/dvnguye2/PRL/ckpts/best_prime_{args.task}_{args.go_branch}.pt"

    elif args.task == "FoldClassification":

        if args.test_set_split is None:
            raise ValueError("Must specify --test_set_split for FoldClassification")

        num_classes = task_cfg["num_classes"]
        ckpt_path = f"/home/dvnguye2/PRL/ckpts/best_prime_{args.task}.pt"

    else:
        num_classes = task_cfg["num_classes"]
        ckpt_path = f"/home/dvnguye2/PRL/ckpts/best_prime_{args.task}.pt"

    print("=" * 50)
    print("Num Classes:", num_classes)
    print("Checkpoint:", ckpt_path)
    print("=" * 50)

    # --------------------------------------------------
    # Build Test Loader
    # --------------------------------------------------

    test_loader = build_graph_dataloaders(
        args.data_config,
        args.task,
        batch_size=args.batch_size,
        test_only=True,
        test_set_split=args.test_set_split,
        device=device
    )

    # --------------------------------------------------
    # Load Model
    # --------------------------------------------------

    model = PRIME(
        num_classes=num_classes,
        input_dims=model_config["hierarchical"]["input_dims"],
        hidden_dim=model_config["hierarchical"]["hidden_dim"],
        encoder_layers=model_config["hierarchical"]["n_layers"],
        head_hidden_dim=model_config["head"][args.task]["hidden_dim"],
        head_layers=model_config["head"][args.task]["num_layers"],
        dropout=model_config["head"][args.task]["dropout"]
    )

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # --------------------------------------------------
    # Run test
    # --------------------------------------------------

    score = test_model(
        model,
        test_loader,
        task_type,
        num_classes,
        device
    )

    if task_type == "multilabel_classification":
        print(f"Fmax (macro-F1): {score:.4f}")
    else:
        print(f"Accuracy: {score:.4f}")