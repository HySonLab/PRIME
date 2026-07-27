import torch
import torch.nn as nn
from models.models import PRIME, PRIME_CrossAttention
import yaml
import argparse
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.hierarchical_graph import *
from utils.helpers import *


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

@torch.no_grad()
def test_model(
    model,
    loader,
    task_type,
    num_classes,
    device,
    task_level="graph"
):
    model.eval()
    metric = get_metric(task_type, num_classes, device)

    for batch in tqdm(loader, desc="Testing", leave=False):

        # ==================================================
        # Node-level task (e.g. BindingSite)
        # ==================================================
        if task_level == "node":
            for sample in batch:
                graph  = sample["graph"]
                labels = sample["label"].to(device)

                logits = model(graph).squeeze(-1)
                probs  = torch.sigmoid(logits).cpu()
                metric.update(probs, labels.long().cpu())

        # ==================================================
        # Graph-level task
        # ==================================================
        else:
            logits_list = []
            labels_list = []

            for sample in batch:
                logits = model(sample["graph"])
                logits_list.append(logits.squeeze(0))

                if task_type == "multilabel_classification":
                    y = to_multihot(sample["label"], num_classes, device)
                    labels_list.append(y)
                else:
                    labels_list.append(
                        torch.tensor(
                            sample["label"], dtype=torch.long, device=device
                        )
                    )

            logits = torch.stack(logits_list, dim=0)
            labels = torch.stack(labels_list, dim=0)
            
            metric.update(logits, labels)

    return metric.compute().item()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_config",    type=str, default="config/data_config.yaml")
    parser.add_argument("--model_config",   type=str, default="config/model_config.yaml")
    parser.add_argument("--task",           type=str, default="FoldClassification")
    parser.add_argument("--go_branch",      type=str, default=None)
    parser.add_argument("--test_set_split", type=str, default=None)
    parser.add_argument("--batch_size",     type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--active_levels",
        nargs="+",
        default=["surface", "atom", "residue", "sse", "protein"]
    )
    parser.add_argument(
        "--readout_level",
        type=str,
        default="residue"
    )

    parser.add_argument(
        "--cross_attention",
        action="store_true",
        default=False,
        help="Use PRIME_CrossAttention instead of standard PRIME"
    )

    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # Load configs
    # --------------------------------------------------
    data_config  = load_config(args.data_config)
    task_cfg     = data_config["tasks"][args.task]
    task_type    = task_cfg["task_type"]
    task_level   = task_cfg.get("task_level", "graph")
    model_config = load_config(args.model_config)

    # --------------------------------------------------
    # Num classes
    # --------------------------------------------------
    if args.task == "GeneOntology":
        if args.go_branch is None:
            raise ValueError("Must specify --go_branch for GeneOntology")
        num_classes = task_cfg["num_classes"][args.go_branch]
    elif task_type == "node_classification":
        num_classes = 2
    else:
        num_classes = task_cfg["num_classes"]

    # --------------------------------------------------
    # Checkpoint path — matches training naming
    # --------------------------------------------------
    level_tag = "_".join(args.active_levels)
    model_tag = "prime_ca" if args.cross_attention else "prime"
    seed_tag  = f"seed{args.seed}"

    if args.task == "GeneOntology":
        ckpt_path = f"./ckpts/best_{model_tag}_{args.task}_{args.go_branch}_{level_tag}_{seed_tag}.pt"
    else:
        ckpt_path = f"./ckpts/best_{model_tag}_{args.task}_{level_tag}_{seed_tag}.pt"

    print("=" * 50)
    print(f"Task:            {args.task}")
    print(f"Task level:      {task_level}")
    print(f"Num classes:     {num_classes}")
    print(f"Active levels:   {args.active_levels}")
    print(f"Readout level:   {args.readout_level}")
    print(f"Cross attention: {args.cross_attention}")
    print(f"Checkpoint:      {ckpt_path}")
    print("=" * 50)

    # --------------------------------------------------
    # Test loader
    # --------------------------------------------------
    test_loader = build_graph_dataloaders(
        args.data_config,
        args.task,
        batch_size=args.batch_size,
        test_only=True,
        test_set_split=args.test_set_split,
        device=device,
        go_branch=args.go_branch,
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    if args.cross_attention:
        model = PRIME_CrossAttention(
            num_classes=num_classes,
            input_dims=model_config["hierarchical"]["input_dims"],
            active_levels=args.active_levels,
            hidden_dim=model_config["hierarchical"]["hidden_dim"],
            encoder_layers=model_config["hierarchical"]["n_layers"],
            head_hidden_dim=model_config["head"][args.task]["hidden_dim"],
            head_layers=model_config["head"][args.task]["num_layers"],
            dropout=model_config["head"][args.task]["dropout"],
            task_level=task_level,
        )
    else:
        model = PRIME(
            num_classes=num_classes,
            input_dims=model_config["hierarchical"]["input_dims"],
            active_levels=args.active_levels,
            readout_level=args.readout_level,
            hidden_dim=model_config["hierarchical"]["hidden_dim"],
            encoder_layers=model_config["hierarchical"]["n_layers"],
            head_hidden_dim=model_config["head"][args.task]["hidden_dim"],
            head_layers=model_config["head"][args.task]["num_layers"],
            dropout=model_config["head"][args.task]["dropout"],
            task_level=task_level,
        )

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------
    score = test_model(
        model,
        test_loader,
        task_type,
        num_classes,
        device,
        task_level=task_level,
    )

    # --------------------------------------------------
    # Print result
    # --------------------------------------------------
    metric_name = {
        "multilabel_classification": "Fmax (macro-F1)",
        "node_classification":       "AUPRC",
        "multiclass_classification": "Accuracy",
    }.get(task_type, "Score")

    print(f"\n{metric_name}: {score:.4f}")
