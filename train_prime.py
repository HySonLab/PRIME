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

def filter_go_labels(y, num_classes):

    offsets = {
        489: (0, 489),
        1943: (489, 489 + 1943),
        320: (489 + 1943, 489 + 1943 + 320)
    }

    if num_classes not in offsets:
        raise ValueError("Unknown GO num_classes")

    start, end = offsets[num_classes]

    return [
        cls - start
        for cls in y
        if start <= cls < end
    ]

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    task_type,
    num_classes
):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)

    for batch in pbar:

        # ---------------------------------------
        # Prepare labels
        # ---------------------------------------
        if task_type == "multilabel_classification":
            labels = torch.zeros(len(batch), num_classes, device=device)

            for i, sample in enumerate(batch):
                y = filter_go_labels(sample["label"], num_classes)
                labels[i, y] = 1.0

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
        # Backward
        # ---------------------------------------
        optimizer.zero_grad(set_to_none=True)

        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    task_type,
    num_classes
):
    model.eval()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Evaluating", leave=False)

    for batch in pbar:

        # ---------------------------------------
        # Prepare labels
        # ---------------------------------------
        if task_type == "multilabel_classification":
            labels = torch.zeros(len(batch), num_classes, device=device)

            for i, sample in enumerate(batch):
                y = sample["label"]
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

        loss = criterion(logits, labels)
        total_loss += loss.item()

    return total_loss / len(loader)

def train_prime(
    model,
    train_loader,
    val_loader,
    task_type,
    task_name,
    num_classes,
    criterion,
    output_path,
    log_path="training_log.txt",
    epochs=150,
    lr=1e-4,
    patience_es=10,
    patience_lr=5,
):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.6,
        patience=patience_lr
    )

    best_val_loss = float("inf")
    early_stop_counter = 0

    # ---- Open log file ----
    with open(log_path, "w") as log_file:
        log_file.write(f"Task: {task_name}\n")
        log_file.write(f"Num classes: {num_classes}\n")
        log_file.write(f"LR: {lr}\n")
        log_file.write(f"Epochs: {epochs}\n")
        log_file.write("=" * 50 + "\n")
        log_file.flush()

        for epoch in range(epochs):

            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                task_type,
                num_classes
            )

            val_loss = evaluate(
                model,
                val_loader,
                criterion,
                device,
                task_type,
                num_classes
            )

            scheduler.step(val_loss)

            # ---- Write to file ----
            log_file.write(f"Epoch {epoch+1:03d}\n")
            log_file.write(f"Train Loss: {train_loss:.6f}\n")
            log_file.write(f"Val   Loss: {val_loss:.6f}\n")
            log_file.write("-" * 40 + "\n")
            log_file.flush()

            print(f"Epoch {epoch+1:03d}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val   Loss: {val_loss:.4f}")
            print("-" * 40)
            log_file.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                torch.save(model.state_dict(), output_path)
                log_file.write("New best model saved.\n")
            else:
                early_stop_counter += 1
                log_file.write(f"EarlyStopping counter: {early_stop_counter}/{patience_es}\n")

            if early_stop_counter >= patience_es:
                log_file.write("Early stopping triggered.\n")
                break

        log_file.write(f"Best Val Loss: {best_val_loss:.6f}\n")
        log_file.write("Training Finished\n")

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", type=str, default="config/data_config.yaml")
    parser.add_argument("--model_config", type=str, default="config/model_config.yaml")
    parser.add_argument("--task", type=str, default="FoldClassification")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--go_branch", type=str, default=None, help="MF | BP | CC (required for GeneOntology)")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # Load YAML config
    # --------------------------------------------------
    data_config = load_config(args.data_config)
    task_cfg = data_config["tasks"][args.task]
    
    model_config = load_config(args.model_config)

    # --------------------------------------------------
    # Determine num_classes
    # --------------------------------------------------
    if args.task == "GeneOntology":

        if args.go_branch is None:
            raise ValueError("GeneOntology requires --go_branch (MF/BP/CC)")

        if args.go_branch not in task_cfg["num_classes"]:
            raise ValueError(f"Invalid go_branch: {args.go_branch}")

        num_classes = task_cfg["num_classes"][args.go_branch]

    else:
        num_classes = task_cfg["num_classes"]
    
    
    # --------------------------------------------------
    # Determine criterion
    # --------------------------------------------------

    task_type = task_cfg["task_type"]

    if task_type == "multilabel_classification":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------
    # Build Graph DataLoader
    # --------------------------------------------------

    train_loader, val_loader = build_graph_dataloaders(
        args.data_config,
        args.task,
        batch_size=args.batch_size,
        device=device,
    )

    # --------------------------------------------------
    # Initialize PRIME model
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

    # --------------------------------------------------
    # Output paths
    # --------------------------------------------------

    if args.task == "GeneOntology":
        output_path = f"/home/dvnguye2/PRL/ckpts/best_prime_{args.task}_{args.go_branch}.pt"
        log_path = f"/home/dvnguye2/PRL/logs/training_log_prime_{args.task}_{args.go_branch}.txt"
    else:
        output_path = f"/home/dvnguye2/PRL/ckpts/best_prime_{args.task}.pt"
        log_path = f"/home/dvnguye2/PRL/logs/training_log_prime_{args.task}.txt"

    # --------------------------------------------------
    # Train PRIME
    # --------------------------------------------------

    train_prime(
        model,
        train_loader,
        val_loader,
        task_type=task_type,
        task_name=args.task,
        num_classes=num_classes,
        criterion=criterion,
        output_path=output_path,
        log_path=log_path,
        epochs=args.epochs,
        lr=args.lr
    )