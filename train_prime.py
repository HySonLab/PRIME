import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR
from models.models import PRIME, PRIME_CrossAttention
import argparse
from tqdm import tqdm
import random

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.hierarchical_graph import *
from utils.helpers import *

def train_one_epoch(
    model, loader, optimizer, criterion,
    device, task_type, num_classes, metric, grad_clip=1.0,
    task_level="graph"
):
    model.train()
    metric.reset()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):

        if task_level == "node":
            optimizer.zero_grad(set_to_none=True)
            batch_loss = torch.tensor(0.0, device=device)

            for sample in batch:
                graph  = sample["graph"]
                labels = sample["label"].float().to(device)

                logits = model(graph).squeeze(-1)              # (N_res,)
                loss   = criterion(logits, labels)
                batch_loss += loss

                with torch.no_grad():
                    probs = torch.sigmoid(logits).cpu()
                    metric.update(probs, labels.long().cpu())

            (batch_loss / len(batch)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += (batch_loss / len(batch)).item()

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
                        torch.tensor(sample["label"], dtype=torch.long, device=device)
                    )

            logits = torch.stack(logits_list, dim=0)
            labels = torch.stack(labels_list, dim=0)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()

            with torch.no_grad():
                metric.update(logits, labels)

    return total_loss / len(loader), metric.compute().item()

@torch.no_grad()
def evaluate(
    model, loader, criterion,
    device, task_type, num_classes, metric,
    task_level="graph"
):
    model.eval()
    metric.reset()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Evaluating", leave=False):

        if task_level == "node":
            batch_loss = 0.0

            for sample in batch:
                graph  = sample["graph"]
                labels = sample["label"].float().to(device)

                logits = model(graph).squeeze(-1) 
                loss   = criterion(logits, labels)
                batch_loss += loss.item()

                probs = torch.sigmoid(logits).cpu()
                metric.update(probs, labels.long().cpu())

            total_loss += batch_loss / len(batch)

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
                        torch.tensor(sample["label"], dtype=torch.long, device=device)
                    )

            logits = torch.stack(logits_list, dim=0)
            labels = torch.stack(labels_list, dim=0)

            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            metric.update(logits, labels)

    return total_loss / len(loader), metric.compute().item()

def train_prime(
    model,
    train_loader,
    val_loader,
    task_type,
    task_name,
    num_classes,
    criterion,
    output_path,
    device,
    log_path="training_log.txt",
    epochs=100,
    lr=1e-3,
    weight_decay=1e-4,
    patience_es=20,
    factor=0.6,
    patience_lr=5,
    grad_clip=5.0,
    warmup_epochs=3,
    task_level="graph",
):
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=factor,
        patience=patience_lr
    )

    train_metric = get_metric(task_type, num_classes, device)
    val_metric   = get_metric(task_type, num_classes, device)

    best_val_metric    = -float("inf")
    early_stop_counter = 0

    with open(log_path, "w") as log_file:
        log_file.write(f"Task:        {task_name}\n")
        log_file.write(f"Num classes: {num_classes}\n")
        log_file.write(f"LR:          {lr} | WD: {weight_decay}\n")
        log_file.write(f"Epochs:      {epochs}\n")
        log_file.write("=" * 50 + "\n")
        log_file.flush()

        for epoch in range(epochs):

            train_loss, train_score = train_one_epoch(
                model, train_loader, optimizer, criterion,
                device, task_type, num_classes, train_metric, grad_clip,
                task_level=task_level
            )

            val_loss, val_score = evaluate(
                model, val_loader, criterion,
                device, task_type, num_classes, val_metric,
                task_level=task_level
            )

            if epoch < warmup_epochs:
                warmup.step()
            else:
                plateau.step(val_score)

            current_lr = optimizer.param_groups[0]['lr']

            log_file.write(f"Epoch {epoch+1:03d}\n")
            log_file.write(f"Train Loss: {train_loss:.6f} | Train Metric: {train_score:.6f}\n")
            log_file.write(f"Val   Loss: {val_loss:.6f} | Val   Metric: {val_score:.6f}\n")
            log_file.write(f"LR: {current_lr:.2e}\n")
            log_file.write("-" * 40 + "\n")
            log_file.flush()

            print(f"Epoch {epoch+1:03d} | "
                  f"Train Loss: {train_loss:.4f} | Train: {train_score:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val: {val_score:.4f} | "
                  f"LR: {current_lr:.2e}")

            if val_score > best_val_metric:
                best_val_metric    = val_score
                early_stop_counter = 0
                torch.save(model.state_dict(), output_path)
                log_file.write("New best model saved.\n")
                print("  ✓ Best model saved")
            else:
                early_stop_counter += 1
                log_file.write(f"EarlyStopping: {early_stop_counter}/{patience_es}\n")
                if early_stop_counter >= patience_es:
                    log_file.write("Early stopping triggered.\n")
                    print("Early stopping triggered.")
                    break

            if (epoch + 1) % 1 == 0:
                ckpt_path = output_path.replace(".pt", f"_epoch{epoch+1}.pt")
                torch.save(model.state_dict(), ckpt_path)
                log_file.write(f"Checkpoint saved at epoch {epoch+1}.\n")
                print(f"Checkpoint saved at epoch {epoch+1}")

        log_file.write(f"Best Val Metric: {best_val_metric:.6f}\n")
        log_file.write("Training Finished\n")

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config",  type=str, default="config/data_config.yaml")
    parser.add_argument("--model_config", type=str, default="config/model_config.yaml")
    parser.add_argument("--task",         type=str, default="FoldClassification")
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip",    type=float, default=5.0)
    parser.add_argument("--go_branch",    type=str,   default=None)
    parser.add_argument("--pos_weight",   type=float, default=None)
    parser.add_argument("--resume",       type=str,   default=None)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument(
        "--active_levels",
        nargs="+",
        default=["surface", "atom", "residue", "sse", "protein"],
    )
    parser.add_argument("--readout_level",  type=str, default="residue")
    parser.add_argument(
        "--cross_attention",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="bidirectional",
        choices=["bidirectional", "bottom_up_only", "top_down_only"],
        help="Message passing direction (bidirectional=default, no suffix in ckpt name)"
    )

    args   = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # Load configs
    # --------------------------------------------------
    data_config  = load_config(args.data_config)
    task_cfg     = data_config["tasks"][args.task]
    model_config = load_config(args.model_config)
    task_type    = task_cfg["task_type"]
    task_level   = task_cfg.get("task_level", "graph")

    # --------------------------------------------------
    # Num classes
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
    # DataLoaders
    # --------------------------------------------------
    train_loader, val_loader = build_graph_dataloaders(
        args.data_config,
        args.task,
        batch_size=args.batch_size,
        device=device,
        go_branch=args.go_branch,
    )

    # --------------------------------------------------
    # Loss function
    # --------------------------------------------------
    if task_type == "node_classification":
        if args.pos_weight is not None:
            pw = torch.tensor([args.pos_weight], device=device)
        else:
            pos_count, neg_count = 0, 0
            for batch in train_loader:
                for sample in batch:
                    labels     = sample["label"]
                    pos_count += labels.sum().item()
                    neg_count += (labels == 0).sum().item()
                break
            pw = torch.tensor([neg_count / (pos_count + 1e-6)], device=device)
            print(f"Estimated pos_weight: {pw.item():.2f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    elif task_type == "multilabel_classification":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

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
            direction=args.direction,    # ✅
        ).to(device)
        print("Using PRIME_CrossAttention readout")
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
            direction=args.direction,    # ✅
        ).to(device)
        print(f"Using PRIME with fixed readout: {args.readout_level}")

    # --------------------------------------------------
    # Resume from checkpoint
    # --------------------------------------------------
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise ValueError(f"Checkpoint not found: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from checkpoint: {args.resume}")

    # --------------------------------------------------
    # Output paths
    # --------------------------------------------------
    level_tag     = "_".join(args.active_levels)
    model_tag     = "prime_ca" if args.cross_attention else "prime"
    seed_tag      = f"seed{args.seed}"
    direction_tag = f"_{args.direction}" if args.direction != "bidirectional" else ""

    if args.task == "GeneOntology":
        output_path = f"./ckpts/best_{model_tag}_{args.task}_{args.go_branch}_{level_tag}{direction_tag}_{seed_tag}.pt"
        log_path    = f"./logs/training_log_{model_tag}_{args.task}_{args.go_branch}_{level_tag}{direction_tag}_{seed_tag}.txt"
    else:
        output_path = f"./ckpts/best_{model_tag}_{args.task}_{level_tag}{direction_tag}_{seed_tag}_esm_split.pt"
        log_path    = f"./logs/training_log_{model_tag}_{args.task}_{level_tag}{direction_tag}_{seed_tag}_esm_split.txt"

    # --------------------------------------------------
    # Print summary
    # --------------------------------------------------
    print("=" * 40)
    print(f"Task:          {args.task}")
    print(f"GO Branch:     {args.go_branch}")
    print(f"Num classes:   {num_classes}")
    print(f"Task type:     {task_type}")
    print(f"Task level:    {task_level}")
    print(f"Readout level: {args.readout_level}")
    print(f"Cross attn:    {args.cross_attention}")
    print(f"Direction:     {args.direction}")    # ✅
    print(f"Resume:        {args.resume}")
    print(f"Seed:          {args.seed}")
    print("=" * 40)

    # --------------------------------------------------
    # Train
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
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        device=device,
        task_level=task_level,
    )