import torch
import torch.nn.functional as F
from torch.optim import Adam
import argparse
import numpy as np
from Bio.PDB import PDBParser
from torch_cluster import knn_graph
from models.egnn_autoencoder import EGNN_AutoEncoder
from glob import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from torch.utils.data import random_split
import yaml
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from utils.helpers import add_noise, normalize_coords
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.dataset import AtomicDataset, collate_graphs

# ============================================================
# Train
# ============================================================

def train_epoch(model, loader, optimizer, device):

    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc="Train", leave=False)

    for h, x, edge_index, edge_attr in pbar:

        h = h.to(device)
        x = x.to(device)
        edge_index = edge_index.to(device)

        if edge_attr is not None:
            edge_attr = edge_attr.to(device)

        # ----------------------------
        # Add noise
        # ----------------------------
        x_noisy, x_clean, noise_std = add_noise(x)
        noise_std = float(noise_std)
        
        target = x_clean - x_noisy

        # ----------------------------
        # Forward
        # ----------------------------
        pred_noise = model(
            h, x_noisy, edge_index, edge_attr
        )

        # ----------------------------
        # Denoising MSE loss
        # ----------------------------
        loss = ((pred_noise - target) ** 2).mean(dim=-1)
        loss = loss.mean()

        # ----------------------------
        # Smoothness regularization
        # ----------------------------
        row, col = edge_index
        smooth_loss = ((pred_noise[row] - pred_noise[col]) ** 2).mean()

        loss = loss + 0.01 * smooth_loss

        # ----------------------------
        # Backprop
        # ----------------------------
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# ============================================================
# Evaluate
# ============================================================

def evaluate_epoch(model, loader, device):

    model.eval()
    total_loss = 0

    with torch.no_grad():

        pbar = tqdm(loader, desc="Val", leave=False)

        for h, x, edge_index, edge_attr in pbar:

            h = h.to(device)
            x = x.to(device)
            edge_index = edge_index.to(device)

            if edge_attr is not None:
                edge_attr = edge_attr.to(device)

            # ----------------------------
            # Add noise
            # ----------------------------
            x_noisy, x_clean, noise_std = add_noise(x)
            noise_std = float(noise_std)

            # Target = noise (residual)
            target = x_clean - x_noisy

            # ----------------------------
            # Forward
            # ----------------------------
            pred_noise = model(
                h,
                x_noisy,
                edge_index,
                edge_attr
            )

            # ----------------------------
            # Denoising MSE loss
            # ----------------------------
            loss = ((pred_noise - target) ** 2).mean(dim=-1)
            loss = loss.mean()

            total_loss += loss.item()

    return total_loss / len(loader)

def train(config):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    atom_cfg = config["atom"]
    data_cfg = config["data"]

    # --------------------------------------------------------
    # Dataset + Split
    # --------------------------------------------------------
    dataset = AtomicDataset(
        pdb_dir=data_cfg["pdb_dir"],
        k=atom_cfg["k"]
    )

    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_graphs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_graphs
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = EGNN_AutoEncoder(
        in_node_nf=atom_cfg["atom_feat_dim"],
        hidden_nf=atom_cfg["hidden_dim"],
        latent_dim=atom_cfg["latent_dim"],
        in_edge_nf=1,
        n_layers=atom_cfg["n_layers"]
    ).to(device)

    optimizer = Adam(model.parameters(), lr=atom_cfg["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=atom_cfg["epochs"], eta_min=1e-5)

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------
    log_path = atom_cfg["log_path"]
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "w") as f:
        f.write("========================================\n")
        f.write("Atom Denoising Autoencoder Training Log\n")
        f.write("========================================\n")

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------
    best_val_loss = float("inf")

    for epoch in range(atom_cfg["epochs"]):

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate_epoch(model, val_loader, device)
        scheduler.step()

        log_str = (
            f"Epoch {epoch:03d}\n"
            f"Train Loss: {train_loss:.6f}\n"
            f"Val   Loss: {val_loss:.6f}\n"
            f"LR:         {scheduler.get_last_lr()[0]:.2e}\n"
            f"----------------------------------------\n"
        )

        print(log_str)

        with open(log_path, "a") as f:
            f.write(log_str)

        # --------------------------------------------------------
        # Save best model (based on loss)
        # --------------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save({
                "model_state_dict": model.encoder.state_dict(),
                "config": config
            }, atom_cfg["save_path"])

            with open(log_path, "a") as f:
                f.write(f"New best model saved (Val Loss={best_val_loss:.6f})\n")

    print(f"Encoder saved to {atom_cfg['save_path']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/home/dvnguye2/PRL/config/model_config.yaml")
    args = parser.parse_args()

    # --------------------------------------------
    # Load YAML config
    # --------------------------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(config)