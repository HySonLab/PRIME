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

from utils.utils import add_noise, normalize_coords

class AtomicDataset(Dataset):
    def __init__(self, pdb_dir, k):

        all_files = sorted(glob(os.path.join(pdb_dir, "*.pdb")))
        self.pdb_files = all_files
        self.k = k
        print(f"Found {len(self.pdb_files)} PDB files")

    def __len__(self):
        return len(self.pdb_files)

    def __getitem__(self, idx):

        pdb_path = self.pdb_files[idx]

        h, x, edge_index, edge_attr = build_atomic_graph_from_pdb(
            pdb_path,
            k=self.k,
            device="cpu"
        )

        return h, x, edge_index, edge_attr
    
def atom_type_encoding(atom_name):
    atom_types = ['C', 'N', 'O', 'S', 'P', 'H']
    
    encoding = [0] * len(atom_types)
    
    first_char = atom_name[0]
    if first_char in atom_types:
        encoding[atom_types.index(first_char)] = 1
    
    return encoding

def collate_graphs(batch):

    h_list = []
    x_list = []
    edge_index_list = []
    edge_attr_list = []

    node_offset = 0

    for h, x, edge_index, edge_attr in batch:

        h_list.append(h)
        x_list.append(x)

        # Shift edge indices
        edge_index = edge_index + node_offset
        edge_index_list.append(edge_index)

        edge_attr_list.append(edge_attr)

        node_offset += h.size(0)

    h = torch.cat(h_list, dim=0)
    x = torch.cat(x_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1)
    edge_attr = torch.cat(edge_attr_list, dim=0)

    return h, x, edge_index, edge_attr

def build_atomic_graph_from_pdb(
    pdb_path,
    k=8,
    device="cpu",
    max_atoms=2048
):

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    coords = []
    features = []

    for atom in structure.get_atoms():

        if atom.element == 'H':
            continue

        coords.append(atom.coord)
        features.append(atom_type_encoding(atom.get_name()))

    coords = np.array(coords)
    features = np.array(features)

    # --------------------------------------------------------
    # Subsample atoms if too large
    # --------------------------------------------------------
    if len(coords) > max_atoms:
        idx = np.random.choice(len(coords), max_atoms, replace=False)
        coords = coords[idx]
        features = features[idx]

    # --------------------------------------------------------
    # Convert to torch
    # --------------------------------------------------------
    coords = torch.from_numpy(coords).float()
    coords = normalize_coords(coords)
    
    h = torch.from_numpy(features).float()

    # --------------------------------------------------------
    # KNN Graph
    # --------------------------------------------------------
    edge_index = knn_graph(coords, k=k, loop=False)

    row, col = edge_index
    dist = torch.norm(coords[row] - coords[col], dim=1, keepdim=True)

    edge_attr = dist

    return h, coords, edge_index, edge_attr

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
        x_noisy, x_target = add_noise(x)

        # ----------------------------
        # Forward
        # ----------------------------
        x_recon = model(
            h,
            x_noisy,
            edge_index,
            edge_attr
        )

        # ----------------------------
        # Loss (reconstruct normalized coords)
        # ----------------------------
        loss = F.mse_loss(x_recon, x_target)

        optimizer.zero_grad()
        loss.backward()

        # Optional: stabilize training
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
            # Add noise (same as training)
            # ----------------------------
            x_noisy, x_target, noise_std = add_noise(x)

            # ----------------------------
            # Forward
            # ----------------------------
            x_recon = model(
                h,
                x_noisy,
                edge_index,
                edge_attr
            )

            # ----------------------------
            # Loss
            # ----------------------------
            loss = F.mse_loss(x_recon, x_target)

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

        log_str = (
            f"Epoch {epoch:03d}\n"
            f"Train Loss: {train_loss:.6f}\n"
            f"Val   Loss: {val_loss:.6f}\n"
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