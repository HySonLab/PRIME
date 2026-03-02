import torch
import torch.nn.functional as F
from torch.optim import Adam
import argparse
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from models.emnn_autoencoder import EMNN_AutoEncoder
from partition import extract_partition_matrices, mesh_simplification_quadric_decimation
from utils.hierarchical_graph import build_surface_graph
from glob import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import yaml

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class SurfaceDataset(Dataset):

    def __init__(self, surface_dir, max_faces=None):

        all_files = sorted(
            glob(os.path.join(surface_dir, "*.ply")) +
            glob(os.path.join(surface_dir, "*.obj")) +
            glob(os.path.join(surface_dir, "*.off"))
        )

        self.surface_files = all_files[:100]
        
        self.max_faces = max_faces

        print(f"Using {len(self.surface_files)} surface meshes")

    def __len__(self):
        return len(self.surface_files)

    def __getitem__(self, idx):

        mesh = trimesh.load(self.surface_files[idx], process=False)

        if self.max_faces is not None:
            if mesh.faces.shape[0] > self.max_faces:
                mesh = mesh_simplification_quadric_decimation(
                    mesh,
                    target_faces=self.max_faces
                )

        # Build graph on CPU
        x, edge_index, face_index, edge_attr = build_surface_graph(
            mesh,
            device="cpu"
        )

        n_nodes = x.size(0)

        h = torch.ones(n_nodes, 16)  # or pass input_dim

        return h, x, edge_index, face_index, edge_attr

def collate_surface_graphs(batch):

    h_list = []
    x_list = []
    edge_index_list = []
    face_index_list = []
    edge_attr_list = []

    node_offset = 0

    for h, x, edge_index, face_index, edge_attr in batch:

        h_list.append(h)
        x_list.append(x)

        edge_index = edge_index + node_offset
        edge_index_list.append(edge_index)

        face_index = face_index + node_offset
        face_index_list.append(face_index)

        edge_attr_list.append(edge_attr)

        node_offset += h.size(0)

    h = torch.cat(h_list, dim=0)
    x = torch.cat(x_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1)
    face_index = torch.cat(face_index_list, dim=1)
    edge_attr = torch.cat(edge_attr_list, dim=0)

    return h, x, edge_index, face_index, edge_attr

# ============================================================
# Negative Sampling
# ============================================================

def sample_negative_edges(n_nodes, edge_index, num_samples):
    existing = set(zip(edge_index[0].tolist(),
                       edge_index[1].tolist()))

    neg_i = []
    neg_j = []

    while len(neg_i) < num_samples:
        i = np.random.randint(0, n_nodes)
        j = np.random.randint(0, n_nodes)

        if i != j and (i, j) not in existing:
            neg_i.append(i)
            neg_j.append(j)

    return (
        torch.LongTensor(neg_i),
        torch.LongTensor(neg_j)
    )

# ============================================================
# Loss
# ============================================================

def edge_loss(pos_logits, neg_logits):
    pos_labels = torch.ones_like(pos_logits)
    neg_labels = torch.zeros_like(neg_logits)

    loss_pos = F.binary_cross_entropy_with_logits(pos_logits, pos_labels)
    loss_neg = F.binary_cross_entropy_with_logits(neg_logits, neg_labels)

    return loss_pos + loss_neg

# ============================================================
# Training
# ============================================================

def train(config):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    surface_cfg = config["surface"]
    data_cfg = config["data"]

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    dataset = SurfaceDataset(
        surface_dir=data_cfg["surface_dir"],
        max_faces=surface_cfg["max_faces"],
    )

    loader = DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_surface_graphs
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = EMNN_AutoEncoder(
        in_node_nf=surface_cfg["input_dim"],
        hidden_nf=surface_cfg["hidden_dim"],
        latent_dim=surface_cfg["latent_dim"],
        in_edge_nf=1,
        n_layers=surface_cfg["n_layers"]
    ).to(device)

    optimizer = Adam(model.parameters(), lr=surface_cfg["lr"])

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------
    for epoch in range(surface_cfg["epochs"]):

        model.train()
        total_loss = 0

        for h, x, edge_index, face_index, edge_attr in loader:

            h = h.to(device)
            x = x.to(device)
            edge_index = edge_index.to(device)
            face_index = face_index.to(device)
            edge_attr = edge_attr.to(device)

            n_nodes = x.size(0)
            pos_edges = edge_index

            neg_edges = sample_negative_edges(
                n_nodes=n_nodes,
                edge_index=edge_index.cpu(),
                num_samples=pos_edges.shape[1]
            )

            neg_edges = (
                neg_edges[0].to(device),
                neg_edges[1].to(device)
            )

            pos_logits, neg_logits = model(
                h,
                x,
                edge_index,
                face_index,
                edge_attr,
                pos_edges,
                neg_edges
            )

            loss = edge_loss(pos_logits, neg_logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch:03d} | Avg Loss {total_loss/len(loader):.4f}")

    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------
    os.makedirs(os.path.dirname(surface_cfg["save_path"]), exist_ok=True)

    torch.save({
        "model_state_dict": model.encoder.state_dict(),
        "config": config
    }, surface_cfg["save_path"])

    print(f"Encoder saved to {surface_cfg['save_path']}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # --------------------------------------------
    # Load YAML config
    # --------------------------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(config)

