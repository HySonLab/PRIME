from torch import nn
import torch
import os
import sys

import argparse
import yaml

import scipy.sparse as sp
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class GraphBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.activation = nn.GELU()

    def forward(self, X, A):
        # Pre-norm
        H = self.norm(X)

        # Aggregate neighbors
        # AX = torch.sparse.mm(A, H)
        AX = torch.sparse.mm(A.cpu(), H.cpu()).to(H.device)

        # Transform
        H = self.linear(AX)

        # Residual
        return X + self.activation(H)

class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim, dim)
        self.update = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, z, context):
        g = torch.sigmoid(self.gate(context))
        u = self.update(context)
        return self.norm(z + g * u)
    
class HierarchicalLayer(nn.Module):
    def __init__(self, dims, active_levels):
        super().__init__()

        self.active_levels = active_levels

        # -----------------------------
        # Intra-level GNN
        # -----------------------------
        self.gnns = nn.ModuleDict()

        for level in active_levels:
            self.gnns[level] = GraphBlock(dims[level])

        # -----------------------------
        # Cross-level projections
        # -----------------------------
        self.proj = nn.ModuleDict()
        self.fuse = nn.ModuleDict()

        if "surface" in active_levels and "atom" in active_levels:
            self.proj["surface_atom"] = nn.Linear(dims["surface"], dims["atom"])
            self.fuse["atom"] = GatedFusion(dims["atom"])

        if "atom" in active_levels and "residue" in active_levels:
            self.proj["atom_residue"] = nn.Linear(dims["atom"], dims["residue"])
            self.fuse["residue"] = GatedFusion(dims["residue"])

        if "residue" in active_levels and "sse" in active_levels:
            self.proj["residue_sse"] = nn.Linear(dims["residue"], dims["sse"])
            self.fuse["sse"] = GatedFusion(dims["sse"])

        if "sse" in active_levels and "protein" in active_levels:
            self.proj["sse_protein"] = nn.Linear(dims["sse"], dims["protein"])
            self.fuse["protein"] = GatedFusion(dims["protein"])

        if "surface" in active_levels:
            self.fuse["surface"] = GatedFusion(dims["surface"])

    def forward(self, graph, H):

        # ==================================================
        # Bottom-Up
        # ==================================================

        if "surface" in H and "atom" in H:
            Pi_sa_T = graph.partitions["surface_to_atom_T"]
            atom_up = torch.sparse.mm(Pi_sa_T.cpu(), H["surface"].cpu()).to(H["surface"].device)
            atom_up = self.proj["surface_atom"](atom_up)
            H["atom"] = self.fuse["atom"](H["atom"], atom_up)

        if "atom" in H and "residue" in H:
            Pi_ar_T = graph.partitions["atom_to_residue_T"]
            residue_up = torch.sparse.mm(Pi_ar_T.cpu(), H["atom"].cpu()).to(H["atom"].device)
            residue_up = self.proj["atom_residue"](residue_up)
            H["residue"] = self.fuse["residue"](H["residue"], residue_up)

        if "residue" in H and "sse" in H:
            Pi_rs_T = graph.partitions["residue_to_sse_T"]
            sse_up = torch.sparse.mm(Pi_rs_T.cpu(), H["residue"].cpu()).to(H["residue"].device)
            sse_up = self.proj["residue_sse"](sse_up)
            H["sse"] = self.fuse["sse"](H["sse"], sse_up)

        if "sse" in H and "protein" in H:
            Pi_sp_T = graph.partitions["sse_to_protein_T"]
            protein_up = torch.sparse.mm(Pi_sp_T.cpu(), H["sse"].cpu()).to(H["sse"].device)
            protein_up = self.proj["sse_protein"](protein_up)
            H["protein"] = self.fuse["protein"](H["protein"], protein_up)

        # ==================================================
        # Intra-Level Message Passing
        # ==================================================

        for level in self.active_levels:
            A = getattr(graph, level).A
            H[level] = self.gnns[level](H[level], A)

        # ==================================================
        # Top-Down
        # ==================================================

        if "protein" in H and "sse" in H:
            Pi_sp = graph.partitions["sse_to_protein"]
            sse_down = torch.sparse.mm(Pi_sp.cpu(), H["protein"].cpu()).to(H["protein"].device)
            H["sse"] = self.fuse["sse"](H["sse"], sse_down)

        if "sse" in H and "residue" in H:
            Pi_rs = graph.partitions["residue_to_sse"]
            residue_down = torch.sparse.mm(Pi_rs.cpu(), H["sse"].cpu()).to(H["sse"].device)
            H["residue"] = self.fuse["residue"](H["residue"], residue_down)

        if "residue" in H and "atom" in H:
            Pi_ar = graph.partitions["atom_to_residue"]
            atom_down = torch.sparse.mm(Pi_ar.cpu(), H["residue"].cpu()).to(H["residue"].device)
            H["atom"] = self.fuse["atom"](H["atom"], atom_down)

        if "atom" in H and "surface" in H:
            Pi_sa = graph.partitions["surface_to_atom"]
            surface_down = torch.sparse.mm(Pi_sa.cpu(), H["atom"].cpu()).to(H["atom"].device)
            H["surface"] = self.fuse["surface"](H["surface"], surface_down)

        return H

class HierarchicalGNN(nn.Module):
    def __init__(
        self,
        input_dims,
        active_levels,
        hidden_dim=128,
        n_layers=3,
        dropout=0.1
    ):
        super().__init__()

        self.active_levels = active_levels

        # -----------------------------
        # Input Projection
        # -----------------------------
        self.input_proj = nn.ModuleDict({
            level: nn.Linear(input_dims[level], hidden_dim)
            for level in active_levels
        })

        self.input_norm = nn.ModuleDict({
            level: nn.LayerNorm(hidden_dim)
            for level in active_levels
        })

        self.dropout = nn.Dropout(dropout)

        # -----------------------------
        # Hierarchical Layers
        # -----------------------------
        dims = {level: hidden_dim for level in active_levels}

        self.layers = nn.ModuleList([
            HierarchicalLayer(dims, active_levels)
            for _ in range(n_layers)
        ])

        # -----------------------------
        # Final Norm
        # -----------------------------
        self.final_norm = nn.ModuleDict({
            level: nn.LayerNorm(hidden_dim)
            for level in active_levels
        })

    def forward(self, graph):

        H = {}

        # Input projection
        for level in self.active_levels:
            x = getattr(graph, level).X
            x = self.input_proj[level](x)
            x = torch.relu(x)
            x = self.input_norm[level](x)
            x = self.dropout(x)
            H[level] = x

        # Hierarchical layers
        for layer in self.layers:
            H = layer(graph, H)

        # Final normalization
        for level in H:
            H[level] = self.final_norm[level](H[level])

        return H
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hierarchical Protein GNN"
    )

    parser.add_argument("--config", type=str, required=True)
    
    args = parser.parse_args()
    
    # --------------------------------------------------
    # Load config
    # --------------------------------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    hier_cfg = config["hierarchical"]
    input_dims = hier_cfg["input_dims"]

    model = HierarchicalGNN(
        input_dims=input_dims,
        hidden_dim=hier_cfg["hidden_dim"],
        n_layers=hier_cfg["n_layers"]
    )