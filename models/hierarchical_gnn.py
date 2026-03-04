from torch import nn
import torch
import os
import sys

import argparse
import yaml

import scipy.sparse as sp
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.hierarchical_graph import HierarchicalProteinGraph, build_hierarchical_protein_graph

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
    def __init__(self, dims):
        super().__init__()

        # Intra-level message passing
        self.surface_gnn = GraphBlock(dims["surface"])
        self.atom_gnn = GraphBlock(dims["atom"])
        self.residue_gnn = GraphBlock(dims["residue"])
        self.sse_gnn = GraphBlock(dims["sse"])
        self.protein_gnn = GraphBlock(dims["protein"])

        # Cross-level projections
        self.surface_to_atom = nn.Linear(dims["surface"], dims["atom"])
        self.atom_to_residue = nn.Linear(dims["atom"], dims["residue"])
        self.residue_to_sse = nn.Linear(dims["residue"], dims["sse"])
        self.sse_to_protein = nn.Linear(dims["sse"], dims["protein"])

        # Gated fusion
        self.atom_fuse = GatedFusion(dims["atom"])
        self.residue_fuse = GatedFusion(dims["residue"])
        self.sse_fuse = GatedFusion(dims["sse"])
        self.surface_fuse = GatedFusion(dims["surface"])
        self.protein_fuse = GatedFusion(dims["protein"])

    def forward(self, graph, H):

        H_surface = H["surface"]
        H_atom = H["atom"]
        H_residue = H["residue"]
        H_sse = H["sse"]
        H_protein = H["protein"]

        # ==================================================
        # Bottom-Up
        # ==================================================

        Pi_sa_T = graph.partitions["surface_to_atom_T"]
        # atom_up = torch.sparse.mm(Pi_sa_T, H_surface)
        atom_up = torch.sparse.mm(Pi_sa_T.cpu(), H_surface.cpu()).to(H_surface.device)
        
        atom_up = self.surface_to_atom(atom_up)
        H_atom = self.atom_fuse(H_atom, atom_up)

        Pi_ar_T = graph.partitions["atom_to_residue_T"]
        # residue_up = torch.sparse.mm(Pi_ar_T, H_atom)
        residue_up = torch.sparse.mm(Pi_ar_T.cpu(), H_atom.cpu()).to(H_atom.device)
        
        residue_up = self.atom_to_residue(residue_up)
        H_residue = self.residue_fuse(H_residue, residue_up)

        Pi_rs_T = graph.partitions["residue_to_sse_T"]
        # sse_up = torch.sparse.mm(Pi_rs_T, H_residue)
        sse_up = torch.sparse.mm(Pi_rs_T.cpu(), H_residue.cpu()).to(H_residue.device)
        
        sse_up = self.residue_to_sse(sse_up)
        H_sse = self.sse_fuse(H_sse, sse_up)

        Pi_sp_T = graph.partitions["sse_to_protein_T"]
        # protein_up = torch.sparse.mm(Pi_sp_T, H_sse)
        protein_up = torch.sparse.mm(Pi_sp_T.cpu(), H_sse.cpu()).to(H_sse.device)
        
        protein_up = self.sse_to_protein(protein_up)
        H_protein = self.protein_fuse(H_protein, protein_up)

        # ==================================================
        # Intra-Level Message Passing
        # ==================================================

        H_surface = self.surface_gnn(H_surface, graph.surface.A)
        H_atom = self.atom_gnn(H_atom, graph.atom.A)
        H_residue = self.residue_gnn(H_residue, graph.residue.A)
        H_sse = self.sse_gnn(H_sse, graph.sse.A)
        H_protein = self.protein_gnn(H_protein, graph.protein.A)

        # ==================================================
        # Top-Down
        # ==================================================

        Pi_sp = graph.partitions["sse_to_protein"]
        # sse_down = torch.sparse.mm(Pi_sp, H_protein)
        sse_down = torch.sparse.mm(Pi_sp.cpu(), H_protein.cpu()).to(H_protein.device)
        H_sse = self.sse_fuse(H_sse, sse_down)

        Pi_rs = graph.partitions["residue_to_sse"]
        # residue_down = torch.sparse.mm(Pi_rs, H_sse)
        residue_down = torch.sparse.mm(Pi_rs.cpu(), H_sse.cpu()).to(H_sse.device)
        H_residue = self.residue_fuse(H_residue, residue_down)

        Pi_ar = graph.partitions["atom_to_residue"]
        atom_down = torch.sparse.mm(Pi_ar, H_residue)
        H_atom = self.atom_fuse(H_atom, atom_down)

        Pi_sa = graph.partitions["surface_to_atom"]
        # surface_down = torch.sparse.mm(Pi_sa, H_atom)
        surface_down = torch.sparse.mm(Pi_sa.cpu(), H_atom.cpu()).to(H_atom.device)
        H_surface = self.surface_fuse(H_surface, surface_down)

        return {
            "surface": H_surface,
            "atom": H_atom,
            "residue": H_residue,
            "sse": H_sse,
            "protein": H_protein,
        }

class HierarchicalGNN(nn.Module):
    def __init__(self, input_dims, hidden_dim=128, n_layers=3, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim

        # --------------------------------------------------
        # Input Projection
        # --------------------------------------------------
        self.input_proj = nn.ModuleDict({
            level: nn.Linear(input_dims[level], hidden_dim)
            for level in input_dims
        })

        self.input_norm = nn.ModuleDict({
            level: nn.LayerNorm(hidden_dim)
            for level in input_dims
        })

        self.dropout = nn.Dropout(dropout)

        # --------------------------------------------------
        # Hierarchical Layers
        # --------------------------------------------------
        dims = {level: hidden_dim for level in input_dims}

        self.layers = nn.ModuleList([
            HierarchicalLayer(dims)
            for _ in range(n_layers)
        ])

        # Final normalization
        self.final_norm = nn.ModuleDict({
            level: nn.LayerNorm(hidden_dim)
            for level in input_dims
        })

    def forward(self, graph):

        # --------------------------------------------------
        # Input Projection
        # --------------------------------------------------
        H = {}

        for level in self.input_proj:
            x = self.input_proj[level](getattr(graph, level).X)
            x = torch.relu(x)
            x = self.input_norm[level](x)
            x = self.dropout(x)
            H[level] = x

        # --------------------------------------------------
        # Hierarchical Message Passing
        # --------------------------------------------------
        for layer in self.layers:
            H = layer(graph, H)

        # --------------------------------------------------
        # Final Normalization
        # --------------------------------------------------
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