# model.py
import torch
from torch import nn
from .hierarchical_gnn import HierarchicalGNN

from esm.sdk.api import ESMProtein, LogitsConfig
from esm.models.esmc import ESMC

import torch
from torch import nn
from .hierarchical_gnn import HierarchicalGNN
from .classification_head import MLP_Head

class PRIME(nn.Module):
    def __init__(
        self,
        num_classes,
        input_dims,
        active_levels,
        readout_level,
        hidden_dim=512,
        encoder_layers=3,
        head_hidden_dim=512,
        head_layers=3,
        dropout=0.3
    ):
        super().__init__()

        self.readout_level = readout_level

        # -----------------------------------------
        # Encoder
        # -----------------------------------------
        self.encoder = HierarchicalGNN(
            input_dims=input_dims,
            active_levels=active_levels,
            hidden_dim=hidden_dim,
            n_layers=encoder_layers
        )

        # -----------------------------------------
        # Prediction head
        # -----------------------------------------
        self.head = MLP_Head(
            in_dim=hidden_dim,
            out_dim=num_classes,
            hidden_dims=[head_hidden_dim] * head_layers,
            activations=["gelu"] * head_layers + ["identity"],
            dropout=dropout,
            skip=True
        )

    def forward(self, graph):

        # Encode hierarchy
        H = self.encoder(graph)

        # Select representation for prediction
        embedding = H[self.readout_level]

        # If graph has multiple nodes (atom/residue etc.)
        # we pool them to a graph representation
        if embedding.dim() == 2:
            embedding = embedding.mean(dim=0, keepdim=True)

        logits = self.head(embedding)

        return logits

class ESMC_Baseline(nn.Module):

    def __init__(
        self,
        esm_client,
        embedding_dim,
        num_classes,
        head_hidden_dim=512,
        head_layers=3,
        dropout=0.3,
        freeze_encoder=True
    ):
        super().__init__()

        self.client = esm_client
        self.freeze_encoder = freeze_encoder

        self.head = MLP_Head(
            in_dim=embedding_dim,
            out_dim=num_classes,
            hidden_dims=[head_hidden_dim] * head_layers,
            activations=["gelu"] * head_layers + ["identity"],
            dropout=dropout,
            skip=True
        )

    def forward(self, protein_seqs):

        embeddings_list = []

        for seq in protein_seqs:

            protein = ESMProtein(sequence=seq)

            if self.freeze_encoder:
                with torch.no_grad():
                    protein_tensor = self.client.encode(protein)
            else:
                protein_tensor = self.client.encode(protein)

            logits_output = self.client.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True)
            )

            embeddings = logits_output.embeddings.squeeze()
            residue_embeddings = embeddings[1:-1]
            embedding = residue_embeddings.mean(dim=0)
            embeddings_list.append(embedding)

        protein_embedding = torch.stack(embeddings_list, dim=0)

        logits = self.head(protein_embedding)
        return logits