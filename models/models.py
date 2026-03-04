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
        hidden_dim = 512,
        encoder_layers=3,
        head_hidden_dim=512,
        head_layers=3,
        dropout=0.1
    ):
        super().__init__()

        # ------------------------------
        # Hierarchical Encoder
        # ------------------------------
        self.encoder = HierarchicalGNN(
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            n_layers=encoder_layers
        )

        # ------------------------------
        # Classification Head
        # ------------------------------
        self.head = MLP_Head(
            in_dim=hidden_dim,
            out_dim=num_classes,
            hidden_dim=head_hidden_dim,
            num_layers=head_layers,
            dropout=dropout
        )

    def forward(self, graph):
        protein_embedding = self.encoder(graph)["protein"]
        logits = self.head(protein_embedding)
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
        
        # ------------------------------
        # Classification Head
        # ------------------------------

        self.head = MLP_Head(
            in_dim=embedding_dim,
            out_dim=num_classes,
            hidden_dim=head_hidden_dim,
            num_layers=head_layers,
            dropout=dropout
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
            embedding = embeddings[1:-1].mean(dim=0) # Average pooling over sequence dimension, excluding CLS and EOS tokens
            embeddings_list.append(embedding)

        protein_embedding = torch.stack(embeddings_list, dim=0)

        logits = self.head(protein_embedding)
        return logits