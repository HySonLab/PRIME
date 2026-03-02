from torch import nn
import torch
from .egnn import EGNN

class EdgeDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, 1)
        )

    def forward(self, z_i, z_j):
        diff = torch.abs(z_i - z_j)
        x = torch.cat([z_i, z_j, diff], dim=-1)
        return self.mlp(x).squeeze(-1)

class EGNN_AutoEncoder(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, latent_dim,
                 in_edge_nf=0, n_layers=4):
        super().__init__()

        # Encoder
        self.encoder = EGNN(
            in_node_nf=in_node_nf,
            hidden_nf=hidden_nf,
            out_node_nf=latent_dim,
            in_edge_nf=in_edge_nf,
            n_layers=n_layers
        )

        # Decoder
        self.decoder = EdgeDecoder(latent_dim)

    def forward(self, h, x, edge_index, edge_attr,
                pos_edge_pairs, neg_edge_pairs):

        # Encode
        z, _ = self.encoder(h, x, edge_index, edge_attr)

        # Positive edges
        zi_pos = z[pos_edge_pairs[0]]
        zj_pos = z[pos_edge_pairs[1]]
        pos_logits = self.decoder(zi_pos, zj_pos)

        # Negative edges
        zi_neg = z[neg_edge_pairs[0]]
        zj_neg = z[neg_edge_pairs[1]]
        neg_logits = self.decoder(zi_neg, zj_neg)

        return pos_logits, neg_logits