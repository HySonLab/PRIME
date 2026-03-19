from torch import nn
import torch
from .egnn import EGNN

class EGNN_AutoEncoder(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, latent_dim,
                 in_edge_nf=0, n_layers=4):
        super().__init__()

        # ----------------------------
        # Encoder
        # ----------------------------
        self.encoder = EGNN(
            in_node_nf=in_node_nf,
            hidden_nf=hidden_nf,
            out_node_nf=latent_dim,
            in_edge_nf=in_edge_nf,
            n_layers=n_layers
        )

        # ----------------------------
        # Coordinate Decoder
        # ----------------------------
        self.coord_decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, 3)
        )

    def forward(self, h, x_noisy, edge_index, edge_attr=None):
        """
        Input:
            h         : node features
            x_noisy   : noisy coordinates
        Output:
            x_recon   : reconstructed coordinates
        """

        # Encode (EGNN returns updated node features and coords)
        z, _ = self.encoder(h, x_noisy, edge_index, edge_attr)

        # Decode coordinates
        x_recon = self.coord_decoder(z)

        return x_recon