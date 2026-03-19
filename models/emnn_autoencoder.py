from torch import nn
import torch
from .emnn import EMNN 

class EMNN_AutoEncoder(nn.Module):
    """
    Denoising autoencoder using EMNN encoder.
    Suitable for mesh/surface graphs with face_index.
    """

    def __init__(self,
                 in_node_nf,
                 hidden_nf,
                 latent_dim,
                 in_edge_nf=0,
                 n_layers=4):

        super().__init__()

        # ----------------------------
        # Encoder (EMNN)
        # ----------------------------
        self.encoder = EMNN(
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

    # =====================================================
    # Forward
    # =====================================================
    def forward(self,
                h,
                x_noisy,
                edge_index,
                face_index,
                edge_attr=None):

        """
        Input:
            h         : node features
            x_noisy   : noisy coordinates
        Output:
            x_recon   : reconstructed coordinates
        """

        # ----------------------------
        # Encode with EMNN
        # ----------------------------
        z, _ = self.encoder(
            h,
            x_noisy,
            edge_index,
            face_index,
            edge_attr
        )

        # ----------------------------
        # Decode coordinates
        # ----------------------------
        x_recon = self.coord_decoder(z)

        return x_recon