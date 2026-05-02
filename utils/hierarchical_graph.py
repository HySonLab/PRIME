from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
import argparse
import trimesh

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import open3d as o3d
import torch
import torch.nn.functional as F
from glob import glob
from tqdm import tqdm
import yaml
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.partition import extract_partition_matrices, mesh_simplification_quadric_decimation
from utils.export_surface import export_surface
from models.egnn import EGNN
from models.emnn import EMNN
from torch_cluster import knn_graph
from pymol import cmd
from torch_geometric.data import Data
from scipy.sparse import diags
from scipy import spatial

from esm import pretrained
# Load ESM-2 model once
esm2_model, alphabet = pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
esm2_model = esm2_model.eval().to('cuda')
batch_converter = alphabet.get_batch_converter()

MAX_FACES = 1024

ATOM37_NAMES = [
    "N", "CA", "C", "O", "CB",
    "CG", "CG1", "CG2", "CD", "CD1", "CD2",
    "CE", "CE1", "CE2", "CE3",
    "CZ", "CZ2", "CZ3",
    "CH2",
    "ND1", "ND2", "NE", "NE1", "NE2",
    "NH1", "NH2",
    "NZ",
    "OD1", "OD2",
    "OE1", "OE2",
    "OG", "OG1",
    "OH",
    "SD",
    "SG",
    "OXT"
]

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

def get_full_atom_coords(coords: torch.Tensor, fill_value: float = 1e-5):
    """
    Flatten per-residue atom coordinates and return a residue index mapping.
    Matches ProteinWorkshop's exact implementation.

    :param coords: Shape (N_residues, 37, 3)
    :return:
        - flat_coords:    (N_atoms, 3)  — valid atom coordinates
        - residue_index:  (N_atoms,)    — which residue each atom belongs to
        - atom_type:      (N_atoms,)    — atom type index [0-36]
    """

    filled        = coords[:, :, 0] != fill_value   # (N_res, 37)
    nz            = filled.nonzero()
    residue_index = nz[:, 0]                        # (N_atoms,)
    atom_type     = nz[:, 1]                        # (N_atoms,)

    flat_coords = coords.reshape(-1, 3)
    flat_coords = flat_coords[coords.reshape(-1, 3)[:, 0] != fill_value]  # (N_atoms, 3)

    return flat_coords, residue_index, atom_type

class BindingSiteTransform:
    def __init__(self, radius: float = 3.5, ca_only: bool = False):
        self.radius     = radius
        self.fill_value = 1e-5
        self.ca_only    = ca_only

    def __call__(self, data):
        chain_strs    = [res.split(":")[0] for res in data.residue_id]
        chain_strs    = list(np.unique(chain_strs))

        target_chains = []
        for chain in data.graph_y:
            if chain in chain_strs:
                target_chains.append(chain_strs.index(chain))

        if len(target_chains) == 0:
            raise ValueError(
                f"None of the target chains {list(data.graph_y)} "
                f"found in residue_id chains {chain_strs}"
            )

        target_chains  = torch.tensor(target_chains)
        target_indices = torch.where(torch.isin(data.chains, target_chains))[0]

        mask = torch.zeros(data.coords.shape[0], dtype=torch.bool)
        mask[target_indices] = True

        target_struct = data.coords[mask]
        other_chains  = data.coords[~mask]

        N_TARGET_RESIDUES = target_struct.shape[0]

        if N_TARGET_RESIDUES == 0:
            raise ValueError("No residues found in target chain after masking.")

        # Flatten and remove fill-value padding rows
        other_chains = other_chains.reshape(-1, 3)
        other_chains = other_chains[
            ~torch.all(other_chains == self.fill_value, dim=1)
        ]

        if other_chains.shape[0] == 0:
            label = torch.zeros(N_TARGET_RESIDUES, dtype=torch.long)
        else:
            if self.ca_only:
                # Index 1 = Cα atom
                kd_tree = spatial.KDTree(target_struct[:, 1, :])
            else:
                # Use all heavy atoms, track atom->residue mapping
                coords, res_idx, _ = get_full_atom_coords(target_struct)
                kd_tree = spatial.KDTree(coords.numpy())

            indices = kd_tree.query_ball_point(other_chains.numpy(), self.radius)
            indices = [item for sublist in indices for item in sublist]
            indices = torch.tensor(indices, dtype=torch.long)

            # Map atom indices back to residue indices
            if not self.ca_only:
                indices = torch.unique(res_idx[indices])

            label          = torch.zeros(N_TARGET_RESIDUES, dtype=torch.long)
            label[indices] = 1

        data.node_y = label

        if hasattr(data, "graph_y"):
            del data.graph_y

        data.coords     = target_struct
        data.residues   = np.array(data.residues)[mask.numpy()]
        data.residue_id = np.array(data.residue_id)[mask.numpy()]
        data.chains     = data.chains[mask]

        if hasattr(data, "residue_type") and data.residue_type is not None:
            data.residue_type = data.residue_type[mask]

        if hasattr(data, "x") and data.x is not None:
            data.x = data.x[mask]

        if hasattr(data, "seq_pos"):
            data.seq_pos = data.seq_pos[mask]

        if hasattr(data, "amino_acid_one_hot"):
            data.amino_acid_one_hot = data.amino_acid_one_hot[mask]

        return data

_binding_site_transform = BindingSiteTransform(radius=3.5, ca_only=False)

def normalize_coords(x, mode="std", eps=1e-6):
    """
    Normalize 3D coordinates.

    Args:
        x: (N, 3) tensor
        mode: "std" or "radius"
        eps: numerical stability

    Returns:
        normalized x
    """

    # Center
    x = x - x.mean(dim=0, keepdim=True)

    if mode == "std":
        std = x.std(dim=0, keepdim=True)
        std[std < eps] = 1.0
        x = x / std

    elif mode == "radius":
        radius = torch.norm(x, dim=1).max()
        x = x / (radius + eps)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return x

def write_atom_to_pdb(data, output_path):
    coords = data.coords.numpy()
    residue_names = data.residues
    chains = data.chains.numpy()
    residue_ids = data.residue_id

    atom_counter = 1
    lines = []

    for res_idx in range(coords.shape[0]):

        residue_name = residue_names[res_idx]
        chain_id = residue_ids[res_idx].split(":")[0]
        res_number = int(residue_ids[res_idx].split(":")[2])

        for atom_idx in range(coords.shape[1]):

            x, y, z = coords[res_idx, atom_idx]

            # Skip padding atoms
            # if np.linalg.norm([x, y, z]) < 1e-4:
            #     continue

            atom_name = ATOM37_NAMES[atom_idx]
            element = atom_name[0]  # crude but works

            line = (
                f"ATOM  {atom_counter:5d} "
                f"{atom_name:<4s}"
                f"{residue_name:>4s} "
                f"{chain_id}"
                f"{res_number:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00 20.00           {element:>2s}"
            )

            lines.append(line)
            atom_counter += 1

    lines.append("END")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

def get_sequence_from_pdb(
    pdb_file: str,
    model_id: int = 0
) -> dict:
    """
    Extract protein sequences from ATOM records of a PDB file.

    Parameters
    ----------
    pdb_file : str
        Path to PDB file
    model_id : int
        Model index (default: 0)

    Returns
    -------
    sequences : dict
        {chain_id: sequence}
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    sequences = {}

    model = structure[model_id]
    for chain in model:
        seq = []
        for residue in chain:
            if residue.id[0] == " ":  # standard amino acid
                try:
                    seq.append(seq1(residue.resname))
                except KeyError:
                    # non-standard residue
                    continue
        sequences[chain.id] = "".join(seq)

    return sequences

def build_surface_graph(mesh, device):
    """
    Build vertex-based surface graph for EMNN.

    Nodes: mesh vertices
    Edges: triangle adjacency
    Faces: triangle indices
    """

    # --------------------------------------------------
    # Vertices
    # --------------------------------------------------
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    x = torch.tensor(vertices, dtype=torch.float32)
    
    x = normalize_coords(x)

    # --------------------------------------------------
    # Build edges from faces (vectorized, NO loop)
    # --------------------------------------------------
    # faces: (F, 3)
    i = faces[:, 0]
    j = faces[:, 1]
    k = faces[:, 2]

    # 6 directed edges per triangle
    edges = np.stack([
        np.concatenate([i, j, k, j, k, i]),
        np.concatenate([j, k, i, i, j, k])
    ], axis=0)

    edge_index = torch.from_numpy(edges).long()

    # Remove duplicates
    edge_index = torch.unique(edge_index, dim=1)

    # --------------------------------------------------
    # Edge features
    # --------------------------------------------------
    row, col = edge_index
    edge_attr = torch.norm(x[row] - x[col], dim=1, keepdim=True)

    # --------------------------------------------------
    # Face index (3, F)
    # --------------------------------------------------
    face_index = torch.from_numpy(faces.T).long()

    # --------------------------------------------------
    # Move to device
    # --------------------------------------------------
    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    face_index = face_index.to(device)

    return x, edge_index, face_index, edge_attr

def coarsen_adjacency(
    A_fine: csr_matrix,
    Pi: csr_matrix,
    normalize: bool = False
) -> csr_matrix:
    A_coarse = Pi.T @ A_fine @ Pi

    if normalize:
        sizes = np.array(Pi.sum(axis=0)).flatten()
        inv = np.reciprocal(sizes, where=sizes > 0)
        D = csr_matrix(np.diag(inv))
        A_coarse = D @ A_coarse @ D

    return A_coarse

def build_surface_knn_adjacency(
    face_centroids: np.ndarray,
    k: int = 8,
) -> csr_matrix:
    """
    Build kNN adjacency for surface faces.
    Guarantees exactly k neighbors per node — memory predictable.
    """
    from torch_cluster import knn_graph

    centroids_t = torch.tensor(face_centroids, dtype=torch.float32)
    edge_index  = knn_graph(centroids_t, k=k, loop=False)

    row  = edge_index[0].numpy()
    col  = edge_index[1].numpy()
    data = np.ones(len(row), dtype=np.float32)

    A = csr_matrix(
        (data, (row, col)),
        shape=(len(face_centroids), len(face_centroids))
    )

    # make symmetric
    A = A + A.T
    A.data[:] = 1.0   # binarize after symmetrization

    return A

AA_LIST = [
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

def amino_acid_one_hot(residues):
    idx = [AA_TO_IDX.get(r, 0) for r in residues]
    return F.one_hot(torch.tensor(idx), num_classes=len(AA_LIST)).float()

def compute_alpha(coords):
    # angle between (i-1, i, i+1)
    v1 = coords[1:-1] - coords[:-2]
    v2 = coords[2:] - coords[1:-1]

    cos_angle = F.cosine_similarity(v1, v2, dim=-1)
    angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))

    # pad to match N
    angle = F.pad(angle, (1, 1))
    return angle.unsqueeze(-1)

def compute_kappa(coords):
    # curvature approximation
    v1 = coords[1:-1] - coords[:-2]
    v2 = coords[2:] - coords[1:-1]

    cross = torch.cross(v1, v2, dim=-1)
    norm_cross = torch.norm(cross, dim=-1)
    norm_v1 = torch.norm(v1, dim=-1) + 1e-6
    norm_v2 = torch.norm(v2, dim=-1) + 1e-6

    kappa = norm_cross / (norm_v1 * norm_v2)

    kappa = F.pad(kappa, (1, 1))
    return kappa.unsqueeze(-1)

def compute_dihedrals(coords):
    # torsion angle for 4 consecutive points
    p0 = coords[:-3]
    p1 = coords[1:-2]
    p2 = coords[2:-1]
    p3 = coords[3:]

    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1 = F.normalize(b1, dim=-1)

    v = b0 - (b0 * b1).sum(-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdim=True) * b1

    x = (v * w).sum(-1)
    y = torch.cross(b1, v, dim=-1).mul(w).sum(-1)

    angle = torch.atan2(y, x)

    angle = F.pad(angle, (1, 2))  # match N
    return angle.unsqueeze(-1)

def compute_scalar_node_features(res_data, node_features):
    """
    Args:
        res_data.coords : (N, 3)
        res_data.residues : list[str]

    Returns:
        Tensor (N, F)
    """

    coords = res_data.coords
    residues = res_data.residues

    feats = []

    for feature in node_features:

        if feature == "amino_acid_one_hot":
            feats.append(amino_acid_one_hot(residues))

        elif feature == "alpha":
            feats.append(compute_alpha(coords))

        elif feature == "kappa":
            feats.append(compute_kappa(coords))

        elif feature == "dihedrals":
            feats.append(compute_dihedrals(coords))

        else:
            raise ValueError(f"Unknown feature: {feature}")

    # ensure consistent shape
    feats = [f if f.ndim == 2 else f.unsqueeze(-1) for f in feats]

    return torch.cat(feats, dim=-1)

def normalize_adjacency(A):
    A = A.astype(np.float32)
    A.eliminate_zeros()

    degrees = np.array(A.sum(axis=1)).flatten()

    d_inv_sqrt = np.zeros_like(degrees)
    mask = degrees > 0
    d_inv_sqrt[mask] = 1.0 / np.sqrt(degrees[mask])

    D_inv_sqrt = diags(d_inv_sqrt)

    return (D_inv_sqrt @ A @ D_inv_sqrt).tocsr()

def get_surface_features(
        mesh,
        encoder=None,
        device="cuda",
        input_dim=16):

    faces    = mesh.faces
    vertices = mesh.vertices

    # --------------------------------------------------
    # Handcrafted features (always compute) — face-based
    # --------------------------------------------------
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    face_areas = 0.5 * np.linalg.norm(
        np.cross(v1 - v0, v2 - v0), axis=1
    )

    centroids      = (v0 + v1 + v2) / 3.0
    protein_center = vertices.mean(axis=0)
    dist_to_center = np.linalg.norm(centroids - protein_center, axis=1)

    handcrafted = np.stack([face_areas, dist_to_center], axis=1)
    handcrafted = (handcrafted - handcrafted.mean(0)) / (handcrafted.std(0) + 1e-6)

    # ==========================================================
    # OPTION 1: Use pretrained encoder
    # ==========================================================
    if encoder is not None:

        x, edge_index, face_index, edge_attr = build_surface_graph(
            mesh, device=device
        )

        vertices_t     = torch.tensor(vertices, dtype=torch.float32).to(device)
        center         = vertices_t.mean(dim=0, keepdim=True)
        dist           = torch.norm(vertices_t - center, dim=1, keepdim=True)
        vertex_normals = torch.tensor(
            mesh.vertex_normals.copy(), dtype=torch.float32
        ).to(device)
        z_coord        = vertices_t[:, 2:3]

        h = torch.cat([vertex_normals, dist, z_coord], dim=1)  # (N_v, 5)
        h = (h - h.mean(0)) / (h.std(0) + 1e-6)
        pad = torch.zeros(h.size(0), input_dim - h.size(1), device=device)
        h   = torch.cat([h, pad], dim=1)                       # (N_v, 16)

        with torch.no_grad():
            z_vertex, _ = encoder(
                h,
                x,
                edge_index,
                face_index,
                edge_attr
            )

        z_vertex = z_vertex.cpu().numpy()                      # (N_v, latent_dim)

        z_face = (
            z_vertex[faces[:, 0]] +
            z_vertex[faces[:, 1]] +
            z_vertex[faces[:, 2]]
        ) / 3.0                                                # (N_f, latent_dim)

        # concat learned + handcrafted
        X_surface = np.concatenate([z_face, handcrafted], axis=1)

        return X_surface.astype(np.float32)

    # ==========================================================
    # OPTION 2: Handcrafted only
    # ==========================================================
    else:
        bias      = np.ones((handcrafted.shape[0], 1))
        X_surface = np.concatenate([handcrafted, bias], axis=1)

        return X_surface.astype(np.float32)

def get_atom_features(
        process_data,
        encoder=None,
        device="cuda",
        k=8):

    coords_np = process_data.coords.numpy()  # (N_res, 37, 3)

    elements    = ["C", "N", "O", "S", "P", "H"]
    elem_to_idx = {e: i for i, e in enumerate(elements)}
    backbone_indices = {0, 1, 2, 3}  # N, CA, C, O in ATOM37

    coords_list   = []
    features_list = []
    backbone_list = []

    for res_idx in range(coords_np.shape[0]):
        for atom_idx in range(coords_np.shape[1]):
            x, y, z = coords_np[res_idx, atom_idx]

            if np.linalg.norm([x, y, z]) < 1e-4:
                continue

            atom_name = ATOM37_NAMES[atom_idx]
            element   = atom_name[0]

            if element == "H":
                continue

            coords_list.append([x, y, z])

            one_hot = np.zeros(len(elements), dtype=np.float32)
            if element in elem_to_idx:
                one_hot[elem_to_idx[element]] = 1.0
            features_list.append(one_hot)

            backbone_list.append(float(atom_idx in backbone_indices))

    if len(coords_list) == 0:
        raise ValueError("No valid atoms found in processed data.")

    coords      = np.array(coords_list,   dtype=np.float32)
    features    = np.array(features_list, dtype=np.float32)
    is_backbone = np.array(backbone_list, dtype=np.float32)[:, None]

    # ==========================================================
    # OPTION 1: Use pretrained encoder
    # ==========================================================
    if encoder is not None:

        coords_t      = torch.from_numpy(coords).float().to(device)
        is_backbone_t = torch.from_numpy(is_backbone).float().to(device)

        h        = torch.from_numpy(features).float().to(device)
        h        = torch.cat([h, is_backbone_t], dim=-1)           # (N, 7)

        h_mean   = coords_t.mean(dim=0, keepdim=True)
        h_std    = coords_t.std(dim=0, keepdim=True).clamp(min=1e-6)
        coords_t = (coords_t - h_mean) / h_std

        edge_index = knn_graph(coords_t, k=k, loop=False)
        row, col   = edge_index
        dist       = torch.norm(coords_t[row] - coords_t[col], dim=1, keepdim=True)

        with torch.no_grad():
            z, _ = encoder(h, coords_t, edge_index, dist)

        return z.cpu().numpy().astype(np.float32)                  # (N, latent_dim)

    # ==========================================================
    # OPTION 2: Handcrafted only
    # ==========================================================
    else:
        return np.concatenate([features, is_backbone], axis=1).astype(np.float32)

def build_residue_data(process_data):
    """
    Build residue data from processed .pt file.
    Uses CA coords (index 1 in ATOM37).
    """
    coords_np = process_data.coords.numpy()  # (N_res, 37, 3)
    residues  = list(process_data.residues)

    ca_coords = torch.tensor(coords_np[:, 1, :], dtype=torch.float32)  # (N_res, 3)

    data          = Data()
    data.coords   = ca_coords
    data.residues = residues
    data.batch    = torch.zeros(ca_coords.size(0), dtype=torch.long)

    return data

def get_residue_features(process_data):
    """
    Returns:
        Tensor (N_res, F)
    """
    res_data = build_residue_data(process_data)

    X_res = compute_scalar_node_features(
        res_data,
        node_features=[
            "amino_acid_one_hot",
            "alpha",
            "kappa",
            "dihedrals"
        ]
    )

    return X_res

def get_sse_features(partition, sse_labels):
    """
    SSE-level features using SSE type encoding only.
    Each SSE node gets a one-hot of its type + length.

    sse_labels: list of int (per residue), 0=loop, 1=helix, 2=strand
    partition:  csr_matrix (N_res, N_sse)
    """
    N_sse = partition.shape[1]
    N_types = 3  # loop, helix, strand

    # One SSE label = the label of its first residue
    # (all residues in an SSE share the same label by construction)
    Pi_dense = partition.toarray()  # (N_res, N_sse)
    sse_labels = np.array(sse_labels)

    one_hot = np.zeros((N_sse, N_types), dtype=np.float32)
    lengths  = np.zeros((N_sse, 1),      dtype=np.float32)

    for i in range(N_sse):
        idx = Pi_dense[:, i] > 0
        if idx.sum() == 0:
            continue
        label = sse_labels[idx][0]   # all same by construction
        one_hot[i, label] = 1.0
        lengths[i, 0] = idx.sum()

    # Normalize length
    lengths = (lengths - lengths.mean()) / (lengths.std() + 1e-6)

    return np.concatenate([one_hot, lengths], axis=1)  # (N_sse, 4)

def get_protein_features(protein_seq: str, device: str = "cuda") -> np.ndarray:
    """
    Protein-level features using ESM-2 CLS token.

    Returns:
        (1, 1280)
    """
    
    data = [("protein", protein_seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    with torch.no_grad():
        results = esm2_model(tokens, repr_layers=[33], return_contacts=False)

    # CLS token is index 0 → single vector for whole protein
    cls_embedding = results["representations"][33][:, 0, :]  # (1, 1280)

    return cls_embedding.cpu().numpy()

@dataclass
class ProteinGraphLevel:
    name: str
    A: csr_matrix              # scipy sparse
    X: np.ndarray              # numpy features

    def to_torch(self, device):
        """
        Convert adjacency and features to torch tensors.
        """

        # Convert adjacency once
        A = self.A.tocoo()
        indices = torch.stack([
            torch.from_numpy(A.row),
            torch.from_numpy(A.col)
        ]).long()

        values = torch.from_numpy(A.data).float()

        self.A = torch.sparse_coo_tensor(
            indices,
            values,
            size=A.shape,
            device=device
        ).coalesce()
        
        if self.X is not None:
            self.X = self.X.to(device)

    @property
    def num_nodes(self):
        return self.A.shape[0]

class HierarchicalProteinGraph:
    """
    Protein-specific hierarchical graph with physics-informed coarsening.
    Graph levels store adjacency + node features (no coordinates).
    """

    def __init__(
        self,
        surface: ProteinGraphLevel,
        atom: ProteinGraphLevel,
        residue: ProteinGraphLevel,
        sse: ProteinGraphLevel,
        protein: ProteinGraphLevel,
        partitions: dict[str, csr_matrix],
    ):
        self.surface = surface
        self.atom = atom
        self.residue = residue
        self.sse = sse
        self.protein = protein
        self.partitions = partitions
        
        self._validate()
    
    def to_torch(self, device):
        """
        Convert entire hierarchical graph to torch tensors.
        """

        # Convert all graph levels
        self.surface.to_torch(device)
        self.atom.to_torch(device)
        self.residue.to_torch(device)
        self.sse.to_torch(device)
        self.protein.to_torch(device)

        # Convert partitions and cache transpose
        new_partitions = {}

        for key, Pi in self.partitions.items():
            Pi = Pi.tocoo()

            indices = torch.stack([
                torch.from_numpy(Pi.row),
                torch.from_numpy(Pi.col)
            ]).long()

            values = torch.from_numpy(Pi.data).float()

            Pi_torch = torch.sparse_coo_tensor(
                indices,
                values,
                size=Pi.shape,
                device=device
            ).coalesce()

            new_partitions[key] = Pi_torch
            new_partitions[key + "_T"] = Pi_torch.transpose(0, 1).coalesce()

        self.partitions = new_partitions

    def _validate(self):
        required = {
            "surface_to_atom",
            "atom_to_residue",
            "residue_to_sse",
        }
        missing = required - self.partitions.keys()
        if missing:
            raise ValueError(f"Missing partitions: {missing}")

        Pi_sa = self.partitions["surface_to_atom"]
        Pi_ar = self.partitions["atom_to_residue"]
        Pi_rs = self.partitions["residue_to_sse"]

        # surface → atom
        assert self.surface.num_nodes == Pi_sa.shape[0]
        assert self.atom.num_nodes == Pi_sa.shape[1]

        # atom → residue
        assert self.atom.num_nodes == Pi_ar.shape[0]
        assert self.residue.num_nodes == Pi_ar.shape[1]

        # residue → SSE
        assert self.residue.num_nodes == Pi_rs.shape[0]
        assert self.sse.num_nodes == Pi_rs.shape[1]

def build_hierarchical_protein_graph(
        surface_path: str,
        process_data,
        surface_k: int = 8,
        atom_encoder=None,
        surface_encoder=None,
        device: str = "cuda"
) -> HierarchicalProteinGraph:

    partitions, sse_labels = extract_partition_matrices(
        surface_path=surface_path,
        process_data=process_data,
    )

    mesh = trimesh.load(surface_path, process=False)
    mesh = mesh_simplification_quadric_decimation(
        mesh,
        target_faces=partitions['surface_to_atom'].shape[0]
    )

    face_centroids = mesh.vertices[mesh.faces].mean(axis=1)

    # ==================================================
    # Surface Level
    # ==================================================
    A_surface     = build_surface_knn_adjacency(face_centroids, k=surface_k)
    X_surface     = get_surface_features(mesh, encoder=surface_encoder, device=device)
    X_surface     = torch.tensor(X_surface, dtype=torch.float32)
    surface_level = ProteinGraphLevel("surface", A_surface, X_surface)

    # ==================================================
    # Atom Level — from .pt
    # ==================================================
    Pi_sa      = partitions["surface_to_atom"]
    A_atom_raw = coarsen_adjacency(A_surface, Pi_sa)
    A_atom     = normalize_adjacency(A_atom_raw)

    X_atom     = get_atom_features(process_data, encoder=atom_encoder, device=device)
    X_atom     = torch.tensor(X_atom, dtype=torch.float32)
    atom_level = ProteinGraphLevel("atom", A_atom, X_atom)

    # ==================================================
    # Residue Level — from .pt
    # ==================================================
    Pi_ar     = partitions["atom_to_residue"]
    A_res_raw = coarsen_adjacency(A_atom_raw, Pi_ar)
    A_res     = normalize_adjacency(A_res_raw)

    X_res         = get_residue_features(process_data)
    residue_level = ProteinGraphLevel("residue", A_res, X_res)

    # ==================================================
    # SSE Level
    # ==================================================
    Pi_rs     = partitions["residue_to_sse"]
    A_sse_raw = coarsen_adjacency(A_res_raw, Pi_rs)
    A_sse     = normalize_adjacency(A_sse_raw)

    X_sse_np  = get_sse_features(Pi_rs, sse_labels)
    X_sse     = torch.tensor(X_sse_np, dtype=torch.float32)
    sse_level = ProteinGraphLevel("sse", A_sse, X_sse)

    # ==================================================
    # Protein Level
    # ==================================================
    Pi_sp         = partitions["sse_to_protein"]
    A_protein_raw = coarsen_adjacency(A_sse_raw, Pi_sp)
    A_protein     = normalize_adjacency(A_protein_raw)
    protein_level = ProteinGraphLevel("protein", A_protein, X=None)

    # ==================================================
    # Return
    # ==================================================
    graph = HierarchicalProteinGraph(
        surface=surface_level,
        atom=atom_level,
        residue=residue_level,
        sse=sse_level,
        protein=protein_level,
        partitions=partitions,
    )

    return graph

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Hierarchical protein graph construction"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--pt_dir",
        type=str,
        help="Directory containing .pt files to reconstruct PDB and build graph"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save generated graph .pt files"
    )
    
    parser.add_argument(
    "--atom_encoder_path",
    type=str,
    default=None,
    help="Path to pretrained atom encoder (default: None)"
    )

    parser.add_argument(
        "--surface_encoder_path",
        type=str,
        default=None,
        help="Path to pretrained surface encoder (default: None)"
    )
    
    parser.add_argument("--task", type=str, default=None)
        
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --------------------------------------------------
    # Load pretrained encoders (encoder-only)
    # --------------------------------------------------

    atom_encoder = None
    surface_encoder = None

    # ----------------------------
    # Atom encoder
    # ----------------------------
    if args.atom_encoder_path is not None:
        if not os.path.exists(args.atom_encoder_path):
            raise ValueError(f"Atom encoder path not found: {args.atom_encoder_path}")

        print(f"Loading atom encoder from {args.atom_encoder_path}")

        checkpoint = torch.load(args.atom_encoder_path, map_location="cpu")

        # Load config from checkpoint
        atom_cfg = checkpoint["config"]["atom"]

        # Build encoder (EGNN)
        atom_encoder = EGNN(
            in_node_nf=atom_cfg["atom_feat_dim"],
            hidden_nf=atom_cfg["hidden_dim"],
            out_node_nf=atom_cfg["latent_dim"],
            in_edge_nf=1,
            n_layers=atom_cfg["n_layers"]
        )

        # Load weights directly
        atom_encoder.load_state_dict(checkpoint["model_state_dict"])
        atom_encoder = atom_encoder.to(device)
        atom_encoder.eval()

    else:
        print("No atom encoder provided → using None")


    # ----------------------------
    # Surface encoder
    # ----------------------------
    if args.surface_encoder_path is not None:
        if not os.path.exists(args.surface_encoder_path):
            raise ValueError(f"Surface encoder path not found: {args.surface_encoder_path}")

        print(f"Loading surface encoder from {args.surface_encoder_path}")

        checkpoint = torch.load(args.surface_encoder_path, map_location="cpu")

        # Load config from checkpoint
        surface_cfg = checkpoint["config"]["surface"]

        # Build ONLY encoder (EMNN)
        surface_encoder = EMNN(
            in_node_nf=surface_cfg["input_dim"],
            hidden_nf=surface_cfg["hidden_dim"],
            out_node_nf=surface_cfg["latent_dim"],
            in_edge_nf=1,
            n_layers=surface_cfg["n_layers"]
        )

        # Load weights
        surface_encoder.load_state_dict(checkpoint["model_state_dict"])
        surface_encoder = surface_encoder.to(device)
        surface_encoder.eval()

    else:
        print("No surface encoder provided → using None")
    
    # --------------------------------------------------
    # Run
    # --------------------------------------------------

    pt_files = sorted(glob(os.path.join(args.pt_dir, "*.pt")))
    print(f"Found {len(pt_files)} PT files")

    for pt_path in tqdm(pt_files, desc="Processing PT files"):
        
        base_name = os.path.splitext(os.path.basename(pt_path))[0]
        output_path = os.path.join(args.output_dir, base_name + ".pt")
        if os.path.exists(output_path):
            continue


        tmp_dir = os.path.join(args.output_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        reconstructed_pdb_path = os.path.join(tmp_dir, f"{base_name}.pdb")
        surface_obj_path       = os.path.join(tmp_dir, f"{base_name}.obj")

        try:
            process_data = torch.load(pt_path, map_location="cpu")

            if args.task == "BindingSite":
                process_data = _binding_site_transform(process_data)

            write_atom_to_pdb(process_data, reconstructed_pdb_path)

            export_surface(
                pdb_path=reconstructed_pdb_path,
                output_path=surface_obj_path,
                surface_quality=0,
                solvent_radius=1.4,
                selection="all"
            )

            graph = build_hierarchical_protein_graph(
                surface_path=surface_obj_path,
                process_data=process_data,
                atom_encoder=atom_encoder,
                surface_encoder=surface_encoder,
                device=device
            )

            if args.task == "BindingSite":
                graph.node_y = process_data.node_y

            torch.save(graph, os.path.join(args.output_dir, base_name + ".pt"))

        except Exception as e:
            print(f"Skipping {base_name}: {e}")

        finally:
            if os.path.exists(reconstructed_pdb_path):
                os.remove(reconstructed_pdb_path)
            if os.path.exists(surface_obj_path):
                os.remove(surface_obj_path)