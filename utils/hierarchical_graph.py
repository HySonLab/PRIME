from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
import argparse
import trimesh
from esm.sdk.api import ESMProtein, LogitsConfig

from esm.models.esmc import ESMC
client = ESMC.from_pretrained("esmc_600m").to("cuda")

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import open3d as o3d
import torch
from glob import glob
from tqdm import tqdm
import yaml
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.partition import extract_partition_matrices, mesh_simplification_quadric_decimation
from models.egnn import EGNN
from models.emnn import EMNN
from torch_cluster import knn_graph
from pymol import cmd

MAX_FACES = 1024

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

def write_atom_to_pdb(data, output_path):

    coords = data.coords.numpy()
    print(f"Coords shape: {coords.shape}")
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

    print(f"PDB written to {output_path}")

def export_surface(
    pdb_path,
    output_path,
    surface_quality=0,
    solvent_radius=1.4,
    selection="all",
):
    cmd.load(pdb_path, "prot")

    cmd.hide("everything", selection)
    cmd.show("surface", selection)

    cmd.set("surface_quality", surface_quality)
    cmd.set("solvent_radius", solvent_radius)

    cmd.save(output_path, selection)
    cmd.delete("all")

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

def build_surface_radius_adjacency(
    face_centroids: np.ndarray,
    radius: float,
):
    """
    Build radius-based adjacency for surface faces.

    Parameters
    ----------
    face_centroids : (N_faces, 3) np.ndarray
    radius : float
        Cutoff distance (Å).

    Returns
    -------
    A_surface : csr_matrix, shape (N_faces, N_faces)
    """

    tree = cKDTree(face_centroids)
    pairs = tree.query_pairs(r=radius)

    rows, cols = zip(*pairs) if pairs else ([], [])

    data = np.ones(len(rows), dtype=np.int8)
    A = csr_matrix((data, (rows, cols)),
                   shape=(len(face_centroids), len(face_centroids)))

    # make symmetric
    A = A + A.T

    return A

def get_surface_features(
        mesh,
        use_pretrained=False,
        encoder=None,
        device="cpu",
        input_dim=16):
    """
    Surface features.

    If use_pretrained=False:
        Returns handcrafted face-level features (N_faces, 3)

    If use_pretrained=True:
        Returns pretrained encoder embeddings (N_faces, latent_dim)
    """

    # ==========================================================
    # OPTION 1: Use pretrained encoder
    # ==========================================================
    if use_pretrained:

        if encoder is None:
            raise ValueError("Encoder must be provided when use_pretrained=True")

        # Build surface graph (vertex-level)
        x, edge_index, face_index, edge_attr = build_surface_graph(
            mesh,
            device=device
        )

        # Dummy node features (same as during pretraining)
        h = torch.ones(x.size(0), input_dim).to(device)

        # Run encoder
        with torch.no_grad():
            z_vertex, _ = encoder(
                h,
                x,
                edge_index,
                face_index,
                edge_attr
            )

        # ---------------------------------------------
        # Convert vertex embeddings → face embeddings
        # (average over 3 vertices per face)
        # ---------------------------------------------
        faces = mesh.faces
        z_vertex = z_vertex.cpu().numpy()

        z_face = (
            z_vertex[faces[:, 0]] +
            z_vertex[faces[:, 1]] +
            z_vertex[faces[:, 2]]
        ) / 3.0

        return z_face.astype(np.float32)

    # ==========================================================
    # OPTION 2: Handcrafted features (original behavior)
    # ==========================================================
    else:

        faces = mesh.faces
        vertices = mesh.vertices

        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        face_areas = 0.5 * np.linalg.norm(
            np.cross(v1 - v0, v2 - v0), axis=1
        )

        centroids = (v0 + v1 + v2) / 3.0
        protein_center = vertices.mean(axis=0)

        dist_to_center = np.linalg.norm(
            centroids - protein_center,
            axis=1
        )

        bias = np.ones_like(face_areas)

        X_surface = np.stack(
            [face_areas, dist_to_center, bias],
            axis=1
        ).astype(np.float32)

        return X_surface

def get_atom_features(
        structure,
        use_pretrained=False,
        encoder=None,
        device="cpu",
        k=16):
    """
    Atom-level features.

    If use_pretrained=False:
        Returns handcrafted features (N_atoms, 7)

    If use_pretrained=True:
        Returns pretrained encoder embeddings (N_atoms, latent_dim)
    """

    # ==========================================================
    # OPTION 1: Use pretrained encoder
    # ==========================================================
    if use_pretrained:

        if encoder is None:
            raise ValueError("Encoder must be provided when use_pretrained=True")

        # Build atomic graph
        coords = []
        features = []

        elements = ["C", "N", "O", "S", "P", "H"]
        elem_to_idx = {e: i for i, e in enumerate(elements)}

        for atom in structure.get_atoms():

            if atom.element == 'H':
                continue

            coords.append(atom.coord)

            # simple one-hot initial feature
            one_hot = np.zeros(len(elements), dtype=np.float32)
            elem = atom.element.strip()
            if elem in elem_to_idx:
                one_hot[elem_to_idx[elem]] = 1.0

            features.append(one_hot)

        coords = torch.from_numpy(np.array(coords)).float().to(device)
        h = torch.from_numpy(np.array(features)).float().to(device)

        # Build KNN graph
        edge_index = knn_graph(coords, k=k, loop=False)

        row, col = edge_index
        dist = torch.norm(coords[row] - coords[col], dim=1, keepdim=True)

        # Run encoder
        with torch.no_grad():
            z, _ = encoder(h, coords, edge_index, dist)

        return z.cpu().numpy()

    # ==========================================================
    # OPTION 2: Handcrafted features (original behavior)
    # ==========================================================
    else:

        elements = ["C", "N", "O", "S", "P", "H"]
        elem_to_idx = {e: i for i, e in enumerate(elements)}

        features = []

        for atom in structure.get_atoms():
            elem = atom.element.strip()

            one_hot = np.zeros(len(elements), dtype=np.float32)
            if elem in elem_to_idx:
                one_hot[elem_to_idx[elem]] = 1.0

            is_backbone = atom.get_name() in {"N", "CA", "C", "O"}

            features.append(
                np.concatenate([one_hot, [float(is_backbone)]])
            )

        return np.asarray(features, dtype=np.float32)

def get_residue_features(protein_seq: str):
    """
    Residue-level features.
    """
    protein = ESMProtein(sequence=protein_seq)
    protein_tensor = client.encode(protein)
    logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))

    embeddings = logits_output.embeddings
    # remove CLS and EOS
    residue_features = embeddings[:, 1:-1, :]
    # convert to numpy
    residue_features = residue_features.cpu().numpy().squeeze(0)
    return residue_features

def get_sse_features(partition, residue_features):
    """
    SSE-level features.
    """  
    sse_features = partition.T @ residue_features
    counts = np.array(partition.sum(axis=0)).flatten()
    counts[counts == 0] = 1
    sse_features /= counts[:, None]
    return sse_features

def get_protein_features(residue_features):
    """
    Protein-level features using ESMC embeddings.
    """
    protein_feature = residue_features.mean(axis=0, keepdims=True)
    return protein_feature

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
        pdb_path: str,
        surface_path: str,
        surface_radius: float = 1.0,
        use_pretrained_atom_encoder: bool = False,
        use_pretrained_surface_encoder: bool = False,
        atom_encoder=None,
        surface_encoder=None,
        device: str = "cpu"
) -> HierarchicalProteinGraph:
    """
    Build a hierarchical protein graph.

    Supports optional pretrained atom and surface encoders.
    """

    # --------------------------------------------------
    # Extract partition matrices
    # --------------------------------------------------
    print("Extracting partition matrices...")

    structure, partitions = extract_partition_matrices(
        pdb_path=pdb_path,
        surface_path=surface_path,
    )

    mesh = trimesh.load(surface_path, process=False)

    # Match surface resolution to partition size
    mesh = mesh_simplification_quadric_decimation(
        mesh,
        target_faces=partitions['surface_to_atom'].shape[0]
    )

    face_centroids = mesh.vertices[mesh.faces].mean(axis=1)

    protein_seq = "".join(get_sequence_from_pdb(pdb_path).values())

    # ==================================================
    # ---------- Surface Level ----------
    # ==================================================
    A_surface = build_surface_radius_adjacency(
        face_centroids,
        radius=surface_radius,
    )

    X_surface = get_surface_features(
        mesh,
        use_pretrained=use_pretrained_surface_encoder,
        encoder=surface_encoder,
        device=device,
    )
    
    X_surface = torch.tensor(X_surface, dtype=torch.float32)

    surface_level = ProteinGraphLevel(
        name="surface",
        A=A_surface,
        X=X_surface,
    )

    # ==================================================
    # ---------- Atom Level ----------
    # ==================================================
    Pi_sa = partitions["surface_to_atom"]
    A_atom = coarsen_adjacency(A_surface, Pi_sa)

    A_atom.data[:] = 1.0
    A_atom.eliminate_zeros()

    X_atom = get_atom_features(
        structure,
        use_pretrained=use_pretrained_atom_encoder,
        encoder=atom_encoder,
        device=device,
    )
    
    X_atom = torch.tensor(X_atom, dtype=torch.float32)

    atom_level = ProteinGraphLevel(
        name="atom",
        A=A_atom,
        X=X_atom,
    )

    # ==================================================
    # ---------- Residue Level ----------
    # ==================================================
    Pi_ar = partitions["atom_to_residue"]
    A_res = coarsen_adjacency(A_atom, Pi_ar)

    A_res.data[:] = 1.0
    A_res.eliminate_zeros()
    
    X_res = get_residue_features(protein_seq)
    X_res = torch.tensor(X_res, dtype=torch.float32)

    residue_level = ProteinGraphLevel(
        name="residue",
        A=A_res,
        X=X_res,
    )

    # ==================================================
    # ---------- SSE Level ----------
    # ==================================================
    Pi_rs = partitions["residue_to_sse"]
    A_sse = coarsen_adjacency(A_res, Pi_rs)

    A_sse.data[:] = 1.0
    A_sse.eliminate_zeros()
    
    X_sse = get_sse_features(Pi_rs, residue_level.X.numpy())
    X_sse = torch.tensor(X_sse, dtype=torch.float32)

    sse_level = ProteinGraphLevel(
        name="sse",
        A=A_sse,
        X=X_sse,
    )

    # ==================================================
    # ---------- Protein Level ----------
    # ==================================================
    Pi_sp = partitions["sse_to_protein"]
    A_protein = coarsen_adjacency(A_sse, Pi_sp)

    A_protein.data[:] = 1.0
    A_protein.eliminate_zeros()
     
    X_protein = get_protein_features(residue_level.X.numpy())
    X_protein = torch.tensor(X_protein, dtype=torch.float32)

    protein_level = ProteinGraphLevel(
        name="protein",
        A=A_protein,
        X=X_protein,
    )

    # ==================================================
    return HierarchicalProteinGraph(
        surface=surface_level,
        atom=atom_level,
        residue=residue_level,
        sse=sse_level,
        protein=protein_level,
        partitions=partitions,
    )

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
        "--use_pretrained_atom_encoder",
        action="store_true",
        help="Use pretrained atom-level features"
    )

    parser.add_argument(
        "--use_pretrained_surface_encoder",
        action="store_true",
        help="Use pretrained surface-level features"
    )
        
    args = parser.parse_args()
    
    # --------------------------------------------------
    # Run
    # --------------------------------------------------

    pt_files = sorted(glob(os.path.join(args.pt_dir, "*.pt")))
    print(f"Found {len(pt_files)} PT files")

    for pt_path in tqdm(pt_files, desc="Processing PT files"):

        base_name = os.path.splitext(os.path.basename(pt_path))[0]
        reconstructed_pdb_path = os.path.join("/home/dvnguye2/PRL/tmp/tmp.pdb")
        
        process_data = torch.load(pt_path, map_location="cpu")

        # Reconstruct PDB
        write_atom_to_pdb(process_data, reconstructed_pdb_path)
        
        # Create surface mesh
        surface_obj_path = f"/home/dvnguye2/PRL/tmp/tmp.obj"
        export_surface(
            pdb_path=reconstructed_pdb_path,
            output_path=surface_obj_path,
            surface_quality=0,
            solvent_radius=1.4,
            selection="all"
        )

        try:
            # Build Graph
            graph = build_hierarchical_protein_graph(
                pdb_path=reconstructed_pdb_path,
                surface_path=surface_obj_path,
                use_pretrained_atom_encoder=args.use_pretrained_atom_encoder,
                use_pretrained_surface_encoder=args.use_pretrained_surface_encoder
            )

            # Save Graph
            graph_output_path = os.path.join(args.output_dir, base_name + ".pt")
            torch.save(graph, graph_output_path)
        except Exception as e:
            print(f"Error processing {pt_path}: {e}")
            continue