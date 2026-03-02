from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
import argparse
import trimesh
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.models.esmc import ESMC
client = ESMC.from_pretrained("esmc_300m").to("cuda")
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

def build_surface_graph(mesh, device):
    """
    Build vertex-based surface graph for EMNN.

    Nodes: mesh vertices
    Edges: mesh edges (true triangle adjacency)
    Faces: mesh triangles (vertex indices)
    """

    # --------------------------------------------------
    # Vertices
    # --------------------------------------------------
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    x = torch.tensor(vertices, dtype=torch.float32).to(device)

    # --------------------------------------------------
    # Build edges from faces
    # Each triangle (i, j, k) gives edges:
    # (i,j), (j,k), (k,i)
    # --------------------------------------------------
    rows, cols = [], []

    for tri in faces:
        i, j, k = tri

        rows.extend([i, j, k])
        cols.extend([j, k, i])

        rows.extend([j, k, i])
        cols.extend([i, j, k])

    edge_index = torch.LongTensor([rows, cols]).to(device)

    # Optional: remove duplicate edges
    edge_index = torch.unique(edge_index, dim=1)

    # Dummy edge features (can upgrade later)
    edge_attr = torch.ones(edge_index.shape[1], 1).to(device)

    # --------------------------------------------------
    # Face index (3, F)
    # --------------------------------------------------
    face_index = torch.LongTensor(faces.T).to(device)

    return x, edge_index, face_index, edge_attr

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
    """
    A single level in the protein hierarchy.
    """
    name: str
    A: csr_matrix              # (N, N) adjacency
    X: np.ndarray              # (N, d) node features

    @property
    def num_nodes(self) -> int:
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
        surface_radius: float = 1.0,
        use_pretrained_atom: bool = False,
        use_pretrained_surface: bool = False,
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
    )

    # --------------------------------------------------
    # Load surface mesh
    # --------------------------------------------------
    surface_path = pdb_path.replace(".pdb", "_surface.obj")

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
        use_pretrained=use_pretrained_surface,
        encoder=surface_encoder,
        device=device,
    )

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
        use_pretrained=use_pretrained_atom,
        encoder=atom_encoder,
        device=device,
    )

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

    residue_level = ProteinGraphLevel(
        name="residue",
        A=A_res,
        X=get_residue_features(protein_seq),
    )

    # ==================================================
    # ---------- SSE Level ----------
    # ==================================================
    Pi_rs = partitions["residue_to_sse"]
    A_sse = coarsen_adjacency(A_res, Pi_rs)

    A_sse.data[:] = 1.0
    A_sse.eliminate_zeros()

    sse_level = ProteinGraphLevel(
        name="sse",
        A=A_sse,
        X=get_sse_features(Pi_rs, residue_level.X),
    )

    # ==================================================
    # ---------- Protein Level ----------
    # ==================================================
    Pi_sp = partitions["sse_to_protein"]
    A_protein = coarsen_adjacency(A_sse, Pi_sp)

    A_protein.data[:] = 1.0
    A_protein.eliminate_zeros()

    protein_level = ProteinGraphLevel(
        name="protein",
        A=A_protein,
        X=get_protein_features(residue_level.X),
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

    # --------------------------------------------------
    # Input (single or directory)
    # --------------------------------------------------
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--pdb_path",
        type=str,
        help="Path to single PDB file"
    )

    group.add_argument(
        "--pdb_dir",
        type=str,
        help="Directory containing multiple PDB files"
    )

    # --------------------------------------------------
    # Surface options
    # --------------------------------------------------
    parser.add_argument(
        "--surface_radius",
        type=float,
        default=1.0,
        help="Surface radius parameter (Å)"
    )

    # --------------------------------------------------
    # Pretrained encoders
    # --------------------------------------------------
    parser.add_argument(
        "--use_pretrained_atom",
        action="store_true",
        help="Use pretrained atomic encoder"
    )

    parser.add_argument(
        "--use_pretrained_surface",
        action="store_true",
        help="Use pretrained surface encoder"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to unified YAML config file"
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------
    # Load encoders
    # --------------------------------------------------
    atom_encoder = None
    surface_encoder = None

    # --------------------------------------------
    # Load Atom Encoder
    # --------------------------------------------
    if args.use_pretrained_atom:

        atom_cfg = config["atom"]

        atom_encoder = EGNN(
            in_node_nf=atom_cfg["atom_feat_dim"],
            hidden_nf=atom_cfg["hidden_dim"],
            out_node_nf=atom_cfg["latent_dim"],
            in_edge_nf=1,
            n_layers=atom_cfg["n_layers"],
            device=device
        )

        ckpt = torch.load(atom_cfg["save_path"], map_location=device)

        # If you saved config + state_dict
        if "model_state_dict" in ckpt:
            atom_encoder.load_state_dict(ckpt["model_state_dict"])
        else:
            atom_encoder.load_state_dict(ckpt)

        atom_encoder.eval()

    # --------------------------------------------
    # Load Surface Encoder
    # --------------------------------------------
    if args.use_pretrained_surface:

        surface_cfg = config["surface"]

        surface_encoder = EMNN(
            in_node_nf=surface_cfg["input_dim"],
            hidden_nf=surface_cfg["hidden_dim"],
            out_node_nf=surface_cfg["latent_dim"],
            in_edge_nf=1,
            n_layers=surface_cfg["n_layers"],
            device=device
        )

        ckpt = torch.load(surface_cfg["save_path"], map_location=device)

        if "model_state_dict" in ckpt:
            surface_encoder.load_state_dict(ckpt["model_state_dict"])
        else:
            surface_encoder.load_state_dict(ckpt)

        surface_encoder.eval()

    # --------------------------------------------------
    # Helper function to process one PDB
    # --------------------------------------------------
    def process_pdb(pdb_path):

        graph = build_hierarchical_protein_graph(
            pdb_path=pdb_path,
            surface_radius=args.surface_radius,
            use_pretrained_atom=args.use_pretrained_atom,
            use_pretrained_surface=args.use_pretrained_surface,
            atom_encoder=atom_encoder,
            surface_encoder=surface_encoder,
            device=device
        )

        print("\n=== Hierarchical Protein Graph Summary ===")
        for level_name, level in [
            ("surface", graph.surface),
            ("atom", graph.atom),
            ("residue", graph.residue),
            ("sse", graph.sse),
            ("protein", graph.protein),
        ]:
            print(
                f"{level_name:8s} | "
                f"nodes: {level.num_nodes:5d} | "
                f"edges: {level.A.nnz // 2:6d} | "
                f"feat shape: {level.X.shape}"
            )

        print("\nHierarchy construction SUCCESS")

    # --------------------------------------------------
    # Run
    # --------------------------------------------------
    if args.pdb_path:

        process_pdb(args.pdb_path)

    else:
        pdb_files = sorted(glob(os.path.join(args.pdb_dir, "*.pdb")))

        print(f"Found {len(pdb_files)} PDB files")

        for pdb_path in tqdm(pdb_files, desc="Processing PDBs"):
            process_pdb(pdb_path)