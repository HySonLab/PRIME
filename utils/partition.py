from addict import Dict
import numpy as np
from Bio.PDB import PDBParser
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
import argparse
from Bio.PDB import Structure
import pydssp
import torch
import subprocess
from pathlib import Path
import trimesh
from Bio.PDB import Selection
import open3d as o3d

MAX_FACES = 500

def mesh_simplification_quadric_decimation(mesh, target_faces):
    """
    Simplify mesh using Open3D's quadric decimation.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh
    target_faces : int
        Desired number of faces after simplification

    Returns
    -------
    simplified_mesh : trimesh.Trimesh
        Simplified mesh
    """

    # Convert to Open3D mesh
    mesh_o3d = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces)
    )

    mesh_o3d.compute_vertex_normals()

    # Simplify using quadric decimation
    simplified_o3d = mesh_o3d.simplify_quadric_decimation(
        target_number_of_triangles=target_faces
    )

    # Convert back to Trimesh
    simplified_mesh = trimesh.Trimesh(
        vertices=np.asarray(simplified_o3d.vertices),
        faces=np.asarray(simplified_o3d.triangles),
        process=False,
    )

    return simplified_mesh

def build_atom_to_residue_assignment(
    structure: Structure,
    include_hetero: bool = False
):
    """
    Build atom -> residue assignment matrix from a parsed PDB structure.

    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        Parsed PDB structure.
    include_hetero : bool, optional
        Whether to include hetero atoms (HETATM). Default is False.

    Returns
    -------
    Pi : scipy.sparse.csr_matrix, shape (N_atoms, N_residues)
        Atom-to-residue assignment matrix.
    """

    residue_keys = []

    for model in structure:
        for chain in model:
            for residue in chain:
                hetero_flag, _, _ = residue.id

                if not include_hetero and hetero_flag.strip():
                    continue

                res_key = (chain.id, residue.id)

                for _ in residue:
                    residue_keys.append(res_key)

    # --- residue indexing ---
    unique_residues = list(dict.fromkeys(residue_keys))
    residue_to_index = {res: i for i, res in enumerate(unique_residues)}

    N_atoms = len(residue_keys)
    N_res = len(unique_residues)

    rows = np.arange(N_atoms)
    cols = np.array([residue_to_index[rk] for rk in residue_keys])
    data = np.ones(N_atoms, dtype=np.int8)

    Pi = csr_matrix((data, (rows, cols)), shape=(N_atoms, N_res))

    # --- sanity ---
    assert np.all(Pi.sum(axis=1).A1 == 1)
    assert np.all(Pi.sum(axis=0).A1 > 0)

    return Pi

def build_residue_to_sse_assignment(
    structure: Structure,
    include_hetero: bool = False
):
    """
    Build residue -> secondary structure element assignment matrix
    using pydssp, from a parsed PDB structure.

    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        Parsed PDB structure.
    include_hetero : bool, optional
        Whether to include hetero residues (default: False).

    Returns
    -------
    Pi : scipy.sparse.csr_matrix, shape (N_residues, N_SSEs)
        Residue-to-secondary-structure-element assignment matrix.
    """

    backbone_atoms = ["N", "CA", "C", "O"]

    coords = []
    residue_indices = []  # keeps residue order consistent

    # --- Extract backbone coordinates per residue ---
    for model in structure:
        for chain in model:
            for residue in chain:
                hetero_flag, _, _ = residue.id

                if not include_hetero and hetero_flag.strip():
                    continue

                # ensure residue has full backbone
                if not all(atom_name in residue for atom_name in backbone_atoms):
                    continue

                residue_indices.append((chain.id, residue.id))

                atom_coords = []
                for atom_name in backbone_atoms:
                    atom_coords.append(residue[atom_name].coord)

                coords.append(atom_coords)

    if len(coords) == 0:
        raise ValueError("No valid residues with complete backbone found.")

    # shape: (L, 4, 3)
    coord_array = np.asarray(coords, dtype=np.float32)   # (L, 4, 3)
    coord_tensor = torch.from_numpy(coord_array)

    # --- Run pydssp ---
    dssp = pydssp.assign(coord_tensor, out_type="index")
    labels = dssp.cpu().numpy().tolist()  # 0: loop, 1: helix, 2: strand

    N_res = len(labels)

    # --- Group residues into SSE segments ---
    sse_ids = np.zeros(N_res, dtype=np.int32)
    current_sse = 0
    sse_ids[0] = current_sse

    for i in range(1, N_res):
        if labels[i] != labels[i - 1]:
            current_sse += 1
        sse_ids[i] = current_sse

    N_sse = current_sse + 1

    # --- Build assignment matrix ---
    rows = np.arange(N_res)
    cols = sse_ids
    data = np.ones(N_res, dtype=np.int8)

    Pi = csr_matrix((data, (rows, cols)), shape=(N_res, N_sse))

    # --- Sanity checks ---
    assert np.all(Pi.sum(axis=1).A1 == 1)
    assert np.all(Pi.sum(axis=0).A1 > 0)

    return Pi

def build_surface_to_atom_assignment(
    structure: Structure,
    surface_path: str,
    include_hetero: bool = False
):
    """
    Build surface vertex -> atom assignment matrix using spatial proximity.

    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        Parsed PDB structure.
    surface_path : str
        Path to surface mesh file (OBJ format).
    include_hetero : bool, optional
        Whether to include hetero atoms (HETATM). Default is False.

    Returns
    -------
    Pi : scipy.sparse.csr_matrix, shape (N_vertices, N_atoms)
        Surface vertex-to-atom assignment matrix.
    """

    # --- Load surface vertices ---
    mesh = trimesh.load(surface_path)
    mesh = mesh_simplification_quadric_decimation(mesh, target_faces=MAX_FACES)
    surface_vertices = np.asarray(mesh.vertices)   # (Nv, 3)
    surface_faces = np.asarray(mesh.faces)         # (Nf, 3)
    face_centroids = surface_vertices[surface_faces].mean(axis=1)

    print(f"Number of vertices: {surface_vertices.shape[0]}")
    print(f"Number of faces:    {surface_faces.shape[0]}")

    # --- Extract atom coordinates and indices ---
    atom_coords = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                hetero_flag, _, _ = residue.id

                if not include_hetero and hetero_flag.strip():
                    continue

                for atom in residue:
                    atom_coords.append(atom.coord)

    if len(atom_coords) == 0:
        raise ValueError("No valid atoms found in structure.")

    atom_coords = np.asarray(atom_coords, dtype=np.float32)  # shape: (N_atoms, 3)

    # --- Build KD-tree and query nearest atoms for each vertex ---
    kdtree = cKDTree(atom_coords)
    distances, nearest_atom_indices = kdtree.query(face_centroids, k=1)

    N_faces = face_centroids.shape[0]
    N_atoms = atom_coords.shape[0]

    rows = np.arange(N_faces)
    cols = nearest_atom_indices
    data = np.ones(N_faces, dtype=np.int8)

    Pi = csr_matrix((data, (rows, cols)), shape=(N_faces, N_atoms))
    
    return Pi

def print_structure_info(
    structure: Structure,
    include_hetero: bool = False
):
    """
    Print basic information about the PDB structure.

    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        Parsed PDB structure.
    include_hetero : bool, optional
        Whether to include hetero residues (HETATM).
    """

    num_atoms = 0
    num_residues = 0

    for model in structure:
        for chain in model:
            for residue in chain:
                hetero_flag, _, _ = residue.id

                if not include_hetero and hetero_flag.strip():
                    continue

                num_residues += 1
                num_atoms += len(residue)

    print("Number of residues:", num_residues)
    print("Number of atoms:   ", num_atoms)
    
def extract_partition_matrices(
    pdb_path: str | None = None,
    surface_path: str | None = None,
) -> Dict[str, "torch.Tensor"]:
    """
    Extract all hierarchical partition matrices for a protein structure.

    Parameters
    ----------
    pdb_path : str, optional
        Path to the original PDB file (needed for surface generation).
    surface_path : str, optional
        Path to precomputed surface OBJ file. If not provided,
        it will be inferred from pdb_path.

    Returns
    -------
    partitions : dict
        Dictionary containing partition matrices:
        - 'surface_to_atom'
        - 'atom_to_residue'
        - 'residue_to_sse'
    """

    partitions = {}

    # ---------- Surface to Atom ----------
    if surface_path is None:
        if pdb_path is None:
            raise ValueError(
                "Either `surface_path` or `pdb_path` must be provided."
            )
        surface_path = pdb_path.replace(".pdb", "_surface.obj")

    if pdb_path is not None:
        # --- Parse PDB---
        parser_pdb = PDBParser(QUIET=True)
        structure = parser_pdb.get_structure("protein", pdb_path)
        
        print_structure_info(structure)

    Pi_surface_to_atom = build_surface_to_atom_assignment(
        structure, surface_path
    )
    partitions["surface_to_atom"] = Pi_surface_to_atom

    # ---------- Atom to Residue ----------
    Pi_atom_to_res = build_atom_to_residue_assignment(structure)
    partitions["atom_to_residue"] = Pi_atom_to_res

    # ---------- Residue to SSE ----------
    Pi_res_to_sse = build_residue_to_sse_assignment(structure)
    partitions["residue_to_sse"] = Pi_res_to_sse
    
    # ---------- SSE to Protein ----------
    N_sse = Pi_res_to_sse.shape[1]
    Pi_sse_to_prot = csr_matrix(
        (
            np.ones(N_sse, dtype=np.int8),
            (np.arange(N_sse), np.zeros(N_sse, dtype=np.int32)),
        ),
        shape=(N_sse, 1),
    )

    partitions["sse_to_protein"] = Pi_sse_to_prot

    print("Extracted partition matrices:")
    for name, Pi in partitions.items():
        print(f"  {name}: shape {Pi.shape}")
    
    return structure, partitions

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Hierarchical protein partition builder"
    )
    parser.add_argument(
        "--pdb_path",
        type=str,
        help="Path to input PDB file"
    )
    
    args = parser.parse_args()
    
    structure, partitions = extract_partition_matrices(pdb_path=args.pdb_path)