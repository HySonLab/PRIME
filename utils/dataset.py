import os
import yaml
import torch
from torch.utils.data import Dataset
import torch
import yaml
import torch

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.hierarchical_graph import *
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np

def get_sequence(residues):
    sequence = []

    for res in residues:
        res = res.upper()
        if res in THREE_TO_ONE:
            sequence.append(THREE_TO_ONE[res])
        else:
            sequence.append("X")

    sequence = "".join(sequence)
    
    return sequence

LABEL_LINE = {"MF": 1, "BP": 5, "CC": 9}

def _parse_go_labels(task_dir, branch):
    """
    Parse GO labels from nrPDB-GO_annot.tsv for a specific branch.
    Returns dict: {"1AD3-A": tensor([local indices])}
    """
    annot_path = os.path.join(task_dir, "nrPDB-GO_annot.tsv")

    with open(annot_path, "r") as f:
        all_labels = f.readlines()[LABEL_LINE[branch]].strip("\n").split("\t")

    df = pd.read_csv(annot_path, sep="\t", skiprows=12)
    df.columns = ["PDB", "MF", "BP", "CC"]
    df.set_index("PDB", inplace=True)

    labels = df[branch].dropna().to_dict()
    labels = {k: v.split(",") for k, v in labels.items()}

    label_encoder = LabelEncoder().fit(all_labels)
    labels = {
        k: torch.tensor(label_encoder.transform(v), dtype=torch.long)
        for k, v in tqdm(labels.items(), desc=f"Encoding GO-{branch}")
    }

    print(f"[GO-{branch}] {len(labels)} proteins, {len(all_labels)} classes")
    return labels

class HierarchicalGraphDataset(Dataset):
    """
    Dataset for hierarchical graph downstream tasks.
    Loads graph from graph_dir.
    Loads label from processed_dir or annot file (GeneOntology).
    """

    def __init__(
        self,
        config_path,
        task_name,
        split="train",
        fold_test_type=None,
        device=None,
        go_branch=None,
    ):
        self.device     = device
        self.go_branch  = go_branch
        super().__init__()

        # --------------------------------------------------
        # Load YAML config
        # --------------------------------------------------
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)

        self.data_root      = full_config["data_root"]
        self.task_name      = task_name
        self.task_cfg       = full_config["tasks"][task_name]
        self.split          = split
        self.fold_test_type = fold_test_type

        self.task_dir = os.path.join(self.data_root, task_name)

        self.processed_dir = os.path.join(
            self.task_dir, self.task_cfg["processed_dir"]
        )
        self.graph_dir = os.path.join(
            self.task_dir, self.task_cfg["graph_dir"]
        )

        # --------------------------------------------------
        # Load split IDs
        # --------------------------------------------------
        split_file = self._resolve_split_file()
        raw_ids    = self._load_ids(split_file)

        # --------------------------------------------------
        # Filter only samples that have BOTH graph + label
        # --------------------------------------------------
        valid_ids = []
        for pid in raw_ids:
            resolved_pid = self._resolve_pt_filename(pid)
            graph_path   = os.path.join(self.graph_dir,     f"{resolved_pid}.pt")
            label_path   = os.path.join(self.processed_dir, f"{resolved_pid}.pt")
            if os.path.exists(graph_path) and os.path.exists(label_path):
                valid_ids.append(pid)

        self.ids = valid_ids

        # --------------------------------------------------
        # GeneOntology — parse labels from annot file
        # --------------------------------------------------
        self.go_labels = None
        if task_name == "GeneOntology" and go_branch is not None:
            self.go_labels = _parse_go_labels(self.task_dir, go_branch)

            # filter to only proteins with labels for this branch
            valid_ids = []
            for pid in self.ids:
                resolved_pid = self._resolve_pt_filename(pid)
                parts        = resolved_pid.split("_")
                go_key       = f"{parts[0].upper()}-{parts[1]}"
                if go_key in self.go_labels:
                    valid_ids.append(pid)

            self.ids = valid_ids
            print(f"[GO-{go_branch}] {len(self.ids)} proteins in {split}")

    # ======================================================
    # Split logic
    # ======================================================

    def _resolve_split_file(self):
        splits = self.task_cfg["splits"]

        if self.task_name == "FoldClassification" and self.split == "test":
            if self.fold_test_type is None:
                raise ValueError(
                    "Specify fold_test_type: family / superfamily / fold"
                )
            filename = splits["test"][self.fold_test_type]
        else:
            filename = splits[self.split]

        return os.path.join(self.task_dir, filename)

    def _load_ids(self, filepath):
        with open(filepath, "r") as f:
            ids = [line.strip() for line in f if line.strip()]
        return ids

    def _resolve_pt_filename(self, raw_id):
        pid = raw_id.strip().split()[0]

        if self.task_name == "BindingSite":
            return pid

        if self.task_name == "GeneOntology":
            pid = pid.replace("-", "_")

        if self.task_name == "ECReaction":
            pid = pid.replace(".", "_")

        if "_" in pid:
            pdb_id, chain_id = pid.split("_")
            pid = pdb_id.lower() + "_" + chain_id
        else:
            pid = pid.lower()

        return pid

    # ======================================================
    # Dataset Interface
    # ======================================================

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid          = self.ids[idx]
        resolved_pid = self._resolve_pt_filename(pid)

        graph_path = os.path.join(self.graph_dir,     f"{resolved_pid}.pt")
        label_path = os.path.join(self.processed_dir, f"{resolved_pid}.pt")

        graph      = torch.load(graph_path)
        label_data = torch.load(label_path)

        if self.device is not None:
            graph.to_torch(self.device)

        # --------------------------------------------------
        # ESM protein embedding
        # --------------------------------------------------
        protein_X = graph.protein.X
        needs_esm = (
            protein_X is None      or
            protein_X.numel() == 0 or
            protein_X.shape[-1] != 1280
        )
        if needs_esm:
            protein_seq     = get_sequence(label_data.residues)
            X_protein_np    = get_protein_features(protein_seq, self.device)
            graph.protein.X = torch.tensor(
                X_protein_np, dtype=torch.float32
            ).to(self.device)

        # --------------------------------------------------
        # Label
        # --------------------------------------------------
        if self.task_name == "BindingSite":
            label = graph.node_y
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label)
            return {"graph": graph, "label": label}

        if self.task_name == "GeneOntology":
            parts  = resolved_pid.split("_")
            go_key = f"{parts[0].upper()}-{parts[1]}"
            if go_key not in self.go_labels:
                return self.__getitem__((idx + 1) % len(self.ids))
            label = self.go_labels[go_key]
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label)
            return {"graph": graph, "label": label}

        # --------------------------------------------------
        # All other tasks (FoldClassification, ECReaction)
        # --------------------------------------------------
        label = label_data.graph_y
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)

        return {"graph": graph, "label": label}
 
BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O"}
ELEMENTS = ["C", "N", "O", "S", "P", "H"]
ELEM_TO_IDX = {e: i for i, e in enumerate(ELEMENTS)}

class AtomicDataset(Dataset):
    def __init__(self, pdb_dir, k):
        all_files = sorted(glob(os.path.join(pdb_dir, "*.pdb")))
        self.pdb_files = all_files
        self.k = k
        print(f"Found {len(self.pdb_files)} PDB files")

    def __len__(self):
        return len(self.pdb_files)

    def __getitem__(self, idx):
        pdb_path  = self.pdb_files[idx]
        parser    = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)

        coords_list   = []
        features_list = []
        backbone_list = []

        for atom in structure.get_atoms():
            if atom.element == "H":
                continue

            coords_list.append(atom.coord)

            one_hot = np.zeros(len(ELEMENTS), dtype=np.float32)
            elem    = atom.element.strip()
            if elem in ELEM_TO_IDX:
                one_hot[ELEM_TO_IDX[elem]] = 1.0
            features_list.append(one_hot)

            backbone_list.append(float(atom.get_name() in BACKBONE_ATOM_NAMES))

        coords   = np.array(coords_list,   dtype=np.float32)
        features = np.array(features_list, dtype=np.float32)
        backbone = np.array(backbone_list, dtype=np.float32)[:, None]

        coords_t = torch.from_numpy(coords).float()
        coords_t = normalize_coords(coords_t)

        h = torch.from_numpy(
            np.concatenate([features, backbone], axis=1)
        ).float()

        edge_index = knn_graph(coords_t, k=self.k, loop=False)
        row, col   = edge_index
        dist       = torch.norm(coords_t[row] - coords_t[col], dim=1, keepdim=True)

        return h, coords_t, edge_index, dist

def collate_graphs(batch):
    h_list          = []
    x_list          = []
    edge_index_list = []
    edge_attr_list  = []
    node_offset     = 0

    for h, x, edge_index, edge_attr in batch:
        h_list.append(h)
        x_list.append(x)
        edge_index_list.append(edge_index + node_offset)
        edge_attr_list.append(edge_attr)
        node_offset += h.size(0)

    return (
        torch.cat(h_list,          dim=0),
        torch.cat(x_list,          dim=0),
        torch.cat(edge_index_list, dim=1),
        torch.cat(edge_attr_list,  dim=0),
    )

class SurfaceDataset(Dataset):

    def __init__(self, surface_dir, max_faces=None):

        all_files = sorted(
            glob(os.path.join(surface_dir, "*.ply")) +
            glob(os.path.join(surface_dir, "*.obj")) +
            glob(os.path.join(surface_dir, "*.off"))
        )

        self.surface_files = all_files
        
        self.max_faces = max_faces

        print(f"Using {len(self.surface_files)} surface meshes")

    def __len__(self):
        return len(self.surface_files)

    def __getitem__(self, idx):

        mesh = trimesh.load(self.surface_files[idx], process=False)

        if self.max_faces is not None:
            if mesh.faces.shape[0] > self.max_faces:
                mesh = mesh_simplification_quadric_decimation(
                    mesh,
                    target_faces=self.max_faces
                )

        # Build graph on CPU
        x, edge_index, face_index, edge_attr = build_surface_graph(
            mesh,
            device="cpu"
        )

        n_nodes = x.size(0)

        h = torch.ones(n_nodes, 16)  # or pass input_dim

        return h, x, edge_index, face_index, edge_attr

def collate_surface_graphs(batch):

    h_list = []
    x_list = []
    edge_index_list = []
    face_index_list = []
    edge_attr_list = []

    node_offset = 0

    for h, x, edge_index, face_index, edge_attr in batch:

        h_list.append(h)
        x_list.append(x)

        edge_index = edge_index + node_offset
        edge_index_list.append(edge_index)

        face_index = face_index + node_offset
        face_index_list.append(face_index)

        edge_attr_list.append(edge_attr)

        node_offset += h.size(0)

    h = torch.cat(h_list, dim=0)
    x = torch.cat(x_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1)
    face_index = torch.cat(face_index_list, dim=1)
    edge_attr = torch.cat(edge_attr_list, dim=0)

    return h, x, edge_index, face_index, edge_attr