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

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

class SequenceDataset(Dataset):
    """
    Generic downstream dataset loader.
    """

    def __init__(
        self,
        config_path,
        task_name,
        split="train",
        fold_test_type=None  # only for FoldClassification test
    ):
        super().__init__()

        # --------------------------------------------------
        # Load YAML config
        # --------------------------------------------------
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)

        self.data_root = full_config["data_root"]
        self.task_name = task_name
        self.task_cfg = full_config["tasks"][task_name]
        self.split = split
        self.fold_test_type = fold_test_type

        self.task_dir = os.path.join(self.data_root, task_name)
        self.processed_dir = os.path.join(
            self.task_dir,
            self.task_cfg["processed_dir"]
        )

        # --------------------------------------------------
        # Load split IDs
        # --------------------------------------------------
        split_file = self._resolve_split_file()
        self.ids = self._load_ids(split_file)
        
        raw_ids = self._load_ids(split_file)
        # --------------------------------------------------
        # Filter only existing .pt files
        # --------------------------------------------------
        valid_ids = []

        for pid in raw_ids:
            resolved_pid = self._resolve_pt_filename(pid)
            pt_path = os.path.join(self.processed_dir, f"{resolved_pid}.pt")
            if os.path.exists(pt_path):
                valid_ids.append(pid)

        self.ids = valid_ids

    # ======================================================
    # Resolve split file
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
        """
        Try multiple formatting strategies to find the correct .pt file.
        Returns full path if found.
        """

        # Clean formatting
        pid = raw_id.strip().split()[0]
        if self.task_name == "GeneOntology":
            pid = pid.replace("-", "_")
        if self.task_name == "ECReaction":
            pid = pid.replace(".", "_")
        
        # Keep chain uppercase
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

        pid = self.ids[idx]
        pid = self._resolve_pt_filename(pid)
        pt_path = os.path.join(self.processed_dir, f"{pid}.pt")

        data = torch.load(pt_path)

        # Convert 3-letter → 1-letter
        sequence = []

        for res in data.residues:
            res = res.upper()
            if res in THREE_TO_ONE:
                sequence.append(THREE_TO_ONE[res])
            else:
                sequence.append("X")

        sequence = "".join(sequence)

        # Ensure label tensor
        label = data.graph_y
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)

        data_sample = {
            "sequence": sequence,
            "label": label
        }

        return data_sample

class GraphDataset(Dataset):
    """
    Dataset for hierarchical graph downstream tasks.
    Loads graph from graph_dir
    Loads label from processed_dir
    """

    def __init__(
        self,
        config_path,
        task_name,
        split="train",
        fold_test_type=None,
        device=None
    ):
        self.device = device
        super().__init__()

        # --------------------------------------------------
        # Load YAML config
        # --------------------------------------------------
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)

        self.data_root = full_config["data_root"]
        self.task_name = task_name
        self.task_cfg = full_config["tasks"][task_name]
        self.split = split
        self.fold_test_type = fold_test_type

        self.task_dir = os.path.join(self.data_root, task_name)

        # Separate directories
        self.processed_dir = os.path.join(
            self.task_dir,
            self.task_cfg["processed_dir"]
        )

        self.graph_dir = os.path.join(
            self.task_dir,
            self.task_cfg["graph_dir"]
        )

        # --------------------------------------------------
        # Load split IDs
        # --------------------------------------------------
        split_file = self._resolve_split_file()
        raw_ids = self._load_ids(split_file)

        # --------------------------------------------------
        # Filter only samples that have BOTH graph + label
        # --------------------------------------------------
        valid_ids = []

        for pid in raw_ids:
            resolved_pid = self._resolve_pt_filename(pid)

            graph_path = os.path.join(self.graph_dir, f"{resolved_pid}.pt")
            label_path = os.path.join(self.processed_dir, f"{resolved_pid}.pt")

            if os.path.exists(graph_path) and os.path.exists(label_path):
                valid_ids.append(pid)

        self.ids = valid_ids

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

        pid = self.ids[idx]
        resolved_pid = self._resolve_pt_filename(pid)

        graph_path = os.path.join(self.graph_dir, f"{resolved_pid}.pt")
        label_path = os.path.join(self.processed_dir, f"{resolved_pid}.pt")

        # Load graph
        graph = torch.load(graph_path)
        
        # Move graph to device
        if self.device is not None:
            graph.to_torch(self.device)

        # Load label from processed file
        label_data = torch.load(label_path)

        if hasattr(label_data, "graph_y"):
            label = label_data.graph_y
        else:
            raise ValueError(f"No label found in {resolved_pid}.pt")

        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)

        return {
            "graph": graph,
            "label": label
        }

if __name__ == "__main__":
    
    # Example usage
    config_path = "/home/dvnguye2/PRL/config/data_config.yaml"
    # task_name = "GeneOntology"
    task_name = "FoldClassification"
    # task_name = "ECReaction"
    # task_name = "AntibodyDevelopability"
    split = "train"
    fold_test_type = None
    
    dataset = GraphDataset(
        config_path=config_path,
        task_name=task_name,
        split=split,
        fold_test_type=fold_test_type,
    )

    print("Dataset size:", len(dataset))