import os
from sympy import sequence, test
import yaml
import torch
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import os
import yaml
import torch

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

class ProteinDataset(Dataset):
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

def build_dataloaders(config_path, task_name, batch_size=4, num_workers=0, test_only=False, test_set_split=None):

    if test_only:
        if test_set_split is None:
            test_dataset = ProteinDataset(
                config_path=config_path,
                task_name=task_name,
                split="test",
            )
        else:
            test_dataset = ProteinDataset(
                config_path=config_path,
                task_name=task_name,
                split="test",
                fold_test_type=test_set_split
            )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: x
        )
        return test_loader
        
    train_dataset = ProteinDataset(
        config_path=config_path,
        task_name=task_name,
        split="train",
    )

    val_dataset = ProteinDataset(
        config_path=config_path,
        task_name=task_name,
        split="val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda x: x
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    
    # Example usage
    config_path = "/home/dvnguye2/PRL/config/data_config.yaml"
    # task_name = "GeneOntology"
    task_name = "FoldClassification"
    # task_name = "ECReaction"
    # task_name = "AntibodyDevelopability"
    split = "train"
    fold_test_type = None
    
    dataset = ProteinDataset(
        config_path=config_path,
        task_name=task_name,
        split=split,
        fold_test_type=fold_test_type,
        collate_fn=lambda x: x
    )

    print("Dataset size:", len(dataset))

    sample = dataset[0]
    
    print(sample)
    print("Sample ID:", sample.id)
    print("Label:", sample.graph_y)
    print("Coords shape:", sample.coords.shape)

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))

    print("\nBatch:")
    print(batch)
    print("Batch labels shape:", batch.graph_y.shape)
    
    