import torch
from collections import Counter
import yaml
from .dataset import SequenceDataset, GraphDataset
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from torchmetrics.classification import MulticlassAccuracy, MultilabelF1Score

def add_noise(x, noise_std=None, clamp_noise=True):
    """
    Stable noise injection for denoising AE.
    """

    x_clean = x

    # Better noise schedule
    if noise_std is None:
        noise_std = torch.empty(1).uniform_(0.005, 0.03).item()

    noise = torch.randn_like(x_clean) * noise_std

    if clamp_noise:
        noise = torch.clamp(noise, -0.1, 0.1)

    x_noisy = x_clean + noise

    return x_noisy, x_clean

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

def get_metric(task_type, num_classes, device):
    if task_type == "multilabel_classification":
        return MultilabelF1Score(
            num_labels=num_classes,
            average="macro"
        ).to(device)
    else:
        return MulticlassAccuracy(
            num_classes=num_classes
        ).to(device)

def compute_class_weights(loader, num_classes):
    labels = []

    for batch in loader:
        for sample in batch:
            labels.append(int(sample["label"]))

    labels = np.array(labels)
    present_classes = np.unique(labels)

    weights = np.ones(num_classes, dtype=np.float32)

    present_weights = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=labels
    )

    weights[present_classes] = present_weights
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)

def compute_pos_weight(loader, num_classes):

    pos_counts = torch.zeros(num_classes)
    total_samples = 0

    for batch in loader:
        for sample in batch:
            y = sample["label"]
            for cls in y:
                pos_counts[cls] += 1
            total_samples += 1

    neg_counts = total_samples - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-6)

    return pos_weight

def filter_go_labels(y, num_classes):

    offsets = {
        489: (0, 489),
        1943: (489, 489 + 1943),
        320: (489 + 1943, 489 + 1943 + 320)
    }

    if num_classes not in offsets:
        raise ValueError("Unknown GO num_classes")

    start, end = offsets[num_classes]

    return [
        cls - start
        for cls in y
        if start <= cls < end
    ]

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_sequence_dataloaders(config_path, task_name, batch_size=4, num_workers=0, test_only=False, test_set_split=None):

    if test_only:
        if test_set_split is None:
            test_dataset = SequenceDataset(
                config_path=config_path,
                task_name=task_name,
                split="test",
            )
        else:
            test_dataset = SequenceDataset(
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
        
    train_dataset = SequenceDataset(
        config_path=config_path,
        task_name=task_name,
        split="train",
    )

    val_dataset = SequenceDataset(
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

def build_graph_dataloaders(
    config_path,
    task_name,
    batch_size=4,
    num_workers=0,
    test_only=False,
    test_set_split=None,
    device=None
):

    if test_only:

        test_dataset = GraphDataset(
            config_path=config_path,
            task_name=task_name,
            split="test",
            fold_test_type=test_set_split,
            device=device
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: x
        )

        return test_loader

    train_dataset = GraphDataset(
        config_path=config_path,
        task_name=task_name,
        split="train",
        device=device
    )

    val_dataset = GraphDataset(
        config_path=config_path,
        task_name=task_name,
        split="val",
        device=device
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