import torch
from collections import Counter
import yaml
from .dataset import HierarchicalGraphDataset
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from torchmetrics.classification import MulticlassAccuracy, MultilabelF1Score, BinaryAUROC
from torchmetrics import Metric

class F1Max(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self.preds.append(preds.detach())
        self.targets.append(targets.float().detach())

    def compute(self) -> torch.Tensor:
        preds   = torch.cat(self.preds,   dim=0)
        targets = torch.cat(self.targets, dim=0)
        return self.f1_max(preds, targets)

    def f1_max(self, pred, target):
        pred = torch.softmax(pred, dim=1)  
        if target.ndim == 1:
            target = F.one_hot(
                target.long(), num_classes=pred.shape[1]
            ).float()
        order = pred.argsort(descending=True, dim=1)
        target = target.gather(1, order).int()
        precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
        recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
        is_start = torch.zeros_like(target).bool()
        is_start[:, 0] = 1
        is_start = torch.scatter(is_start, 1, order, is_start)
        all_order = pred.flatten().argsort(descending=True)
        order = (
            order
            + torch.arange(order.shape[0], device=order.device).unsqueeze(1)
            * order.shape[1]
        )
        order = order.flatten()
        inv_order = torch.zeros_like(order)
        inv_order[order] = torch.arange(order.shape[0], device=order.device)
        is_start = is_start.flatten()[all_order]
        all_order = inv_order[all_order]
        precision = precision.flatten()
        recall = recall.flatten()
        all_precision = precision[all_order] - torch.where(
            is_start, torch.zeros_like(precision), precision[all_order - 1]
        )
        all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
        all_recall = recall[all_order] - torch.where(
            is_start, torch.zeros_like(recall), recall[all_order - 1]
        )
        all_recall = all_recall.cumsum(0) / pred.shape[0]
        all_f1 = (
            2
            * all_precision
            * all_recall
            / (all_precision + all_recall + 1e-10)
        )
        return all_f1.max()

def add_noise(x, noise_std=None):
    """
    Stable noise injection for denoising AE.
    Wider noise range gives the model a real learning signal.
    """
    x_clean = x.clone()

    if noise_std is None:
        noise_std = torch.empty(1).uniform_(0.1, 0.5).item()

    noise = torch.randn_like(x_clean) * noise_std
    
    x_noisy = x_clean + noise

    return x_noisy, x_clean, noise_std

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
        return F1Max().to(device)
    elif task_type == "node_classification":
        return BinaryAUROC().to(device)
    else:
        return MulticlassAccuracy(
            num_classes=num_classes
        ).to(device)

def compute_class_weights(loader, num_classes, task_level="graph"):
    if task_level == "node":
        pos_count = 0
        neg_count = 0

        for batch in loader:
            for sample in batch:
                labels     = sample["label"]
                pos_count += labels.sum().item()
                neg_count += (labels == 0).sum().item()

        total      = pos_count + neg_count
        weight_neg = total / (2 * neg_count)
        weight_pos = total / (2 * pos_count)

        print(f"Binding site: {pos_count} pos ({pos_count/total:.3f}), "
              f"{neg_count} neg ({neg_count/total:.3f})")

        return torch.tensor([weight_neg, weight_pos], dtype=torch.float32)

    else:
        labels = []

        for batch in loader:
            for sample in batch:
                labels.append(int(sample["label"]))

        labels          = np.array(labels)
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

def to_multihot(y, num_classes, device=None):
    """
    Convert a tensor/list of class indices to a binary multi-hot vector.
    
    Args:
        y:           tensor or list of local class indices
        num_classes: total number of classes
        device:      target device
    
    Returns:
        binary vector of shape (num_classes,)
    """
    if isinstance(y, torch.Tensor):
        y = y.long().tolist()
    elif isinstance(y, (int, float)):
        y = [int(y)]

    # validate range
    y = [cls for cls in y if 0 <= cls < num_classes]

    vec = torch.zeros(num_classes)
    if len(y) > 0:
        vec[y] = 1.0

    return vec.to(device) if device is not None else vec


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_graph_dataloaders(
    config_path,
    task_name,
    batch_size=4,
    num_workers=0,
    test_only=False,
    test_set_split=None,
    device=None,
    go_branch=None,        
):

    if test_only:

        test_dataset = HierarchicalGraphDataset(
            config_path=config_path,
            task_name=task_name,
            split="test",
            fold_test_type=test_set_split,
            device=device,
            go_branch=go_branch,           
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: x
        )

        return test_loader

    train_dataset = HierarchicalGraphDataset(
        config_path=config_path,
        task_name=task_name,
        split="train",
        device=device,
        go_branch=go_branch,               
    )

    val_dataset = HierarchicalGraphDataset(
        config_path=config_path,
        task_name=task_name,
        split="val",
        device=device,
        go_branch=go_branch,               
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