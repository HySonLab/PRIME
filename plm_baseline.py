"""
PLM Baseline Comparison for Fold Classification.
Extracts frozen embeddings from different PLMs and trains
the same MLP head used in PRIME under identical settings.

Supported PLMs:
  - ESM-2 650M   (baseline, already in PRIME)
  - ESM-C 300M   (EvolutionaryScale/esmc-300m-2024-12)
  - ProtT5       (Rostlab/prot_t5_xl_half_uniref50-enc)
  - SaProt 650M  (westlake-repl/SaProt_650M_AF2_HF)
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import yaml
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.helpers import load_config
from models.classification_head import MLP_Head


# ======================================================
# PLM Embedding Extractors
# ======================================================

class ESM2Embedder:
    """ESM-2 650M — CLS token embedding (1280 dim)"""

    def __init__(self, device):
        from esm import pretrained
        self.model, self.alphabet = pretrained.load_model_and_alphabet(
            "esm2_t33_650M_UR50D"
        )
        self.model    = self.model.eval().to(device)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.device   = device
        self.dim      = 1280

    @torch.no_grad()
    def embed(self, sequence: str) -> torch.Tensor:
        data         = [("protein", sequence)]
        _, _, tokens = self.batch_converter(data)
        tokens       = tokens.to(self.device)
        results      = self.model(tokens, repr_layers=[33], return_contacts=False)
        cls          = results["representations"][33][:, 0, :]  # (1, 1280)
        return cls.squeeze(0).cpu()


class ESMCEmbedder:
    """ESM-C 600M — mean pooled embedding via HuggingFace (biohub)"""

    def __init__(self, device):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            "biohub/ESMC-600M",
            trust_remote_code=True
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            "biohub/ESMC-600M",
            trust_remote_code=True 
        ).to(device).eval()
        self.device = device
        self.dim = self.model.config.d_model

    @torch.no_grad()
    def embed(self, sequence: str) -> torch.Tensor:
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        output = self.model(**inputs, output_hidden_states=True)

        hidden = output.hidden_states[-1]            # (1, L+2, dim)
        emb    = hidden[0, 1:-1, :].mean(dim=0)     # (dim,)
        return emb.float().cpu()

class ProtT5Embedder:
    """ProtT5-XL encoder-only half precision — mean pooled (1024 dim)"""

    def __init__(self, device):
        from transformers import T5EncoderModel, T5Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(
            "Rostlab/prot_t5_xl_half_uniref50-enc",
            do_lower_case=False,
            legacy=True
        )
        self.model     = T5EncoderModel.from_pretrained(
            "Rostlab/prot_t5_xl_half_uniref50-enc"
        ).to(device)
        self.model.eval()
        self.device    = device
        self.dim       = 1024

    @torch.no_grad()
    def embed(self, sequence: str) -> torch.Tensor:
        # ✅ ProtT5 requires space-separated amino acids
        seq_spaced = " ".join(list(sequence))
        ids        = self.tokenizer(
            seq_spaced,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=1024
        ).to(self.device)
        output = self.model(**ids)
        # mean pool over sequence tokens (exclude EOS)
        emb    = output.last_hidden_state[0, :-1, :].mean(dim=0)  # (1024,)
        return emb.float().cpu()

class SaProtEmbedder:
    """SaProt 650M — structure-aware, mean pooled (1280 dim)"""

    def __init__(self, device, foldseek_bin="./bin/foldseek"):
        from transformers import EsmTokenizer, EsmForMaskedLM

        # ✅ correct repo ID
        self.tokenizer = EsmTokenizer.from_pretrained(
            "westlake-repl/SaProt_650M_AF2"
        )
        self.model = EsmForMaskedLM.from_pretrained(
            "westlake-repl/SaProt_650M_AF2"
        ).to(device)
        self.model.eval()
        self.device       = device
        self.foldseek_bin = foldseek_bin
        self.dim          = 1280

    @torch.no_grad()
    def embed(self, sequence: str, seq_3di: str = None) -> torch.Tensor:
        """
        sequence: amino acid sequence
        seq_3di:  3Di structural sequence (same length)
                  if None → use '#' as placeholder (no structure)
        """
        if seq_3di is not None and len(seq_3di) == len(sequence):
            # ✅ interleave aa + 3Di: "Mv" "Ep" etc.
            sa_seq = "".join([aa + di for aa, di in zip(sequence, seq_3di)])
        else:
            # ✅ sequence-only mode — '#' means unknown structure
            sa_seq = "".join([aa + "#" for aa in sequence])

        ids    = self.tokenizer(
            sa_seq,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        output = self.model(**ids, output_hidden_states=True)

        # ✅ mean pool over sequence tokens (exclude CLS + EOS)
        hidden = output.hidden_states[-1]           # (1, L+2, 1280)
        emb    = hidden[0, 1:-1, :].mean(dim=0)    # (1280,)
        return emb.float().cpu()

# ======================================================
# Embedding Cache Dataset
# ======================================================

class EmbeddingDataset(Dataset):
    """
    Pre-computes and caches PLM embeddings for all proteins.
    Loads from cache on subsequent runs.
    """

    def __init__(
    self,
    processed_dir: str,
    split_file:    str,
    embedder,
    cache_dir:     str,
    plm_name:      str,
    ):
        self.processed_dir = processed_dir
        self.embedder      = embedder
        self.cache_dir     = os.path.join(cache_dir, plm_name)
        os.makedirs(self.cache_dir, exist_ok=True)

        # load split IDs
        with open(split_file) as f:
            raw_ids = [line.strip() for line in f if line.strip()]

        # filter to existing processed files using normalized filename
        valid_ids = []
        for pid in raw_ids:
            resolved = self._resolve_pt_filename(pid)
            pt_path  = os.path.join(processed_dir, f"{resolved}.pt")
            if os.path.exists(pt_path):
                valid_ids.append(pid)   # store raw ID, resolve on access

        self.ids = valid_ids
        print(f"  {len(self.ids)} proteins in split")

    def _resolve_pt_filename(self, raw_id: str) -> str:
        """Same normalization as HierarchicalGraphDataset."""
        pid = raw_id.strip().split()[0]

        # FoldClassification uses SCOP IDs like d1ecda_
        if "_" in pid:
            pdb_id, chain_id = pid.split("_")
            pid = pdb_id.lower() + "_" + chain_id
        else:
            pid = pid.lower()

        return pid

    def _get_cache_path(self, pid: str) -> str:
        resolved = self._resolve_pt_filename(pid)
        return os.path.join(self.cache_dir, f"{resolved}.pt")

    def _embed(self, pid: str) -> torch.Tensor:
        cache_path = self._get_cache_path(pid)
        if os.path.exists(cache_path):
            return torch.load(cache_path, map_location="cpu")

        resolved     = self._resolve_pt_filename(pid)
        pt_path      = os.path.join(self.processed_dir, f"{resolved}.pt")
        process_data = torch.load(pt_path, map_location="cpu")

        from utils.dataset import get_sequence
        sequence = get_sequence(process_data.residues)

        emb = self.embedder.embed(sequence)
        torch.save(emb, cache_path)
        return emb
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid          = self.ids[idx]
        resolved     = self._resolve_pt_filename(pid)
        pt_path      = os.path.join(self.processed_dir, f"{resolved}.pt")
        process_data = torch.load(pt_path, map_location="cpu")

        emb   = self._embed(pid)
        label = process_data.graph_y

        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)

        return {"emb": emb, "label": label, "id": pid}
    

def precompute_embeddings(dataset: EmbeddingDataset):
    """Pre-compute and cache all embeddings."""
    missing = [
        pid for pid in dataset.ids
        if not os.path.exists(dataset._get_cache_path(pid))
    ]
    if not missing:
        print("  All embeddings cached.")
        return

    print(f"  Computing {len(missing)} embeddings...")
    for pid in tqdm(missing, desc="Embedding"):
        dataset._embed(pid)


# ======================================================
# Train / Evaluate
# ======================================================

def collate_fn(batch):
    embs   = torch.stack([b["emb"]   for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    return embs, labels


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for embs, labels in tqdm(loader, desc="Train", leave=False):
        embs   = embs.to(device)
        labels = labels.to(device)

        logits = model(embs)
        loss   = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += (logits.argmax(dim=-1) == labels).sum().item()
        total      += len(labels)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for embs, labels in tqdm(loader, desc="Eval", leave=False):
        embs   = embs.to(device)
        labels = labels.to(device)

        logits = model(embs)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        correct    += (logits.argmax(dim=-1) == labels).sum().item()
        total      += len(labels)

    return total_loss / len(loader), correct / total


# ======================================================
# Main
# ======================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config",    type=str, required=True)
    parser.add_argument("--plm",            type=str, required=True,
                        choices=["esm2", "esmc", "prott5", "saprot"])
    parser.add_argument("--cache_dir",      type=str, default="./plm_cache")
    parser.add_argument("--batch_size",     type=int, default=64)
    parser.add_argument("--epochs",         type=int, default=100)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--hidden_dim",     type=int, default=512)
    parser.add_argument("--num_layers",     type=int, default=3)
    parser.add_argument("--dropout",        type=float, default=0.3)
    parser.add_argument("--seed",           type=int, default=1)
    parser.add_argument("--foldseek_bin",   type=str, default="./bin/foldseek")

    args   = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("./ckpts", exist_ok=True)
    os.makedirs("./logs",  exist_ok=True)

    log_path = f"./logs/plm_{args.plm}_FoldClassification_seed{args.seed}.txt"
    ckpt_path = f"./ckpts/best_plm_{args.plm}_FoldClassification_seed{args.seed}.pt"

    def log(msg: str):
        """Print and write to log file simultaneously."""
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")
            f.flush()

    # clear log file
    open(log_path, "w").close()

    # --------------------------------------------------
    # Load config
    # --------------------------------------------------
    data_config   = load_config(args.data_config)
    task_cfg      = data_config["tasks"]["FoldClassification"]
    task_dir      = os.path.join(data_config["data_root"], "FoldClassification")
    processed_dir = os.path.join(task_dir, task_cfg["processed_dir"])
    num_classes   = task_cfg["num_classes"]

    train_split = os.path.join(task_dir, task_cfg["splits"]["train"])
    val_split   = os.path.join(task_dir, task_cfg["splits"]["val"])

    # --------------------------------------------------
    # Load PLM
    # --------------------------------------------------
    log(f"\nLoading PLM: {args.plm}")
    if args.plm == "esm2":
        embedder = ESM2Embedder(device)
    elif args.plm == "esmc":
        embedder = ESMCEmbedder(device)
    elif args.plm == "prott5":
        embedder = ProtT5Embedder(device)
    elif args.plm == "saprot":
        embedder = SaProtEmbedder(device, foldseek_bin=args.foldseek_bin)

    log(f"Embedding dim: {embedder.dim}")

    # --------------------------------------------------
    # Build datasets
    # --------------------------------------------------
    log("\nBuilding datasets...")

    train_dataset = EmbeddingDataset(
        processed_dir=processed_dir,
        split_file=train_split,
        embedder=embedder,
        cache_dir=args.cache_dir,
        plm_name=args.plm,
    )
    val_dataset = EmbeddingDataset(
        processed_dir=processed_dir,
        split_file=val_split,
        embedder=embedder,
        cache_dir=args.cache_dir,
        plm_name=args.plm,
    )
    test_datasets = {
        split_name: EmbeddingDataset(
            processed_dir=processed_dir,
            split_file=os.path.join(task_dir, task_cfg["splits"]["test"][split_name]),
            embedder=embedder,
            cache_dir=args.cache_dir,
            plm_name=args.plm,
        )
        for split_name in ["family", "superfamily", "fold"]
    }

    log("Pre-computing embeddings...")
    precompute_embeddings(train_dataset)
    precompute_embeddings(val_dataset)
    for split_name, ds in test_datasets.items():
        precompute_embeddings(ds)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True,  collate_fn=collate_fn, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=4
    )
    test_loaders = {
        split_name: DataLoader(
            ds, batch_size=args.batch_size,
            shuffle=False, collate_fn=collate_fn, num_workers=4
        )
        for split_name, ds in test_datasets.items()
    }

    # ======================================================
    # Model — same head as PRIME
    # ======================================================

    model = MLP_Head(
        in_dim=embedder.dim,
        out_dim=num_classes,
        hidden_dims=[args.hidden_dim] * args.num_layers,
        activations=["gelu"] * args.num_layers + ["identity"],
        dropout=args.dropout,
        skip=True
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    warmup  = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=3)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.6, patience=5
    )
    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    log(f"\n{'='*50}")
    log(f"PLM:         {args.plm}")
    log(f"Num classes: {num_classes}")
    log(f"Emb dim:     {embedder.dim}")
    log(f"Seed:        {args.seed}")
    log(f"Log:         {log_path}")
    log(f"{'='*50}\n")

    best_val_acc   = -float("inf")
    patience_count = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        if epoch < 3:
            warmup.step()
        else:
            plateau.step(val_acc)

        lr  = optimizer.param_groups[0]["lr"]
        msg = (f"Epoch {epoch+1:03d} | "
               f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
               f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
               f"LR: {lr:.2e}")
        log(msg)

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            torch.save(model.state_dict(), ckpt_path)
            log("  ✓ Best model saved")
        else:
            patience_count += 1
            if patience_count >= 20:
                log("Early stopping.")
                break

    # --------------------------------------------------
    # Test on all 3 splits
    # --------------------------------------------------
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    log(f"\n{'='*50}")
    log(f"RESULTS — PLM: {args.plm} | Seed: {args.seed}")
    log(f"{'='*50}")

    results = {}
    for split_name, loader in test_loaders.items():
        _, acc = evaluate(model, loader, criterion, device)
        results[split_name] = acc * 100
        log(f"  {split_name:<15}: {acc*100:.2f}%")

    log(f"\nFamily: {results['family']:.2f} | "
        f"Superfamily: {results['superfamily']:.2f} | "
        f"Fold: {results['fold']:.2f}")
    log(f"{'='*50}")
    log(f"Log saved to: {log_path}")