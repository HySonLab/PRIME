"""
Collect PRIME_CrossAttention level-attention weights for all tasks and draw
one combined violin figure for the paper.

Panels (2x4 grid, last slot = legend):
    (a) Fold - Family   (b) Fold - Superfamily   (c) Fold - Fold   (d) EC Reaction
    (e) GO - MF         (f) GO - BP              (g) GO - CC

Usage
-----
# evaluate every run, cache weights, then plot
python visualize_attention.py

# only re-draw the figure from cached .npy files (no GPU needed)
python visualize_attention.py --plot_only

# re-run just some panels (others are loaded from cache)
python visualize_attention.py --runs GO_MF GO_BP --overwrite
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.hierarchical_graph import *          # noqa: F401,F403
from utils.helpers import *                     # noqa: F401,F403  (get_metric, to_multihot)
from utils.helpers import load_config, build_graph_dataloaders
from models.models import PRIME_CrossAttention


# ============================================================
# Run definitions (order = panel order in the figure)
# ============================================================

RUNS = {
    "FOLD_family":      dict(task="FoldClassification", go_branch=None, split="family",      title="Fold — Family"),
    "FOLD_superfamily": dict(task="FoldClassification", go_branch=None, split="superfamily", title="Fold — Superfamily"),
    "FOLD_fold":        dict(task="FoldClassification", go_branch=None, split="fold",        title="Fold — Fold"),
    "EC":               dict(task="ECReaction",         go_branch=None, split=None,          title="EC Reaction"),
    "GO_MF":            dict(task="GeneOntology",       go_branch="MF", split=None,          title="GO — MF"),
    "GO_BP":            dict(task="GeneOntology",       go_branch="BP", split=None,          title="GO — BP"),
    "GO_CC":            dict(task="GeneOntology",       go_branch="CC", split=None,          title="GO — CC"),
}

METRIC_NAME = {
    "multilabel_classification": "Fmax",
    "node_classification":       "ROC-AUC",
    "multiclass_classification": "Accuracy",
}

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
LEVEL_NAMES = {"surface": "Surface", "atom": "Atom", "residue": "Residue",
               "sse": "SSE", "protein": "Protein"}


def cache_file(cache_dir, run, level_tag):
    """Same naming as before so old caches keep working."""
    if run["task"] == "GeneOntology":
        name = f"attn_GeneOntology_{run['go_branch']}_{level_tag}"
    else:
        name = f"attn_{run['task']}_{level_tag}"
    if run["split"] is not None:
        name += f"_{run['split']}"
    return os.path.join(cache_dir, name + ".npy")


# ============================================================
# Model loading
# ============================================================

def get_num_classes(task, task_cfg, go_branch):
    if task == "GeneOntology":
        if go_branch is None:
            raise ValueError("GeneOntology requires a go_branch (MF/BP/CC)")
        return task_cfg["num_classes"][go_branch]
    return task_cfg["num_classes"]


def load_model(task, go_branch, task_cfg, model_config, active_levels, ckpt_dir, device):
    level_tag   = "_".join(active_levels)
    num_classes = get_num_classes(task, task_cfg, go_branch)

    if task == "GeneOntology":
        ckpt_path = os.path.join(ckpt_dir, f"best_prime_ca_{task}_{go_branch}_{level_tag}_seed3.pt")
    else:
        ckpt_path = os.path.join(ckpt_dir, f"best_prime_ca_{task}_{level_tag}_seed3.pt")
    print(f"Loading checkpoint: {ckpt_path}")

    head_cfg = model_config["head"][task]
    model = PRIME_CrossAttention(
        num_classes=num_classes,
        input_dims=model_config["hierarchical"]["input_dims"],
        active_levels=active_levels,
        hidden_dim=model_config["hierarchical"]["hidden_dim"],
        encoder_layers=model_config["hierarchical"]["n_layers"],
        head_hidden_dim=head_cfg["hidden_dim"],
        head_layers=head_cfg["num_layers"],
        dropout=head_cfg["dropout"],
        task_level=task_cfg.get("task_level", "graph"),
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device).eval(), num_classes


# ============================================================
# Evaluate + collect attention in ONE pass
# ============================================================

@torch.no_grad()
def evaluate_and_collect(model, loader, task_type, num_classes, n_levels, device, desc):
    metric = get_metric(task_type, num_classes, device)
    metric.reset()
    attn_rows = []
    n_fail = 0

    for batch in tqdm(loader, desc=desc):
        for sample in batch:
            try:
                logits, attn = model(sample["graph"], return_attn=True)
            except Exception as e:  # keep going, but report it
                n_fail += 1
                if n_fail <= 3:
                    print(f"  [warn] forward failed: {e}")
                continue

            # (1, L) for graph-level tasks; (n_nodes, L) for node-level
            attn_rows.append(attn.detach().float().cpu().numpy().reshape(-1, n_levels))

            logits = logits.squeeze(0)
            if task_type == "node_classification":
                labels = sample["label"].float().to(device)
                metric.update(torch.sigmoid(logits).cpu(), labels.long().cpu())
            elif task_type == "multilabel_classification":
                y = to_multihot(sample["label"], num_classes, device)
                metric.update(torch.sigmoid(logits).unsqueeze(0), y.int().unsqueeze(0))
            else:
                label = torch.tensor(sample["label"], dtype=torch.long, device=device)
                metric.update(logits.unsqueeze(0), label.unsqueeze(0))

    if n_fail:
        print(f"  [warn] {n_fail} samples skipped")

    score = metric.compute().item()
    return score, np.concatenate(attn_rows, axis=0)


# ============================================================
# Summary table
# ============================================================

def print_summary(weights, levels):
    print("=" * 58)
    print(f"{'Level':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 58)
    for i, level in enumerate(levels):
        w = weights[:, i]
        print(f"{level:<12} {w.mean():>8.4f} {w.std():>8.4f} {w.min():>8.4f} {w.max():>8.4f}")
    print("=" * 58)


# ============================================================
# Plot
# ============================================================

def draw_violin(ax, weights, levels, title, letter, show_ylabel):
    L = len(levels)
    pos = np.arange(L)

    parts = ax.violinplot([weights[:, i] for i in range(L)], positions=pos,
                          widths=0.75, showmeans=False, showmedians=False,
                          showextrema=False)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(COLORS[i % len(COLORS)])
        body.set_edgecolor("black")
        body.set_linewidth(0.6)
        body.set_alpha(0.75)

    q1, q3 = np.percentile(weights, [25, 75], axis=0)
    means  = weights.mean(axis=0)
    ax.vlines(pos, q1, q3, color="black", linewidth=3, zorder=3)
    ax.scatter(pos, means, s=22, color="white", edgecolor="black",
               linewidth=0.9, zorder=4)

    for x, m, top in zip(pos, means, weights.max(axis=0)):
        ax.text(x, min(top + 0.03, 1.02), f"{m:.2f}",
                ha="center", va="bottom", fontsize=7.5)

    ax.axhline(1.0 / L, color="gray", linestyle="--", linewidth=1.0, zorder=1)

    ax.set_xticks(pos)
    ax.set_xticklabels([LEVEL_NAMES.get(l, l) for l in levels],
                       fontsize=8.5, rotation=30, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.text(-0.02, 1.06, f"({letter})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", ha="right", va="bottom")
    if show_ylabel:
        ax.set_ylabel("Attention weight", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)


def plot_all(weights_by_run, levels, out_path):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), sharey=True)
    axes = axes.flatten()

    for k, (key, run) in enumerate(RUNS.items()):
        ax = axes[k]
        w = weights_by_run.get(key)
        if w is None:
            ax.text(0.5, 0.5, f"missing: {key}", ha="center", va="center",
                    fontsize=8, color="red", transform=ax.transAxes)
            ax.set_title(run["title"], fontsize=10.5, fontweight="bold")
            ax.set_xticks([])
            continue
        draw_violin(ax, w, levels, run["title"], chr(ord("a") + k),
                    show_ylabel=(k % 4 == 0))

    # 8th slot -> shared legend
    leg = axes[7]
    leg.axis("off")
    L = len(levels)
    leg.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=7, label="Mean"),
        Line2D([0], [0], color="black", linewidth=3, label="Interquartile range"),
        Line2D([0], [0], color="gray", linestyle="--",
               label=f"Uniform (1/{L} = {1.0 / L:.2f})"),
    ], loc="center", fontsize=10, frameon=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        plt.savefig(f"{out_path}.{ext}", dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}.{ext}")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_config",  default="config/data_config.yaml")
    p.add_argument("--model_config", default="config/model_config.yaml")
    p.add_argument("--ckpt_dir",     default="./ckpts")
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--active_levels", nargs="+",
                   default=["surface", "atom", "residue", "sse", "protein"])
    p.add_argument("--runs", nargs="+", default=list(RUNS.keys()),
                   choices=list(RUNS.keys()),
                   help="which runs to (re)compute; others are loaded from cache")
    p.add_argument("--overwrite", action="store_true",
                   help="recompute selected runs even if a cache exists")
    p.add_argument("--plot_only", action="store_true",
                   help="skip evaluation, plot from cached .npy files only")
    p.add_argument("--output_dir", default="./plots")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    levels    = args.active_levels
    level_tag = "_".join(levels)
    device    = torch.device(args.device if torch.cuda.is_available() else "cpu")

    scores_path = os.path.join(args.output_dir, f"attn_scores_{level_tag}.json")
    scores = json.load(open(scores_path)) if os.path.exists(scores_path) else {}

    # ---------------- collect ----------------
    if not args.plot_only:
        data_config  = load_config(args.data_config)
        model_config = load_config(args.model_config)

        # group runs by (task, go_branch) so each checkpoint loads once
        todo = [k for k in args.runs
                if args.overwrite or not os.path.exists(cache_file(args.output_dir, RUNS[k], level_tag))]
        groups = {}
        for k in todo:
            groups.setdefault((RUNS[k]["task"], RUNS[k]["go_branch"]), []).append(k)

        for (task, go_branch), keys in groups.items():
            task_cfg  = data_config["tasks"][task]
            task_type = task_cfg["task_type"]
            model, num_classes = load_model(task, go_branch, task_cfg, model_config,
                                            levels, args.ckpt_dir, device)

            for key in keys:
                run = RUNS[key]
                print(f"\n=== {key}: {task} | GO={go_branch} | split={run['split']} ===")
                loader = build_graph_dataloaders(
                    args.data_config, task,
                    batch_size=args.batch_size,
                    test_only=True,
                    test_set_split=run["split"],
                    device=device,
                    go_branch=go_branch,
                )
                score, weights = evaluate_and_collect(
                    model, loader, task_type, num_classes, len(levels), device, desc=key)

                metric_name = METRIC_NAME.get(task_type, "Score")
                print(f"{metric_name}: {score:.4f}   (N = {len(weights)})")
                print_summary(weights, levels)

                np.save(cache_file(args.output_dir, run, level_tag), weights)
                scores[key] = {"metric": metric_name, "score": score, "n": int(len(weights))}
                json.dump(scores, open(scores_path, "w"), indent=2)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---------------- load all caches ----------------
    weights_by_run = {}
    for key, run in RUNS.items():
        path = cache_file(args.output_dir, run, level_tag)
        if os.path.exists(path):
            weights_by_run[key] = np.load(path)
        else:
            print(f"[warn] no cache for {key}: {path}")

    # ---------------- plot ----------------
    plot_all(weights_by_run, levels,
             os.path.join(args.output_dir, f"attn_all_tasks_{level_tag}"))

    if scores:
        print("\nTest scores:")
        for k, v in scores.items():
            print(f"  {k:<18} {v['metric']:<9} {v['score']:.4f}  (N={v['n']})")
    print("\nDone.")


if __name__ == "__main__":
    main()