import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import argparse
import yaml
from tqdm import tqdm
from utils.hierarchical_graph import *
from utils.helpers import *

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.models import PRIME_CrossAttention
from utils.helpers import load_config, build_graph_dataloaders

# ============================================================
# Collect attention weights from val set
# ============================================================

def collect_attention_weights(model, loader):
    """
    Run model on loader and collect per-protein attention weights.
    Returns: (N_proteins, L) numpy array
    """
    model.eval()
    all_weights = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Collecting attention"):
            for sample in batch:
                try:
                    _, attn = model(sample["graph"], return_attn=True)  # (1, L)
                    all_weights.append(attn.cpu().squeeze(0).numpy())   # (L,)
                except Exception as e:
                    continue

    return np.stack(all_weights, axis=0)  # (N, L)

# ============================================================
# Plot 1 — Mean attention bar chart
# ============================================================

def plot_mean_attention(weights, levels, save_path, task_name):
    mean_w = weights.mean(axis=0)
    std_w  = weights.std(axis=0)
    uniform = 1.0 / len(levels)

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    bars   = ax.bar(
        levels, mean_w,
        yerr=std_w,
        color=colors[:len(levels)],
        capsize=6,
        edgecolor="black",
        linewidth=0.8,
        width=0.6
    )

    ax.axhline(
        uniform, color="gray", linestyle="--",
        linewidth=1.5, label=f"Uniform ({uniform:.2f})"
    )

    # annotate values on bars
    for bar, val in zip(bars, mean_w):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std_w[list(mean_w).index(val)] + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    ax.set_title(
        f"Mean Attention Weight per Structural Level\n({task_name})",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.set_xlabel("Hierarchical Level", fontsize=11)
    ax.set_ylabel("Attention Weight", fontsize=11)
    ax.set_ylim(0, max(mean_w + std_w) * 1.25)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

# ============================================================
# Plot
# ============================================================

def plot_combined_attention(weights_dict, levels, save_path, task_name="FoldClassification"):
    """
    weights_dict: {"family": np.array, "superfamily": np.array, "fold": np.array}
    Produces a 2x3 figure: top row = bar charts, bottom row = violin plots
    """
    splits = ["family", "superfamily", "fold"]
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    uniform = 1.0 / len(levels)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Attention Weight Analysis — {task_name}",
        fontsize=15, fontweight="bold", y=1.01
    )

    for col, split in enumerate(splits):
        weights = weights_dict[split]
        mean_w  = weights.mean(axis=0)
        std_w   = weights.std(axis=0)

        # ── Top row: bar chart ──────────────────────────────────
        ax = axes[0, col]
        bars = ax.bar(
            levels, mean_w,
            yerr=std_w,
            color=colors[:len(levels)],
            capsize=5,
            edgecolor="black",
            linewidth=0.7,
            width=0.6
        )
        ax.axhline(uniform, color="gray", linestyle="--", linewidth=1.2,
                   label=f"Uniform ({uniform:.2f})")

        for bar, val, std in zip(bars, mean_w, std_w):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold"
            )

        ax.set_title(f"{split.capitalize()} Split", fontsize=12, fontweight="bold")
        ax.set_ylim(0, max(mean_w + std_w) * 1.3)
        ax.set_xlabel("Hierarchical Level", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col == 0:
            ax.set_ylabel("Attention Weight", fontsize=10)
            ax.legend(fontsize=8)
        else:
            ax.set_ylabel("")

        # ── Bottom row: violin plot ─────────────────────────────
        ax = axes[1, col]
        parts = ax.violinplot(
            [weights[:, i] for i in range(len(levels))],
            positions=range(len(levels)),
            showmeans=True,
            showmedians=True,
            widths=0.6
        )
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_alpha(0.7)
        parts["cmeans"].set_color("black")
        parts["cmedians"].set_color("red")

        ax.axhline(uniform, color="gray", linestyle="--", linewidth=1.2)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels, fontsize=10)
        ax.set_xlabel("Hierarchical Level", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col == 0:
            ax.set_ylabel("Attention Weight", fontsize=10)
            # shared legend for violin only on first column
            mean_patch   = mpatches.Patch(color="black", label="Mean")
            median_patch = mpatches.Patch(color="red",   label="Median")
            uniform_patch = mpatches.Patch(color="gray", label=f"Uniform ({uniform:.2f})")
            ax.legend(handles=[mean_patch, median_patch, uniform_patch], fontsize=8)
        else:
            ax.set_ylabel("")

    # row labels on the left
    axes[0, 0].annotate("Mean ± Std", xy=(-0.25, 0.5), xycoords="axes fraction",
                         fontsize=11, fontweight="bold", rotation=90, va="center")
    axes[1, 0].annotate("Distribution", xy=(-0.25, 0.5), xycoords="axes fraction",
                         fontsize=11, fontweight="bold", rotation=90, va="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

def plot_single_task_attention(weights, levels, save_path, task_name):
    """
    Single figure with bar chart and violin plot side by side.
    Used for non-FoldClassification tasks.
    """
    mean_w  = weights.mean(axis=0)
    std_w   = weights.std(axis=0)
    uniform = 1.0 / len(levels)
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Attention Weight Analysis — {task_name}",
        fontsize=14, fontweight="bold"
    )

    # ── Left: bar chart ────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(
        levels, mean_w,
        yerr=std_w,
        color=colors[:len(levels)],
        capsize=6,
        edgecolor="black",
        linewidth=0.8,
        width=0.6
    )
    ax.axhline(uniform, color="gray", linestyle="--",
               linewidth=1.5, label=f"Uniform ({uniform:.2f})")

    for bar, val, std in zip(bars, mean_w, std_w):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    ax.set_title("Mean ± Std", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hierarchical Level", fontsize=11)
    ax.set_ylabel("Attention Weight",   fontsize=11)
    ax.set_ylim(0, max(mean_w + std_w) * 1.3)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Right: violin plot ─────────────────────────────────
    ax = axes[1]
    parts = ax.violinplot(
        [weights[:, i] for i in range(len(levels))],
        positions=range(len(levels)),
        showmeans=True,
        showmedians=True,
        widths=0.6
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.7)

    parts["cmeans"].set_color("black")
    parts["cmedians"].set_color("red")

    ax.axhline(uniform, color="gray", linestyle="--", linewidth=1.5)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, fontsize=11)
    ax.set_title("Distribution",        fontsize=12, fontweight="bold")
    ax.set_xlabel("Hierarchical Level", fontsize=11)
    ax.set_ylabel("Attention Weight",   fontsize=11)

    mean_patch    = mpatches.Patch(color="black", label="Mean")
    median_patch  = mpatches.Patch(color="red",   label="Median")
    uniform_patch = mpatches.Patch(color="gray",  label=f"Uniform ({uniform:.2f})")
    ax.legend(handles=[mean_patch, median_patch, uniform_patch], fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
    
# ============================================================
# Summary table
# ============================================================

def print_summary(weights, levels):
    print("\n" + "=" * 58)
    print(f"{'Level':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 58)
    for i, level in enumerate(levels):
        print(f"{level:<12} {weights[:, i].mean():>8.4f} "
              f"{weights[:, i].std():>8.4f} "
              f"{weights[:, i].min():>8.4f} "
              f"{weights[:, i].max():>8.4f}")
    print("=" * 58)

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config",  type=str, default="config/data_config.yaml")
    parser.add_argument("--model_config", type=str, default="config/model_config.yaml")
    parser.add_argument("--batch_size",   type=int, default=4)
    parser.add_argument("--device",       type=str, default="cuda")
    parser.add_argument("--task",         type=str, default="FoldClassification",
                        help="FoldClassification | ECReaction | GeneOntology | BindingSite")
    parser.add_argument("--go_branch",    type=str, default=None,
                        help="MF | BP | CC (required for GeneOntology)")
    parser.add_argument(
        "--active_levels",
        nargs="+",
        default=["surface", "atom", "residue", "sse", "protein"]
    )
    parser.add_argument("--output_dir", type=str, default="./plots")

    args   = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # --------------------------------------------------
    # Config
    # --------------------------------------------------
    TASK = args.task

    data_config  = load_config(args.data_config)
    model_config = load_config(args.model_config)
    task_cfg     = data_config["tasks"][TASK]
    task_type    = task_cfg["task_type"]
    task_level   = task_cfg.get("task_level", "graph")

    if TASK == "GeneOntology":
        if args.go_branch is None:
            raise ValueError("GeneOntology requires --go_branch (MF/BP/CC)")
        num_classes = task_cfg["num_classes"][args.go_branch]
    elif task_type == "node_classification":
        num_classes = task_cfg["num_classes"]
    else:
        num_classes = task_cfg["num_classes"]

    # --------------------------------------------------
    # Test splits per task
    # --------------------------------------------------
    if TASK == "FoldClassification":
        TEST_SPLITS = ["family", "superfamily", "fold"]
    else:
        TEST_SPLITS = [None]   # single test set for all other tasks

    # --------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------
    level_tag = "_".join(args.active_levels)

    if TASK == "GeneOntology":
        ckpt_path = f"./ckpts/best_prime_ca_{TASK}_{args.go_branch}_{level_tag}.pt"
    else:
        ckpt_path = f"./ckpts/best_prime_ca_{TASK}_{level_tag}.pt"

    print(f"Loading checkpoint: {ckpt_path}")

    # --------------------------------------------------
    # Build model
    # --------------------------------------------------
    if TASK == "GeneOntology":
        head_key = TASK
    else:
        head_key = TASK

    model = PRIME_CrossAttention(
        num_classes=num_classes,
        input_dims=model_config["hierarchical"]["input_dims"],
        active_levels=args.active_levels,
        hidden_dim=model_config["hierarchical"]["hidden_dim"],
        encoder_layers=model_config["hierarchical"]["n_layers"],
        head_hidden_dim=model_config["head"][head_key]["hidden_dim"],
        head_layers=model_config["head"][head_key]["num_layers"],
        dropout=model_config["head"][head_key]["dropout"],
        task_level=task_level,
    )

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    # --------------------------------------------------
    # Evaluate + visualize per test split
    # --------------------------------------------------
    print("\n" + "=" * 50)
    print(f"Task:      {TASK}")
    print(f"GO Branch: {args.go_branch}")
    print(f"Model:     PRIME_CrossAttention")
    print("=" * 50)

    weights_dict = {}

    for split in TEST_SPLITS:

        split_tag = split if split is not None else "test"
        print(f"\n--- Test Split: {split_tag} ---")

        test_loader = build_graph_dataloaders(
            args.data_config,
            TASK,
            batch_size=args.batch_size,
            test_only=True,
            test_set_split=split,
            device=device,
            go_branch=args.go_branch,     # pass go_branch
        )

        # evaluate
        metric = get_metric(task_type, num_classes, device)
        metric.reset()

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating {split_tag}"):
                for sample in batch:
                    logits, _ = model(sample["graph"], return_attn=True)
                    logits     = logits.squeeze(0)

                    if task_type == "node_classification":
                        labels = sample["label"].float().to(device)
                        probs  = torch.sigmoid(logits).cpu()
                        metric.update(probs, labels.long().cpu())

                    elif task_type == "multilabel_classification":
                        y = to_multihot(sample["label"], num_classes, device)
                        metric.update(
                            torch.sigmoid(logits).unsqueeze(0),
                            y.int().unsqueeze(0)
                        )

                    else:
                        label = torch.tensor(
                            sample["label"], dtype=torch.long, device=device
                        )
                        metric.update(logits.unsqueeze(0), label.unsqueeze(0))

        score = metric.compute().item()

        metric_name = {
            "multilabel_classification": "Fmax",
            "node_classification":       "ROC-AUC",
            "multiclass_classification": "Accuracy",
        }.get(task_type, "Score")

        print(f"{metric_name} ({split_tag}): {score:.4f}")

        # collect attention weights
        weights_dict[split_tag] = collect_attention_weights(model, test_loader)
        print_summary(weights_dict[split_tag], args.active_levels)

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    tag = f"{TASK}_{args.go_branch}_{level_tag}" if TASK == "GeneOntology" \
          else f"{TASK}_{level_tag}"

    if TASK == "FoldClassification":
        # combined plot for 3 splits
        plot_combined_attention(
            weights_dict,
            levels=args.active_levels,
            save_path=os.path.join(args.output_dir, f"attn_combined_{tag}.png"),
            task_name=TASK
        )
    else:
        # single plot for other tasks
        plot_single_task_attention(
            weights_dict["test"],
            levels=args.active_levels,
            save_path=os.path.join(args.output_dir, f"attn_{tag}.png"),
            task_name=TASK
        )

    print("\nDone.")