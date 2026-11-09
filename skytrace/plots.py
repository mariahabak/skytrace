"""Figures for the data sheet and the report.

Authors: Maria Habak, [Partner name]
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CLASS_COLOURS = {"BIRD": "#d08c2c", "DRONE": "#3a6ea5"}


def data_overview(table, sets: dict, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _class_balance(sets, out_dir / "class_balance.png")
    _box_sizes(sets, out_dir / "box_sizes.png")
    _distance_bins(table, out_dir / "distance_bins.png")
    _sample_clips(sets["train"], out_dir / "sample_clips.png")


def _class_balance(sets, path):
    parts = list(sets)
    birds = [int((s.labels == 0).sum()) for s in sets.values()]
    drones = [int((s.labels == 1).sum()) for s in sets.values()]
    x = np.arange(len(parts))
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(x - 0.2, birds, 0.4, label="bird", color=CLASS_COLOURS["BIRD"])
    ax.bar(x + 0.2, drones, 0.4, label="drone", color=CLASS_COLOURS["DRONE"])
    ax.set_xticks(x, parts)
    ax.set_ylabel("clips")
    ax.set_title("Clips per partition")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _box_sizes(sets, path):
    sides = {"BIRD": [], "DRONE": []}
    for s in sets.values():
        for side, label in zip(s.box_side.tolist(), s.labels.tolist()):
            sides["BIRD" if label == 0 else "DRONE"].append(side)
    bins = np.logspace(np.log10(2), np.log10(200), 30)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for cls, values in sides.items():
        ax.hist(values, bins=bins, alpha=0.6, label=cls.lower(),
                color=CLASS_COLOURS[cls])
    ax.set_xscale("log")
    ax.set_xlabel("longest box side (pixels, log scale)")
    ax.set_ylabel("clips")
    ax.set_title("Object size per class")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _distance_bins(table, path):
    order = ["CLOSE", "MEDIUM", "DISTANT"]
    counts = table.groupby(["cls", "distance"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=order, fill_value=0)
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for i, cls in enumerate(counts.index):
        ax.bar(x + (i - 0.5) * 0.4, counts.loc[cls], 0.4, label=cls.lower(),
               color=CLASS_COLOURS[cls])
    ax.set_xticks(x, [o.lower() for o in order])
    ax.set_ylabel("videos")
    ax.set_title("Videos per distance bin (dataset labels)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _sample_clips(clip_set, path, per_class: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = clip_set.labels.numpy()
    clips = clip_set.clips.numpy()
    length = clips.shape[1]
    rows = []
    for label in (0, 1):
        candidates = np.flatnonzero(labels == label)
        rows += [(label, i) for i in rng.choice(candidates, per_class, replace=False)]
    fig, axes = plt.subplots(len(rows), length, figsize=(length * 0.9, len(rows) * 1.0))
    for r, (label, i) in enumerate(rows):
        for t in range(length):
            ax = axes[r, t]
            ax.imshow(clips[i, t], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel("bird" if label == 0 else "drone", fontsize=8)
    fig.suptitle("Training clips (frames left to right)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def loss_curves(history: list, path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend(frameon=False)
    axes[1].plot(epochs, [h["val_macro_f1"] for h in history], color="#3a6ea5")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation macro-F1")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def reliability_diagram(confidence, correct, path, bins: int = 10) -> None:
    edges = np.linspace(0.5, 1.0, bins + 1)
    centres, accuracy, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            centres.append(confidence[mask].mean())
            accuracy.append(correct[mask].mean())
            counts.append(mask.sum())
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0.5, 1], [0.5, 1], "--", color="grey", lw=1)
    ax.plot(centres, accuracy, "o-", color="#3a6ea5")
    ax.set_xlabel("mean score in bin")
    ax.set_ylabel("fraction correct")
    ax.set_title("Reliability")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def error_coverage(curves: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for name, (coverage, error) in curves.items():
        ax.plot(coverage, error, label=name)
    ax.set_xlabel("coverage (fraction of clips answered)")
    ax.set_ylabel("error on answered clips")
    ax.invert_xaxis()
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
