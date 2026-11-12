"""Figures for the data sheet, the report and the error analysis.

Author: Mariah Habak
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from dronebird.augment import (build_distance_augment,  # noqa: E402
                               synthetic_test_degradation)
from dronebird.data.datasets import to_unit  # noqa: E402
from dronebird.data.images import gray_thumbnail, load_rgb  # noqa: E402

COLOURS = {"bird": "#d08c2c", "drone": "#3a6ea5"}
MODEL_COLOURS = ["#8a8a8a", "#d08c2c", "#3a6ea5", "#2f8f5b", "#a8466b"]
NAMES = ("bird", "drone")


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _show(ax, image):
    """Float (3, S, S) in [0, 1] or uint8 -> axis image."""
    if image.dtype == torch.uint8:
        image = to_unit(image)
    ax.imshow(image.permute(1, 2, 0).clamp(0, 1).numpy())
    ax.set_xticks([])
    ax.set_yticks([])


def data_overview(banks: dict, out_dir, data: dict, crop_settings) -> None:
    out_dir = Path(out_dir)
    _class_balance(banks, out_dir / "class_balance.png")
    _object_sizes(banks, data["evaluation"], out_dir / "object_sizes.png")
    if len(banks["train"]):
        _sample_crops(banks["train"], out_dir / "sample_crops.png")
        _recipes(banks["train"], data, crop_settings, out_dir / "train_vs_test_recipe.png")


def _class_balance(banks, path):
    parts = list(banks)
    x = np.arange(len(parts))
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for i, name in enumerate(NAMES):
        counts = [int((b.labels == i).sum()) for b in banks.values()]
        ax.bar(x + (i - 0.5) * 0.4, counts, 0.4, label=name, color=COLOURS[name])
    ax.set_xticks(x, parts)
    ax.set_ylabel("object crops")
    ax.set_title("Crops per partition")
    ax.legend(frameon=False)
    _save(fig, path)


def _object_sizes(banks, evaluation, path):
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    bins = np.logspace(np.log10(4), np.log10(700), 40)
    for i, name in enumerate(NAMES):
        sizes = torch.cat([b.object_size[b.labels == i] for b in banks.values()])
        ax.hist(sizes.numpy(), bins=bins, alpha=0.6, label=name, color=COLOURS[name])
    ax.axvline(evaluation["real_max_size"], color="black", lw=1, ls="--")
    ax.axvline(evaluation["synthetic_min_size"], color="black", lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("object size: longest box side in the frame (px, log scale)")
    ax.set_ylabel("crops")
    ax.set_title("Object sizes (dashed: real-small limit, dotted: synthetic limit)",
                 fontsize=8)
    ax.legend(frameon=False)
    _save(fig, path)


def _sample_crops(bank, path, per_class=8, seed=0):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, per_class, figsize=(per_class * 1.1, 2.6))
    for row in range(2):
        idx = np.flatnonzero(bank.labels.numpy() == row)
        for col, i in enumerate(rng.choice(idx, min(per_class, len(idx)), replace=False)):
            _show(axes[row, col], bank.images[i])
            axes[row, col].set_title(f"{bank.object_size[i]:.0f}px", fontsize=6)
        axes[row, 0].set_ylabel(NAMES[row], fontsize=8)
    fig.suptitle("Training crops (title: object size in the frame)", fontsize=9)
    _save(fig, path)


def _recipes(bank, data, crop_settings, path, seed=0):
    """One large object at every size: test recipe (top) vs training augmentation."""
    ev = data["evaluation"]
    large = torch.nonzero(bank.object_size >= ev["synthetic_min_size"], as_tuple=True)[0]
    if len(large) == 0:
        return
    i = int(large[torch.randint(len(large), (1,), generator=torch.Generator().manual_seed(seed))])
    image = to_unit(bank.images[i]).unsqueeze(0)
    sizes = ev["synthetic_sizes"]
    fig, axes = plt.subplots(2, len(sizes) + 1, figsize=((len(sizes) + 1) * 1.3, 3))
    _show(axes[0, 0], image[0])
    axes[0, 0].set_title("full", fontsize=7)
    _show(axes[1, 0], image[0])
    for col, s in enumerate(sizes, 1):
        test = synthetic_test_degradation(s, ev["test_degradation"], crop_settings.margin,
                                          crop_settings.canvas)(image)
        _show(axes[0, col], test[0])
        axes[0, col].set_title(f"{s}px", fontsize=7)
        settings = dict(data["distance_augment_preview"], p=1.0, min_size=s, max_size=s)
        torch.manual_seed(seed + col)
        aug = build_distance_augment(settings, crop_settings.margin, crop_settings.canvas)(image)
        _show(axes[1, col], aug[0])
    axes[0, 0].set_ylabel("test", fontsize=8)
    axes[1, 0].set_ylabel("train aug", fontsize=8)
    fig.suptitle("Test recipe (bicubic + JPEG) vs DistanceAugment (area + blur + noise)",
                 fontsize=8)
    _save(fig, path)


def duplicate_examples(images: list, groups: list, path, count: int = 6) -> None:
    members = {}
    for image, group in zip(images, groups):
        members.setdefault(group, []).append(image)
    shared = [v for v in members.values() if len(v) > 1][:count]
    if not shared:
        return
    fig, axes = plt.subplots(len(shared), 2, figsize=(3.2, 1.6 * len(shared)), squeeze=False)
    for row, group in enumerate(shared):
        for col in range(2):
            thumb = gray_thumbnail(load_rgb(group[col]), 128)
            axes[row, col].imshow(thumb.numpy(), cmap="gray")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            axes[row, col].set_title(Path(group[col]).name[:22], fontsize=5)
    fig.suptitle("Examples of near-duplicate groups", fontsize=8)
    _save(fig, path)


def loss_curves(history: list, path) -> None:
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend(frameon=False)
    axes[1].plot(epochs, [h["val_accuracy"] for h in history], color="#3a6ea5")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation accuracy")
    _save(fig, path)


def size_curves(table: dict, models: list, columns: list, test_set: str, metric: str,
                path, title: str) -> None:
    """table[(model, test_set, size)] = (mean, std)."""
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    x = np.arange(len(columns))
    for k, model in enumerate(models):
        mean = [table.get((model, test_set, c), (np.nan, np.nan))[0] for c in columns]
        std = [table.get((model, test_set, c), (np.nan, np.nan))[1] for c in columns]
        ax.errorbar(x + (k - len(models) / 2) * 0.04, mean, yerr=std, marker="o",
                    capsize=3, label=model, color=MODEL_COLOURS[k % len(MODEL_COLOURS)])
    ax.set_xticks(x, columns)
    ax.set_xlabel("object size (px)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title, fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    _save(fig, path)


def error_grid(images: torch.Tensor, cams: list, captions: list, path) -> None:
    n = len(captions)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(3.4, 1.7 * n), squeeze=False)
    for r in range(n):
        _show(axes[r, 0], images[r])
        axes[r, 0].set_title(captions[r], fontsize=5)
        _show(axes[r, 1], images[r])
        axes[r, 1].imshow(cams[r].numpy(), cmap="jet", alpha=0.45,
                          extent=(-0.5, images.shape[-1] - 0.5, images.shape[-2] - 0.5, -0.5))
        axes[r, 1].set_title("Grad-CAM", fontsize=5)
    _save(fig, path)
