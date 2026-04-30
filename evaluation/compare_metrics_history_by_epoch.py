import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected non-empty list in {path}")
    return data


def extract_series(history: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    epochs = np.array([item["epoch"] for item in history], dtype=np.float64)
    train_loss = np.array([item["train_loss"] for item in history], dtype=np.float64)
    id_loss = np.array([item["id_loss"] for item in history], dtype=np.float64)
    ood_loss = np.array([item["ood_loss"] for item in history], dtype=np.float64)
    return epochs, train_loss, id_loss, ood_loss


def plot_epoch_aligned_comparison(
    epochs_a: np.ndarray,
    train_a: np.ndarray,
    id_a: np.ndarray,
    ood_a: np.ndarray,
    label_a: str,
    epochs_b: np.ndarray,
    train_b: np.ndarray,
    id_b: np.ndarray,
    ood_b: np.ndarray,
    label_b: str,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    plots = [
        ("Train Loss", train_a, train_b),
        ("ID Loss", id_a, id_b),
        ("OOD Loss", ood_a, ood_b),
    ]

    for ax, (title, values_a, values_b) in zip(axes, plots):
        ax.plot(epochs_a, values_a, marker="o", linewidth=1.6, markersize=3.5, label=label_a)
        ax.plot(epochs_b, values_b, marker="o", linewidth=1.6, markersize=3.5, label=label_b)
        ax.set_title(title)
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Epoch")
    fig.suptitle("Metrics History Comparison Aligned by Epoch")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two metrics_history.json files aligned by epoch.")
    parser.add_argument("--metrics_a", type=str, required=True, help="Path to first metrics_history.json")
    parser.add_argument("--metrics_b", type=str, required=True, help="Path to second metrics_history.json")
    parser.add_argument("--label_a", type=str, default="Experiment A")
    parser.add_argument("--label_b", type=str, default="Experiment B")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save epoch-aligned comparison figure")
    args = parser.parse_args()

    history_a = load_metrics(args.metrics_a)
    history_b = load_metrics(args.metrics_b)

    epochs_a, train_a, id_a, ood_a = extract_series(history_a)
    epochs_b, train_b, id_b, ood_b = extract_series(history_b)

    plot_epoch_aligned_comparison(
        epochs_a,
        train_a,
        id_a,
        ood_a,
        args.label_a,
        epochs_b,
        train_b,
        id_b,
        ood_b,
        args.label_b,
        args.output_path,
    )


if __name__ == "__main__":
    main()
