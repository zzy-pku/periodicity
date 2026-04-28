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


def extract_series(history: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_loss = np.array([item["train_loss"] for item in history], dtype=np.float64)
    id_loss = np.array([item["id_loss"] for item in history], dtype=np.float64)
    ood_loss = np.array([item["ood_loss"] for item in history], dtype=np.float64)
    return train_loss, id_loss, ood_loss


def plot_comparison(
    train_a: np.ndarray,
    id_a: np.ndarray,
    ood_a: np.ndarray,
    label_a: str,
    train_b: np.ndarray,
    id_b: np.ndarray,
    ood_b: np.ndarray,
    label_b: str,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=False, sharey=False)

    axes[0].plot(train_a, id_a, marker="o", linewidth=1.8, label=label_a)
    axes[0].plot(train_b, id_b, marker="o", linewidth=1.8, label=label_b)
    axes[0].set_title("ID Loss vs Train Loss")
    axes[0].set_xlabel("Train Loss")
    axes[0].set_ylabel("ID Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(train_a, ood_a, marker="o", linewidth=1.8, label=label_a)
    axes[1].plot(train_b, ood_b, marker="o", linewidth=1.8, label=label_b)
    axes[1].set_title("OOD Loss vs Train Loss")
    axes[1].set_xlabel("Train Loss")
    axes[1].set_ylabel("OOD Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle("Metrics History Comparison Aligned by Train Loss")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_id_aligned_comparison(
    train_a: np.ndarray,
    id_a: np.ndarray,
    ood_a: np.ndarray,
    label_a: str,
    train_b: np.ndarray,
    id_b: np.ndarray,
    ood_b: np.ndarray,
    label_b: str,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=False, sharey=False)

    axes[0].plot(id_a, train_a, marker="o", linewidth=1.8, label=label_a)
    axes[0].plot(id_b, train_b, marker="o", linewidth=1.8, label=label_b)
    axes[0].set_title("Train Loss vs ID Loss")
    axes[0].set_xlabel("ID Loss")
    axes[0].set_ylabel("Train Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(id_a, ood_a, marker="o", linewidth=1.8, label=label_a)
    axes[1].plot(id_b, ood_b, marker="o", linewidth=1.8, label=label_b)
    axes[1].set_title("OOD Loss vs ID Loss")
    axes[1].set_xlabel("ID Loss")
    axes[1].set_ylabel("OOD Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle("Metrics History Comparison Aligned by ID Loss")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two metrics_history.json files using train_loss as x-axis.")
    parser.add_argument("--metrics_a", type=str, required=True, help="Path to first metrics_history.json")
    parser.add_argument("--metrics_b", type=str, required=True, help="Path to second metrics_history.json")
    parser.add_argument("--label_a", type=str, default="Experiment A")
    parser.add_argument("--label_b", type=str, default="Experiment B")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save comparison figure")
    parser.add_argument("--id_aligned_output_path", type=str, default=None, help="Optional path to save ID-aligned comparison figure.")
    args = parser.parse_args()

    history_a = load_metrics(args.metrics_a)
    history_b = load_metrics(args.metrics_b)

    train_a, id_a, ood_a = extract_series(history_a)
    train_b, id_b, ood_b = extract_series(history_b)

    plot_comparison(
        train_a,
        id_a,
        ood_a,
        args.label_a,
        train_b,
        id_b,
        ood_b,
        args.label_b,
        args.output_path,
    )

    if args.id_aligned_output_path is not None:
        plot_id_aligned_comparison(
            train_a,
            id_a,
            ood_a,
            args.label_a,
            train_b,
            id_b,
            ood_b,
            args.label_b,
            args.id_aligned_output_path,
        )


if __name__ == "__main__":
    main()
