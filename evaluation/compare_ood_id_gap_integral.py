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


def extract_gap_series(history: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    epochs = np.array([item["epoch"] for item in history], dtype=np.float64)
    gaps = np.array([item["ood_loss"] - item["id_loss"] for item in history], dtype=np.float64)
    return epochs, gaps


def cumulative_trapezoid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    area = np.zeros_like(y, dtype=np.float64)
    for i in range(1, len(y)):
        dx = x[i] - x[i - 1]
        area[i] = area[i - 1] + 0.5 * (y[i] + y[i - 1]) * dx
    return area


def interpolate_to_common_grid(x_a: np.ndarray, y_a: np.ndarray, x_b: np.ndarray, y_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = max(x_a.min(), x_b.min())
    right = min(x_a.max(), x_b.max())
    if right <= left:
        raise ValueError("The two histories do not have overlapping epoch ranges.")

    common = np.unique(
        np.concatenate(
            [
                x_a[(x_a >= left) & (x_a <= right)],
                x_b[(x_b >= left) & (x_b <= right)],
            ]
        )
    )
    interp_a = np.interp(common, x_a, y_a)
    interp_b = np.interp(common, x_b, y_b)
    return common, interp_a, interp_b


def plot_cumulative_curves(epochs_a, integral_a, label_a, epochs_b, integral_b, label_b, out_path):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_a, integral_a, marker="o", linewidth=1.8, label=label_a)
    plt.plot(epochs_b, integral_b, marker="o", linewidth=1.8, label=label_b)
    plt.xlabel("Epoch")
    plt.ylabel("Cumulative Integral of (OOD Loss - ID Loss)")
    plt.title("Cumulative OOD-ID Gap Integral")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_difference_curve(common_epochs, diff_curve, out_path):
    plt.figure(figsize=(10, 5))
    plt.plot(common_epochs, diff_curve, marker="o", linewidth=1.8)
    plt.axhline(0.0, color="#555555", linestyle="--", linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("Integral Difference (A - B)")
    plt.title("Difference Between Cumulative OOD-ID Gap Integrals")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_ratio_curve(common_epochs, ratio_curve, out_path):
    plt.figure(figsize=(10, 5))
    plt.plot(common_epochs, ratio_curve, marker="o", linewidth=1.8)
    plt.axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("Integral Ratio (A / B)")
    plt.title("Ratio Between Cumulative OOD-ID Gap Integrals")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare cumulative integrals of (ood_loss - id_loss) from two metrics_history.json files."
    )
    parser.add_argument("--metrics_a", type=str, required=True)
    parser.add_argument("--metrics_b", type=str, required=True)
    parser.add_argument("--label_a", type=str, default="Experiment A")
    parser.add_argument("--label_b", type=str, default="Experiment B")
    parser.add_argument("--integral_output_path", type=str, required=True, help="Output path for the cumulative integral curves.")
    parser.add_argument("--difference_output_path", type=str, required=True, help="Output path for the integral difference curve.")
    parser.add_argument("--ratio_output_path", type=str, required=True, help="Output path for the integral ratio curve.")
    parser.add_argument("--summary_output_path", type=str, default=None, help="Optional JSON summary path.")
    args = parser.parse_args()

    history_a = load_metrics(args.metrics_a)
    history_b = load_metrics(args.metrics_b)

    epochs_a, gaps_a = extract_gap_series(history_a)
    epochs_b, gaps_b = extract_gap_series(history_b)
    integral_a = cumulative_trapezoid(epochs_a, gaps_a)
    integral_b = cumulative_trapezoid(epochs_b, gaps_b)

    common_epochs, interp_a, interp_b = interpolate_to_common_grid(epochs_a, integral_a, epochs_b, integral_b)
    diff_curve = interp_a - interp_b
    ratio_curve = interp_a / (interp_b + 1e-12)

    plot_cumulative_curves(epochs_a, integral_a, args.label_a, epochs_b, integral_b, args.label_b, args.integral_output_path)
    plot_difference_curve(common_epochs, diff_curve, args.difference_output_path)
    plot_ratio_curve(common_epochs, ratio_curve, args.ratio_output_path)

    if args.summary_output_path is not None:
        summary = {
            "metrics_a": args.metrics_a,
            "metrics_b": args.metrics_b,
            "label_a": args.label_a,
            "label_b": args.label_b,
            "final_integral_a": float(integral_a[-1]),
            "final_integral_b": float(integral_b[-1]),
            "final_difference": float(diff_curve[-1]),
            "final_ratio": float(ratio_curve[-1]),
            "common_epoch_start": float(common_epochs[0]),
            "common_epoch_end": float(common_epochs[-1]),
        }
        os.makedirs(os.path.dirname(args.summary_output_path) or ".", exist_ok=True)
        with open(args.summary_output_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
