import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def plot_logit_probe(rows: list[dict], output_path: str, title: str = "True/Predicted Values and Sign Logits") -> None:
    rows = sorted(rows, key=lambda item: item["x_value"])

    xs = np.array([row["x_value"] for row in rows], dtype=np.float64)
    y_true = np.array([row["y_value"] for row in rows], dtype=np.float64)
    y_pred = np.array([row["pred_value"] for row in rows], dtype=np.float64)
    plus_logits = np.array([row["masked_logits"][0][0] for row in rows], dtype=np.float64)
    minus_logits = np.array([row["masked_logits"][0][1] for row in rows], dtype=np.float64)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    line_true = ax1.plot(xs, y_true, color="#1f4e79", linewidth=2.5, label="Ground truth")
    line_pred = ax1.plot(xs, y_pred, color="#c0392b", linewidth=2.0, label="Prediction")
    line_plus = ax2.plot(xs, plus_logits, color="#2e8b57", linestyle="--", linewidth=1.8, label="Logit '+'")
    line_minus = ax2.plot(xs, minus_logits, color="#8e44ad", linestyle="--", linewidth=1.8, label="Logit '-'")

    ax1.set_xlabel("x")
    ax1.set_ylabel("Function value")
    ax2.set_ylabel("Sign logits")
    ax1.set_title(title)
    ax1.grid(alpha=0.3)

    lines = line_true + line_pred + line_plus + line_minus
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot true/predicted values and sign logits from logit_probe.jsonl")
    parser.add_argument("--logit_probe_path", type=str, required=True, help="Path to logit_probe.jsonl")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the figure")
    args = parser.parse_args()

    rows = load_jsonl(args.logit_probe_path)
    plot_logit_probe(rows, args.output_path)


if __name__ == "__main__":
    main()
