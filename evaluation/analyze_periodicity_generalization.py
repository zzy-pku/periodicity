import argparse
import json
import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit
from torch.utils.data import DataLoader

from appendix_h_fanformer_sin import (
    SerializedSinDataset,
    apply_position_mask,
    build_position_mask,
    build_samples,
    collate_batch,
    decode_predictions,
    load_tokenizer,
    safe_float,
    set_seed,
)
from architecture import get_model_by_name


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def resolve_paths(checkpoint_dir: str | None, model_path: str | None, config_path: str | None) -> tuple[str, str]:
    if checkpoint_dir is not None:
        resolved_model = model_path or os.path.join(checkpoint_dir, "last_model.pt")
        resolved_config = config_path or os.path.join(checkpoint_dir, "config.json")
        return resolved_model, resolved_config

    if model_path is None or config_path is None:
        raise ValueError("Provide either --checkpoint_dir or both --model_path and --config_path.")

    return model_path, config_path


@dataclass
class PredictorBundle:
    model: torch.nn.Module
    tokenizer: object
    label_to_index: dict[int, int]
    index_to_token: dict[int, str]
    position_mask: torch.Tensor
    batch_size: int
    device: str


def load_model_bundle(model_path: str, config_path: str, batch_size_override: int | None, device: str) -> tuple[PredictorBundle, dict]:
    train_config = load_json(config_path)
    tokenizer = load_tokenizer(train_config["pretrained_name"])
    allowed_tokens = train_config["allowed_tokens"]
    allowed_token_ids = tokenizer.convert_tokens_to_ids(allowed_tokens)
    label_to_index = {token_id: idx for idx, token_id in enumerate(allowed_token_ids)}
    index_to_token = {idx: token for idx, token in enumerate(allowed_tokens)}
    position_mask = build_position_mask(index_to_token)

    model_name = train_config.get("model_name", "Qwen2.5Embedding-FANformer")
    model = get_model_by_name(
        model_name,
        pretrained_name=train_config["pretrained_name"],
        output_dim=len(allowed_tokens),
        num_layers=train_config["layers"],
        num_heads=train_config["num_heads"],
        norm_first=train_config.get("norm_first", True),
        freeze_emb=True,
        causal=train_config.get("causal", False),
    ).to(device)

    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return (
        PredictorBundle(
            model=model,
            tokenizer=tokenizer,
            label_to_index=label_to_index,
            index_to_token=index_to_token,
            position_mask=position_mask,
            batch_size=batch_size_override or train_config.get("batch_size", 64),
            device=device,
        ),
        train_config,
    )


def predict_values(bundle: PredictorBundle, xs: np.ndarray) -> list[dict]:
    samples = build_samples(xs)
    dataset = SerializedSinDataset(samples, bundle.tokenizer, bundle.label_to_index)
    loader = DataLoader(dataset, batch_size=bundle.batch_size, shuffle=False, collate_fn=collate_batch)
    rows = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(bundle.device)
            logits = bundle.model(input_ids)
            logits = apply_position_mask(logits, bundle.position_mask)
            pred_indices = logits.argmax(dim=-1)
            pred_texts = decode_predictions(pred_indices, bundle.index_to_token)
            for input_text, target_text, pred_text, x_value, y_value in zip(
                batch["input_texts"],
                batch["target_texts"],
                pred_texts,
                batch["x_values"].tolist(),
                batch["y_values"].tolist(),
            ):
                rows.append(
                    {
                        "x_value": x_value,
                        "y_value": y_value,
                        "input_text": input_text,
                        "target_text": target_text,
                        "pred_text": pred_text,
                        "pred_value": safe_float(pred_text),
                    }
                )
    rows.sort(key=lambda item: item["x_value"])
    return rows


def rows_to_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.array([row["x_value"] for row in rows], dtype=np.float64)
    ys = np.array([row["y_value"] for row in rows], dtype=np.float64)
    preds = np.array([row["pred_value"] for row in rows], dtype=np.float64)
    return xs, ys, preds


def compute_mae(preds: np.ndarray, ys: np.ndarray) -> float:
    return float(np.mean(np.abs(preds - ys)))


def compute_mse(preds: np.ndarray, ys: np.ndarray) -> float:
    return float(np.mean((preds - ys) ** 2))


def fit_line(xs: list[float], ys: list[float]) -> dict:
    if len(xs) < 2:
        return {"slope": float("nan"), "intercept": float("nan")}
    slope, intercept = np.polyfit(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), deg=1)
    return {"slope": float(slope), "intercept": float(intercept)}


def make_interval_points(left: float, right: float, num_points: int) -> np.ndarray:
    return np.linspace(left, right, num_points, endpoint=False, dtype=np.float64)


def compute_partition_metrics(
    bundle: PredictorBundle,
    interval_width: float,
    points_per_interval: int,
    id_left: float,
    id_right: float,
    ood_left: float,
    ood_right: float,
) -> dict:
    id_rows = predict_values(bundle, np.linspace(id_left, id_right, points_per_interval * max(1, int(round((id_right - id_left) / interval_width))), endpoint=False))
    _, id_ys, id_preds = rows_to_arrays(id_rows)
    e0 = compute_mae(id_preds, id_ys)
    e0_mse = compute_mse(id_preds, id_ys)

    def build_side(start_boundary: float, end_boundary: float, direction: str) -> dict:
        intervals = []
        k = 1
        current = start_boundary
        while True:
            if direction == "right":
                left = current
                right = min(current + interval_width, end_boundary)
                if left >= end_boundary - 1e-12:
                    break
                current = right
            else:
                right = current
                left = max(current - interval_width, end_boundary)
                if right <= end_boundary + 1e-12:
                    break
                current = left

            xs = make_interval_points(left, right, points_per_interval)
            rows = predict_values(bundle, xs)
            _, ys, preds = rows_to_arrays(rows)
            mae = compute_mae(preds, ys)
            mse = compute_mse(preds, ys)
            intervals.append(
                {
                    "k": k,
                    "left": float(left),
                    "right": float(right),
                    "mae": mae,
                    "mse": mse,
                    "gap": mae - e0,
                    "ratio": mae / (e0 + 1e-8),
                }
            )
            k += 1
        fit = fit_line([item["k"] for item in intervals], [item["mae"] for item in intervals])
        return {"intervals": intervals, "fit": fit}

    return {
        "id_baseline": {"mae": e0, "mse": e0_mse},
        "right": build_side(id_right, ood_right, "right"),
        "left": build_side(id_left, ood_left, "left"),
    }


def compute_period_consistency(
    bundle: PredictorBundle,
    phase_points: int,
    max_k: int,
    ood_left: float,
    ood_right: float,
) -> dict:
    theta = np.linspace(0.0, 2 * math.pi, phase_points, endpoint=False, dtype=np.float64)
    base_rows = predict_values(bundle, theta)
    _, _, base_preds = rows_to_arrays(base_rows)

    right = []
    left = []
    for k in range(1, max_k + 1):
        shifted_right = theta + 2 * math.pi * k
        if shifted_right.max() <= ood_right + 1e-12:
            _, _, preds = rows_to_arrays(predict_values(bundle, shifted_right))
            right.append({"k": k, "consistency": float(np.mean(np.abs(preds - base_preds)))})

        shifted_left = theta - 2 * math.pi * k
        if shifted_left.min() >= ood_left - 1e-12:
            _, _, preds = rows_to_arrays(predict_values(bundle, shifted_left))
            left.append({"k": k, "consistency": float(np.mean(np.abs(preds - base_preds)))})

    return {
        "right": {"values": right, "fit": fit_line([item["k"] for item in right], [item["consistency"] for item in right])},
        "left": {"values": left, "fit": fit_line([item["k"] for item in left], [item["consistency"] for item in left])},
    }


def compute_half_period_metrics(
    bundle: PredictorBundle,
    interval_width: float,
    points_per_interval: int,
    id_right: float,
    ood_right: float,
) -> dict:
    values = []
    k = 1
    current = id_right
    while current < ood_right - 1e-12:
        left = current
        right = min(current + interval_width, ood_right)
        xs = make_interval_points(left, right, points_per_interval)
        _, _, preds = rows_to_arrays(predict_values(bundle, xs))
        _, _, prev_preds = rows_to_arrays(predict_values(bundle, xs - math.pi))
        c_plus = float(np.mean(np.abs(preds - prev_preds)))
        c_minus = float(np.mean(np.abs(preds + prev_preds)))
        values.append({"k": k, "left": float(left), "right": float(right), "c_pi_plus": c_plus, "c_pi_minus": c_minus})
        k += 1
        current = right
    return values


def compute_shift_matching(
    bundle: PredictorBundle,
    interval_width: float,
    points_per_interval: int,
    id_right: float,
    ood_right: float,
    shift_multiples: list[int],
) -> dict:
    intervals = []
    current = id_right
    k = 1
    while current < ood_right - 1e-12:
        left = current
        right = min(current + interval_width, ood_right)
        xs = make_interval_points(left, right, points_per_interval)
        _, _, preds = rows_to_arrays(predict_values(bundle, xs))
        plus_scores = {}
        minus_scores = {}
        for multiple in shift_multiples:
            delta = multiple * math.pi
            shifted = xs - delta
            _, _, shifted_preds = rows_to_arrays(predict_values(bundle, shifted))
            plus_scores[str(multiple)] = float(np.mean(np.abs(preds - shifted_preds)))
            minus_scores[str(multiple)] = float(np.mean(np.abs(preds + shifted_preds)))
        best_plus = min(plus_scores, key=plus_scores.get)
        best_minus = min(minus_scores, key=minus_scores.get)
        intervals.append(
            {
                "k": k,
                "left": float(left),
                "right": float(right),
                "d_plus": plus_scores,
                "d_minus": minus_scores,
                "best_plus_shift_pi": int(best_plus),
                "best_minus_shift_pi": int(best_minus),
            }
        )
        k += 1
        current = right
    return {"shift_multiples": shift_multiples, "intervals": intervals}


def sine_func(x, amplitude, omega, phi, offset):
    return amplitude * np.sin(omega * x + phi) + offset


def compute_local_sine_fits(
    bundle: PredictorBundle,
    interval_width: float,
    points_per_interval: int,
    id_right: float,
    ood_right: float,
) -> list[dict]:
    fits = []
    current = id_right
    k = 1
    while current < ood_right - 1e-12:
        left = current
        right = min(current + interval_width, ood_right)
        xs = np.linspace(left, right, points_per_interval, endpoint=False, dtype=np.float64)
        _, _, preds = rows_to_arrays(predict_values(bundle, xs))

        amp0 = max((preds.max() - preds.min()) / 2, 1e-3)
        p0 = [amp0, 1.0, 0.0, float(np.mean(preds))]
        try:
            params, _ = curve_fit(sine_func, xs, preds, p0=p0, maxfev=20000)
            fit = {
                "k": k,
                "left": float(left),
                "right": float(right),
                "amplitude": float(params[0]),
                "omega": float(params[1]),
                "phi": float(params[2]),
                "offset": float(params[3]),
            }
        except Exception as exc:
            fit = {
                "k": k,
                "left": float(left),
                "right": float(right),
                "amplitude": float("nan"),
                "omega": float("nan"),
                "phi": float("nan"),
                "offset": float("nan"),
                "error": str(exc),
            }
        fits.append(fit)
        k += 1
        current = right
    return fits


def plot_full_prediction(rows: list[dict], out_path: str, id_left: float, id_right: float, ood_left: float, ood_right: float) -> None:
    xs, ys, preds = rows_to_arrays(rows)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axvspan(ood_left, id_left, color="#f4d7d7", alpha=0.35, label="OOD range")
    ax.axvspan(id_left, id_right, color="#d9ecff", alpha=0.35, label="ID range")
    ax.axvspan(id_right, ood_right, color="#f4d7d7", alpha=0.35)
    ax.axvline(id_left, color="#555555", linestyle="--", linewidth=1)
    ax.axvline(id_right, color="#555555", linestyle="--", linewidth=1)
    ax.plot(xs, ys, linewidth=2.3, color="#1f4e79", label="Ground truth")
    ax.plot(xs, preds, linewidth=1.8, color="#c0392b", label="Prediction")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Full Prediction Curve")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_partition_metric(partition: dict, metric: str, out_path: str, title: str) -> None:
    plt.figure(figsize=(10, 5))
    for side, color in [("left", "#8e44ad"), ("right", "#c0392b")]:
        values = partition[side]["intervals"]
        if values:
            plt.plot([item["k"] for item in values], [item[metric] for item in values], marker="o", label=side, color=color)
    plt.xlabel("OOD interval index k")
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_period_consistency(consistency: dict, out_path: str) -> None:
    plt.figure(figsize=(10, 5))
    for side, color in [("left", "#8e44ad"), ("right", "#c0392b")]:
        values = consistency[side]["values"]
        if values:
            plt.plot([item["k"] for item in values], [item["consistency"] for item in values], marker="o", label=side, color=color)
    plt.xlabel("k")
    plt.ylabel("C_k")
    plt.title("Period Consistency")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_half_period_metrics(values: list[dict], out_path: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot([item["k"] for item in values], [item["c_pi_plus"] for item in values], marker="o", label="C_pi_plus")
    plt.plot([item["k"] for item in values], [item["c_pi_minus"] for item in values], marker="o", label="C_pi_minus")
    plt.xlabel("Right OOD interval index k")
    plt.ylabel("Error")
    plt.title("Half-Period Metrics")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_shift_heatmap(shift_matching: dict, key: str, out_path: str, title: str) -> None:
    intervals = shift_matching["intervals"]
    if not intervals:
        return
    shifts = [str(m) for m in shift_matching["shift_multiples"]]
    matrix = np.array([[item[key][shift] for shift in shifts] for item in intervals], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(shifts)))
    ax.set_xticklabels(shifts)
    ax.set_yticks(range(len(intervals)))
    ax.set_yticklabels([item["k"] for item in intervals])
    ax.set_xlabel("Shift multiple of pi")
    ax.set_ylabel("Right OOD interval index k")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_local_fit_params(fits: list[dict], out_path: str) -> None:
    ks = [item["k"] for item in fits]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    params = [("amplitude", "A_k"), ("omega", "omega_k"), ("phi", "phi_k"), ("offset", "c_k")]
    for ax, (field, label) in zip(axes.flat, params):
        ax.plot(ks, [item[field] for item in fits], marker="o")
        ax.set_title(label)
        ax.set_xlabel("Right OOD interval index k")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Second-stage periodicity analysis for serialized sin(x) checkpoints.")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--id_left", type=float, required=True)
    parser.add_argument("--id_right", type=float, required=True)
    parser.add_argument("--ood_left", type=float, required=True)
    parser.add_argument("--ood_right", type=float, required=True)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--interval_width", type=float, default=math.pi)
    parser.add_argument("--points_per_interval", type=int, default=256)
    parser.add_argument("--phase_points", type=int, default=256)
    parser.add_argument("--max_k", type=int, default=6)
    parser.add_argument("--shift_multiples", type=str, default="1,2,3,4")
    parser.add_argument("--full_curve_points", type=int, default=4096)
    args = parser.parse_args()

    model_path, config_path = resolve_paths(args.checkpoint_dir, args.model_path, args.config_path)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    bundle, train_config = load_model_bundle(model_path, config_path, args.batch_size, args.device)
    shifts = [int(item.strip()) for item in args.shift_multiples.split(",") if item.strip()]

    full_xs = np.linspace(args.ood_left, args.ood_right, args.full_curve_points, endpoint=True, dtype=np.float64)
    full_rows = predict_values(bundle, full_xs)
    partition = compute_partition_metrics(
        bundle,
        args.interval_width,
        args.points_per_interval,
        args.id_left,
        args.id_right,
        args.ood_left,
        args.ood_right,
    )
    consistency = compute_period_consistency(bundle, args.phase_points, args.max_k, args.ood_left, args.ood_right)
    half_period = compute_half_period_metrics(bundle, args.interval_width, args.points_per_interval, args.id_right, args.ood_right)
    shift_matching = compute_shift_matching(bundle, args.interval_width, args.points_per_interval, args.id_right, args.ood_right, shifts)
    local_fits = compute_local_sine_fits(bundle, args.interval_width, args.points_per_interval, args.id_right, args.ood_right)

    summary = {
        "analysis_config": vars(args),
        "model_name": train_config.get("model_name", "Qwen2.5Embedding-FANformer"),
        "pretrained_name": train_config["pretrained_name"],
        "partition_metrics": partition,
        "period_consistency": consistency,
        "half_period_metrics": half_period,
        "shift_matching": shift_matching,
        "local_sine_fits": local_fits,
    }
    save_json(os.path.join(args.output_dir, "analysis_summary.json"), summary)
    save_json(os.path.join(args.output_dir, "full_prediction_preview.json"), full_rows[:200])

    plot_full_prediction(full_rows, os.path.join(args.output_dir, "full_prediction_curve.png"), args.id_left, args.id_right, args.ood_left, args.ood_right)
    plot_partition_metric(partition, "mae", os.path.join(args.output_dir, "partition_mae.png"), "Partition MAE")
    plot_partition_metric(partition, "gap", os.path.join(args.output_dir, "partition_gap.png"), "OOD Generalization Gap")
    plot_period_consistency(consistency, os.path.join(args.output_dir, "period_consistency.png"))
    plot_half_period_metrics(half_period, os.path.join(args.output_dir, "half_period_metrics.png"))
    plot_shift_heatmap(shift_matching, "d_plus", os.path.join(args.output_dir, "shift_matching_plus.png"), "Shift Matching D_plus")
    plot_shift_heatmap(shift_matching, "d_minus", os.path.join(args.output_dir, "shift_matching_minus.png"), "Shift Matching D_minus")
    plot_local_fit_params(local_fits, os.path.join(args.output_dir, "local_sine_fit_params.png"))


if __name__ == "__main__":
    main()
