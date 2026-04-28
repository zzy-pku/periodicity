import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation.analyze_periodicity_generalization import (
    compute_half_period_metrics,
    compute_local_sine_fits,
    compute_partition_metrics,
    compute_period_consistency,
    compute_shift_matching,
    load_model_bundle,
    plot_full_prediction,
    plot_half_period_metrics,
    plot_local_fit_params,
    plot_partition_metric,
    plot_period_consistency,
    plot_shift_heatmap,
    predict_values,
    resolve_paths,
    rows_to_arrays,
)
from appendix_h_fanformer_sin import (
    SerializedSinDataset,
    append_timestamp,
    build_datasets,
    build_position_mask,
    build_samples,
    collate_batch,
    evaluate,
    plot_predictions,
    set_seed,
)
from evaluation.compare_metrics_history import (
    extract_series,
    load_metrics as load_metrics_history,
    plot_comparison,
    plot_id_aligned_comparison,
)
from evaluation.compare_ood_id_gap_integral import (
    cumulative_trapezoid,
    extract_gap_series,
    interpolate_to_common_grid,
    plot_cumulative_curves,
    plot_difference_curve,
    plot_ratio_curve,
)
from evaluation.evaluate_serialized_sin_model import (
    build_probe_points,
    load_json,
    plot_metric_bars,
    probe_logits,
    save_json,
    save_jsonl,
)
from evaluation.plot_logit_probe import plot_logit_probe


def resolve_eval_ranges(train_config: dict, args) -> dict:
    default_id_left = train_config.get("id_left", -3 * math.pi)
    default_id_right = train_config.get("id_right", 3 * math.pi)
    default_ood_left = train_config.get("ood_left", -6 * math.pi)
    default_ood_right = train_config.get("ood_right", 6 * math.pi)
    return {
        "id_left": args.id_left if args.id_left is not None else default_id_left,
        "id_right": args.id_right if args.id_right is not None else default_id_right,
        "ood_left": args.ood_left if args.ood_left is not None else default_ood_left,
        "ood_right": args.ood_right if args.ood_right is not None else default_ood_right,
        "id_test_size": args.id_test_size if args.id_test_size is not None else train_config.get("id_test_size", 2000),
        "ood_test_size": args.ood_test_size if args.ood_test_size is not None else train_config.get("ood_test_size", 2000),
    }


def resolve_probe_config(ranges: dict, args) -> dict | None:
    probe_values = [args.logit_probe_left, args.logit_probe_right, args.logit_probe_step]
    if not any(value is not None for value in probe_values):
        return None
    if not all(value is not None for value in probe_values):
        raise ValueError(
            "If you want logit probing, provide all of --logit_probe_left, --logit_probe_right, and --logit_probe_step."
        )
    return {
        "left": args.logit_probe_left,
        "right": args.logit_probe_right,
        "step": args.logit_probe_step,
    }


def run_basic_evaluation(bundle, output_dir: str, ranges: dict) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    _, id_samples, ood_samples = build_datasets(
        train_size=1,
        id_test_size=ranges["id_test_size"],
        ood_test_size=ranges["ood_test_size"],
        id_left=ranges["id_left"],
        id_right=ranges["id_right"],
        ood_left=ranges["ood_left"],
        ood_right=ranges["ood_right"],
    )

    id_dataset = SerializedSinDataset(id_samples, bundle.tokenizer, bundle.label_to_index)
    ood_dataset = SerializedSinDataset(ood_samples, bundle.tokenizer, bundle.label_to_index)
    id_loader = DataLoader(id_dataset, batch_size=bundle.batch_size, shuffle=False, collate_fn=collate_batch)
    ood_loader = DataLoader(ood_dataset, batch_size=bundle.batch_size, shuffle=False, collate_fn=collate_batch)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    id_loss, id_metrics, id_records = evaluate(
        bundle.model,
        id_loader,
        criterion,
        bundle.device,
        bundle.index_to_token,
        bundle.position_mask,
    )
    ood_loss, ood_metrics, ood_records = evaluate(
        bundle.model,
        ood_loader,
        criterion,
        bundle.device,
        bundle.index_to_token,
        bundle.position_mask,
    )

    save_json(
        os.path.join(output_dir, "last_eval.json"),
        {
            "id_loss": id_loss,
            "ood_loss": ood_loss,
            "id_metrics": id_metrics,
            "ood_metrics": ood_metrics,
            "id_records_preview": id_records[:50],
            "ood_records_preview": ood_records[:50],
        },
    )

    plot_predictions(
        id_records,
        ood_records,
        os.path.join(output_dir, "prediction_curve.png"),
        ranges["id_left"],
        ranges["id_right"],
        ranges["ood_left"],
        ranges["ood_right"],
    )
    plot_metric_bars(
        id_loss,
        ood_loss,
        ylabel="Cross Entropy Loss",
        title="Cross Entropy Loss Distribution",
        out_path=os.path.join(output_dir, "cross_entropy_loss_curve.png"),
    )
    plot_metric_bars(
        id_metrics["mse"],
        ood_metrics["mse"],
        ylabel="MSE",
        title="MSE Distribution",
        out_path=os.path.join(output_dir, "mse_curve.png"),
    )
    return {
        "id_loss": id_loss,
        "ood_loss": ood_loss,
        "id_metrics": id_metrics,
        "ood_metrics": ood_metrics,
    }


def run_logit_probe(bundle, output_dir: str, probe_config: dict) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    probe_points = build_probe_points(probe_config["left"], probe_config["right"], probe_config["step"])
    probe_samples = build_samples(probe_points)
    probe_dataset = SerializedSinDataset(probe_samples, bundle.tokenizer, bundle.label_to_index)
    probe_loader = DataLoader(probe_dataset, batch_size=bundle.batch_size, shuffle=False, collate_fn=collate_batch)
    rows = probe_logits(bundle.model, probe_loader, bundle.device, bundle.index_to_token, bundle.position_mask)
    save_jsonl(os.path.join(output_dir, "logit_probe.jsonl"), rows)
    plot_logit_probe(rows, os.path.join(output_dir, "logit_probe_plot.png"))
    return {
        "num_points": len(rows),
        "left": probe_config["left"],
        "right": probe_config["right"],
        "step": probe_config["step"],
    }


def run_periodicity_analysis(bundle, train_config: dict, output_dir: str, ranges: dict, args) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    shifts = [int(item.strip()) for item in args.shift_multiples.split(",") if item.strip()]

    full_xs = np.linspace(ranges["ood_left"], ranges["ood_right"], args.full_curve_points, endpoint=True, dtype=np.float64)
    full_rows = predict_values(bundle, full_xs)
    partition = compute_partition_metrics(
        bundle,
        args.interval_width,
        args.points_per_interval,
        ranges["id_left"],
        ranges["id_right"],
        ranges["ood_left"],
        ranges["ood_right"],
    )
    consistency = compute_period_consistency(bundle, args.phase_points, args.max_k, ranges["ood_left"], ranges["ood_right"])
    half_period = compute_half_period_metrics(bundle, args.interval_width, args.points_per_interval, ranges["id_right"], ranges["ood_right"])
    shift_matching = compute_shift_matching(bundle, args.interval_width, args.points_per_interval, ranges["id_right"], ranges["ood_right"], shifts)
    local_fits = compute_local_sine_fits(bundle, args.interval_width, args.points_per_interval, ranges["id_right"], ranges["ood_right"])

    summary = {
        "analysis_config": {
            "interval_width": args.interval_width,
            "points_per_interval": args.points_per_interval,
            "phase_points": args.phase_points,
            "max_k": args.max_k,
            "shift_multiples": shifts,
            "full_curve_points": args.full_curve_points,
            **ranges,
        },
        "model_name": train_config.get("model_name", "Qwen2.5Embedding-FANformer"),
        "pretrained_name": train_config["pretrained_name"],
        "partition_metrics": partition,
        "period_consistency": consistency,
        "half_period_metrics": half_period,
        "shift_matching": shift_matching,
        "local_sine_fits": local_fits,
    }
    save_json(os.path.join(output_dir, "analysis_summary.json"), summary)
    save_json(os.path.join(output_dir, "full_prediction_preview.json"), full_rows[:200])

    plot_full_prediction(
        full_rows,
        os.path.join(output_dir, "full_prediction_curve.png"),
        ranges["id_left"],
        ranges["id_right"],
        ranges["ood_left"],
        ranges["ood_right"],
    )
    plot_partition_metric(partition, "mae", os.path.join(output_dir, "partition_mae.png"), "Partition MAE")
    plot_partition_metric(partition, "gap", os.path.join(output_dir, "partition_gap.png"), "OOD Generalization Gap")
    plot_period_consistency(consistency, os.path.join(output_dir, "period_consistency.png"))
    plot_half_period_metrics(half_period, os.path.join(output_dir, "half_period_metrics.png"))
    plot_shift_heatmap(shift_matching, "d_plus", os.path.join(output_dir, "shift_matching_plus.png"), "Shift Matching D_plus")
    plot_shift_heatmap(shift_matching, "d_minus", os.path.join(output_dir, "shift_matching_minus.png"), "Shift Matching D_minus")
    plot_local_fit_params(local_fits, os.path.join(output_dir, "local_sine_fit_params.png"))

    _, ys, preds = rows_to_arrays(full_rows)
    return {
        "full_curve_mae": float(np.mean(np.abs(preds - ys))),
        "full_curve_mse": float(np.mean((preds - ys) ** 2)),
        "num_full_curve_points": len(full_rows),
    }


def resolve_metrics_path(run_dir: str | None, metrics_path: str | None) -> str:
    if run_dir is not None:
        return os.path.join(run_dir, "metrics_history.json")
    if metrics_path is None:
        raise ValueError("Provide either a run directory or an explicit metrics_history.json path.")
    return metrics_path


def default_label(path: str) -> str:
    return Path(path).resolve().parent.name


def run_compare_mode(args) -> str:
    output_dir = append_timestamp(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    metrics_a = resolve_metrics_path(args.run_a, args.metrics_a)
    metrics_b = resolve_metrics_path(args.run_b, args.metrics_b)
    label_a = args.label_a or default_label(metrics_a)
    label_b = args.label_b or default_label(metrics_b)

    history_a = load_metrics_history(metrics_a)
    history_b = load_metrics_history(metrics_b)
    train_a, id_a, ood_a = extract_series(history_a)
    train_b, id_b, ood_b = extract_series(history_b)

    metrics_compare_dir = os.path.join(output_dir, "metrics_history_compare")
    integral_compare_dir = os.path.join(output_dir, "ood_id_gap_integral_compare")
    os.makedirs(metrics_compare_dir, exist_ok=True)
    os.makedirs(integral_compare_dir, exist_ok=True)

    train_aligned_path = os.path.join(metrics_compare_dir, "train_aligned.png")
    id_aligned_path = os.path.join(metrics_compare_dir, "id_aligned.png")
    plot_comparison(train_a, id_a, ood_a, label_a, train_b, id_b, ood_b, label_b, train_aligned_path)
    plot_id_aligned_comparison(train_a, id_a, ood_a, label_a, train_b, id_b, ood_b, label_b, id_aligned_path)

    epochs_a, gaps_a = extract_gap_series(history_a)
    epochs_b, gaps_b = extract_gap_series(history_b)
    integral_a = cumulative_trapezoid(epochs_a, gaps_a)
    integral_b = cumulative_trapezoid(epochs_b, gaps_b)
    common_epochs, interp_a, interp_b = interpolate_to_common_grid(epochs_a, integral_a, epochs_b, integral_b)
    diff_curve = interp_a - interp_b
    ratio_curve = interp_a / (interp_b + 1e-12)

    integral_curves_path = os.path.join(integral_compare_dir, "integral_curves.png")
    difference_path = os.path.join(integral_compare_dir, "difference_curve.png")
    ratio_path = os.path.join(integral_compare_dir, "ratio_curve.png")
    plot_cumulative_curves(epochs_a, integral_a, label_a, epochs_b, integral_b, label_b, integral_curves_path)
    plot_difference_curve(common_epochs, diff_curve, difference_path)
    plot_ratio_curve(common_epochs, ratio_curve, ratio_path)

    save_json(
        os.path.join(output_dir, "compare_suite_config.json"),
        {
            "mode": "compare",
            "metrics_a": metrics_a,
            "metrics_b": metrics_b,
            "label_a": label_a,
            "label_b": label_b,
            "output_dir": output_dir,
        },
    )
    save_json(
        os.path.join(output_dir, "compare_suite_summary.json"),
        {
            "metrics_history_compare": {
                "train_aligned_path": train_aligned_path,
                "id_aligned_path": id_aligned_path,
            },
            "ood_id_gap_integral_compare": {
                "integral_curves_path": integral_curves_path,
                "difference_path": difference_path,
                "ratio_path": ratio_path,
                "final_integral_a": float(integral_a[-1]),
                "final_integral_b": float(integral_b[-1]),
                "final_difference": float(diff_curve[-1]),
                "final_ratio": float(ratio_curve[-1]),
                "common_epoch_start": float(common_epochs[0]),
                "common_epoch_end": float(common_epochs[-1]),
            },
        },
    )
    return output_dir


def run_single_mode(args) -> str:
    model_path, config_path = resolve_paths(args.checkpoint_dir, args.model_path, args.config_path)
    train_config = load_json(config_path)
    ranges = resolve_eval_ranges(train_config, args)
    probe_config = resolve_probe_config(ranges, args)

    set_seed(args.seed)
    output_dir = append_timestamp(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    bundle, train_config = load_model_bundle(model_path, config_path, args.batch_size, args.device)
    basic_dir = os.path.join(output_dir, "basic_eval")
    analysis_dir = os.path.join(output_dir, "periodicity_analysis")
    probe_dir = os.path.join(output_dir, "logit_probe")

    basic_summary = run_basic_evaluation(bundle, basic_dir, ranges)
    probe_summary = run_logit_probe(bundle, probe_dir, probe_config) if probe_config is not None else None
    analysis_summary = run_periodicity_analysis(bundle, train_config, analysis_dir, ranges, args)

    save_json(
        os.path.join(output_dir, "evaluation_suite_config.json"),
        {
            "mode": "single",
            "checkpoint_dir": args.checkpoint_dir,
            "model_path": model_path,
            "config_path": config_path,
            "output_dir": output_dir,
            "seed": args.seed,
            "device": args.device,
            "batch_size": bundle.batch_size,
            "ranges": ranges,
            "probe_config": probe_config,
            "analysis_config": {
                "interval_width": args.interval_width,
                "points_per_interval": args.points_per_interval,
                "phase_points": args.phase_points,
                "max_k": args.max_k,
                "shift_multiples": args.shift_multiples,
                "full_curve_points": args.full_curve_points,
            },
        },
    )
    save_json(
        os.path.join(output_dir, "evaluation_suite_summary.json"),
        {
            "basic_eval": basic_summary,
            "logit_probe": probe_summary,
            "periodicity_analysis": analysis_summary,
            "subdirs": {
                "basic_eval": basic_dir,
                "logit_probe": probe_dir if probe_summary is not None else None,
                "periodicity_analysis": analysis_dir,
            },
        },
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified entrypoint for serialized sin(x) evaluation and analysis.")
    parser.add_argument("--mode", choices=["single", "compare"], default="single")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory containing last_model.pt and config.json.")
    parser.add_argument("--output_dir", type=str, required=True, help="Base directory for evaluation outputs. A timestamp will be appended.")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--id_test_size", type=int, default=None)
    parser.add_argument("--ood_test_size", type=int, default=None)
    parser.add_argument("--id_left", type=float, default=None)
    parser.add_argument("--id_right", type=float, default=None)
    parser.add_argument("--ood_left", type=float, default=None)
    parser.add_argument("--ood_right", type=float, default=None)
    parser.add_argument("--logit_probe_left", type=float, default=None)
    parser.add_argument("--logit_probe_right", type=float, default=None)
    parser.add_argument("--logit_probe_step", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--interval_width", type=float, default=math.pi)
    parser.add_argument("--points_per_interval", type=int, default=256)
    parser.add_argument("--phase_points", type=int, default=256)
    parser.add_argument("--max_k", type=int, default=6)
    parser.add_argument("--shift_multiples", type=str, default="1,2,3,4")
    parser.add_argument("--full_curve_points", type=int, default=4096)
    parser.add_argument("--run_a", type=str, default=None, help="First experiment directory for compare mode.")
    parser.add_argument("--run_b", type=str, default=None, help="Second experiment directory for compare mode.")
    parser.add_argument("--metrics_a", type=str, default=None, help="Explicit metrics_history.json path for compare mode.")
    parser.add_argument("--metrics_b", type=str, default=None, help="Explicit metrics_history.json path for compare mode.")
    parser.add_argument("--label_a", type=str, default=None)
    parser.add_argument("--label_b", type=str, default=None)
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "compare":
        if args.run_a is None and args.metrics_a is None:
            parser.error("compare mode requires --run_a or --metrics_a.")
        if args.run_b is None and args.metrics_b is None:
            parser.error("compare mode requires --run_b or --metrics_b.")
        run_compare_mode(args)
        return

    if args.checkpoint_dir is None and (args.model_path is None or args.config_path is None):
        parser.error("single mode requires --checkpoint_dir or both --model_path and --config_path.")
    run_single_mode(args)


if __name__ == "__main__":
    main()
