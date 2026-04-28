import argparse
import json
import os
from datetime import datetime
from pathlib import Path

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


def append_timestamp(output_dir: str) -> str:
    normalized = output_dir.rstrip("/\\")
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    return f"{normalized}_{timestamp}"


def save_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two serialized sin(x) experiment runs.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_a", type=str, default=None, help="First experiment directory.")
    parser.add_argument("--run_b", type=str, default=None, help="Second experiment directory.")
    parser.add_argument("--metrics_a", type=str, default=None, help="Explicit metrics_history.json path for experiment A.")
    parser.add_argument("--metrics_b", type=str, default=None, help="Explicit metrics_history.json path for experiment B.")
    parser.add_argument("--label_a", type=str, default=None)
    parser.add_argument("--label_b", type=str, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.run_a is None and args.metrics_a is None:
        parser.error("compare mode requires --run_a or --metrics_a.")
    if args.run_b is None and args.metrics_b is None:
        parser.error("compare mode requires --run_b or --metrics_b.")
    run_compare_mode(args)


if __name__ == "__main__":
    main()
