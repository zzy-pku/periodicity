import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from appendix_h_fanformer_sin import (
    SerializedSinDataset,
    apply_position_mask,
    build_datasets,
    build_samples,
    build_position_mask,
    collate_batch,
    decode_predictions,
    evaluate,
    load_tokenizer,
    plot_predictions,
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


def plot_metric_bars(id_value: float, ood_value: float, ylabel: str, title: str, out_path: str) -> None:
    plt.figure(figsize=(6, 5))
    xs = ["ID", "OOD"]
    ys = [id_value, ood_value]
    colors = ["#4f81bd", "#c0504d"]
    plt.bar(xs, ys, color=colors)
    plt.ylabel(ylabel)
    plt.title(title)
    for idx, value in enumerate(ys):
        plt.text(idx, value, f"{value:.6f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_probe_points(left: float, right: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("--logit_probe_step must be > 0.")
    if right < left:
        raise ValueError("--logit_probe_right must be >= --logit_probe_left.")
    count = int(np.floor((right - left) / step)) + 1
    points = left + np.arange(count, dtype=np.float64) * step
    if points[-1] < right - 1e-12:
        points = np.append(points, right)
    return points


def probe_logits(model, dataloader, device, index_to_token: dict[int, str], position_mask: torch.Tensor) -> list[dict]:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            raw_logits = model(input_ids)
            masked_logits = apply_position_mask(raw_logits, position_mask)
            pred_indices = masked_logits.argmax(dim=-1)
            pred_texts = decode_predictions(pred_indices, index_to_token)

            raw_np = raw_logits.detach().cpu().numpy()
            masked_np = masked_logits.detach().cpu().numpy()
            for idx, (input_text, target_text, pred_text, x_value, y_value) in enumerate(
                zip(
                    batch["input_texts"],
                    batch["target_texts"],
                    pred_texts,
                    batch["x_values"].tolist(),
                    batch["y_values"].tolist(),
                )
            ):
                rows.append(
                    {
                        "x_value": x_value,
                        "y_value": y_value,
                        "input_text": input_text,
                        "target_text": target_text,
                        "pred_text": pred_text,
                        "pred_value": safe_float(pred_text),
                        "allowed_tokens": [index_to_token[i] for i in range(len(index_to_token))],
                        "raw_logits": raw_np[idx].tolist(),
                        "masked_logits": masked_np[idx].tolist(),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved serialized sin(x) model checkpoint.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory containing last_model.pt and config.json.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to last_model.pt.")
    parser.add_argument("--config_path", type=str, default=None, help="Path to config.json.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save evaluation outputs.")
    parser.add_argument("--id_test_size", type=int, default=2000)
    parser.add_argument("--ood_test_size", type=int, default=2000)
    parser.add_argument("--id_left", type=float, required=True)
    parser.add_argument("--id_right", type=float, required=True)
    parser.add_argument("--ood_left", type=float, required=True)
    parser.add_argument("--ood_right", type=float, required=True)
    parser.add_argument("--logit_probe_left", type=float, default=None)
    parser.add_argument("--logit_probe_right", type=float, default=None)
    parser.add_argument("--logit_probe_step", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None, help="Override evaluation batch size.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_path, config_path = resolve_paths(args.checkpoint_dir, args.model_path, args.config_path)
    train_config = load_json(config_path)

    probe_args = [args.logit_probe_left, args.logit_probe_right, args.logit_probe_step]
    if any(value is not None for value in probe_args) and not all(value is not None for value in probe_args):
        raise ValueError(
            "If you want logit probing, provide all of --logit_probe_left, --logit_probe_right, and --logit_probe_step."
        )

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = load_tokenizer(train_config["pretrained_name"])
    allowed_tokens = train_config["allowed_tokens"]
    allowed_token_ids = tokenizer.convert_tokens_to_ids(allowed_tokens)
    label_to_index = {token_id: idx for idx, token_id in enumerate(allowed_token_ids)}
    index_to_token = {idx: token for idx, token in enumerate(allowed_tokens)}
    position_mask = build_position_mask(index_to_token)

    _, id_samples, ood_samples = build_datasets(
        train_size=max(train_config.get("train_size", 1), 1),
        id_test_size=args.id_test_size,
        ood_test_size=args.ood_test_size,
        id_left=args.id_left,
        id_right=args.id_right,
        ood_left=args.ood_left,
        ood_right=args.ood_right,
    )

    id_dataset = SerializedSinDataset(id_samples, tokenizer, label_to_index)
    ood_dataset = SerializedSinDataset(ood_samples, tokenizer, label_to_index)
    batch_size = args.batch_size or train_config.get("batch_size", 64)
    id_loader = DataLoader(id_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

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
    ).to(args.device)

    state_dict = torch.load(model_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    id_loss, id_metrics, id_records = evaluate(model, id_loader, criterion, args.device, index_to_token, position_mask)
    ood_loss, ood_metrics, ood_records = evaluate(model, ood_loader, criterion, args.device, index_to_token, position_mask)

    save_json(
        os.path.join(args.output_dir, "evaluation_config.json"),
        {
            "checkpoint_dir": args.checkpoint_dir,
            "model_path": model_path,
            "config_path": config_path,
            "output_dir": args.output_dir,
            "id_test_size": args.id_test_size,
            "ood_test_size": args.ood_test_size,
            "id_left": args.id_left,
            "id_right": args.id_right,
            "ood_left": args.ood_left,
            "ood_right": args.ood_right,
            "logit_probe_left": args.logit_probe_left,
            "logit_probe_right": args.logit_probe_right,
            "logit_probe_step": args.logit_probe_step,
            "batch_size": batch_size,
            "device": args.device,
            "seed": args.seed,
            "model_name": model_name,
            "pretrained_name": train_config["pretrained_name"],
            "layers": train_config["layers"],
            "num_heads": train_config["num_heads"],
        },
    )
    save_json(
        os.path.join(args.output_dir, "last_eval.json"),
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
        os.path.join(args.output_dir, "prediction_curve.png"),
        args.id_left,
        args.id_right,
        args.ood_left,
        args.ood_right,
    )
    plot_metric_bars(
        id_loss,
        ood_loss,
        ylabel="Cross Entropy Loss",
        title="Cross Entropy Loss Distribution",
        out_path=os.path.join(args.output_dir, "cross_entropy_loss_curve.png"),
    )
    plot_metric_bars(
        id_metrics["mse"],
        ood_metrics["mse"],
        ylabel="MSE",
        title="MSE Distribution",
        out_path=os.path.join(args.output_dir, "mse_curve.png"),
    )

    if args.logit_probe_left is not None:
        probe_points = build_probe_points(args.logit_probe_left, args.logit_probe_right, args.logit_probe_step)
        probe_samples = build_samples(probe_points)
        probe_dataset = SerializedSinDataset(probe_samples, tokenizer, label_to_index)
        probe_loader = DataLoader(probe_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
        probe_rows = probe_logits(model, probe_loader, args.device, index_to_token, position_mask)
        save_jsonl(os.path.join(args.output_dir, "logit_probe.jsonl"), probe_rows)


if __name__ == "__main__":
    main()
