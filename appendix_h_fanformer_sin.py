import argparse
import json
import math
import os
import random
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from architecture import get_model_by_name


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_signed_zero(value: float, eps: float = 5e-8) -> float:
    if abs(value) < eps:
        return 0.0
    return value


def format_fixed_10(value: float) -> str:
    value = normalize_signed_zero(value)
    sign = "+" if value >= 0 else "-"
    magnitude = abs(value)
    integer_part = int(magnitude)
    fractional_part = magnitude - integer_part
    fractional_digits = int(round(fractional_part * 1_000_000))

    if fractional_digits == 1_000_000:
        integer_part += 1
        fractional_digits = 0

    if integer_part > 99:
        raise ValueError(f"Value {value} exceeds the representable range for fixed 10-char formatting.")

    return f"{sign}{integer_part:02d}.{fractional_digits:06d}"


def safe_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return float("nan")


def load_tokenizer(tokenizer_name: str):
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    except Exception:
        return AutoTokenizer.from_pretrained(tokenizer_name)


@dataclass
class Example:
    x_value: float
    y_value: float
    input_text: str
    target_text: str


def build_samples(points: np.ndarray) -> list[Example]:
    samples = []
    for x in points:
        y = math.sin(float(x))
        samples.append(
            Example(
                x_value=float(x),
                y_value=float(y),
                input_text=format_fixed_10(float(x)),
                target_text=format_fixed_10(float(y)),
            )
        )
    return samples


def validate_ranges(id_left: float, id_right: float, ood_left: float, ood_right: float) -> None:
    if not ood_left < id_left < id_right < ood_right:
        raise ValueError(
            "Expected ordered ranges: ood_left < id_left < id_right < ood_right. "
            f"Got {ood_left=}, {id_left=}, {id_right=}, {ood_right=}."
        )


def build_datasets(
    train_size: int,
    id_test_size: int,
    ood_test_size: int,
    id_left: float,
    id_right: float,
    ood_left: float,
    ood_right: float,
) -> tuple[list[Example], list[Example], list[Example]]:
    validate_ranges(id_left, id_right, ood_left, ood_right)
    train_points = np.linspace(id_left, id_right, train_size, endpoint=True, dtype=np.float64)

    # Offset the in-distribution grid to avoid exact overlap with training samples.
    id_step = (id_right - id_left) / id_test_size
    id_start = id_left + 0.5 * id_step
    id_end = id_right - 0.5 * id_step
    id_points = np.linspace(id_start, id_end, id_test_size, endpoint=True, dtype=np.float64)

    left_size = ood_test_size // 2
    right_size = ood_test_size - left_size
    left_points = np.linspace(ood_left, id_left, left_size, endpoint=False, dtype=np.float64)
    right_points = np.linspace(id_right, ood_right, right_size, endpoint=True, dtype=np.float64)
    ood_points = np.concatenate([left_points, right_points])

    return build_samples(train_points), build_samples(id_points), build_samples(ood_points)


class SerializedSinDataset(Dataset):
    def __init__(self, samples: list[Example], tokenizer, label_to_index: dict[int, int]):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label_to_index = label_to_index

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        input_ids = self.tokenizer.encode(sample.input_text, add_special_tokens=False)
        target_ids = self.tokenizer.encode(sample.target_text, add_special_tokens=False)

        if len(input_ids) != 10 or len(target_ids) != 10:
            raise ValueError(
                f"Expected fixed 10-token strings, got input={len(input_ids)}, target={len(target_ids)} "
                f"for {sample.input_text} -> {sample.target_text}"
            )

        target_labels = [self.label_to_index[token_id] for token_id in target_ids]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_labels": torch.tensor(target_labels, dtype=torch.long),
            "input_text": sample.input_text,
            "target_text": sample.target_text,
            "x_value": sample.x_value,
            "y_value": sample.y_value,
        }


def collate_batch(batch: list[dict]) -> dict:
    input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=0)
    target_labels = pad_sequence([item["target_labels"] for item in batch], batch_first=True, padding_value=-100)
    return {
        "input_ids": input_ids,
        "target_labels": target_labels,
        "input_texts": [item["input_text"] for item in batch],
        "target_texts": [item["target_text"] for item in batch],
        "x_values": torch.tensor([item["x_value"] for item in batch], dtype=torch.float32),
        "y_values": torch.tensor([item["y_value"] for item in batch], dtype=torch.float32),
    }


def decode_predictions(pred_indices: torch.Tensor, index_to_token: dict[int, str]) -> list[str]:
    texts = []
    for row in pred_indices.cpu().tolist():
        texts.append("".join(index_to_token[idx] for idx in row))
    return texts


def build_position_mask(index_to_token: dict[int, str], seq_len: int = 10) -> torch.Tensor:
    mask = torch.zeros(seq_len, len(index_to_token), dtype=torch.bool)
    digit_indices = [idx for idx, token in index_to_token.items() if token.isdigit()]
    sign_indices = [idx for idx, token in index_to_token.items() if token in {"+", "-"}]
    dot_indices = [idx for idx, token in index_to_token.items() if token == "."]

    mask[0, sign_indices] = True
    mask[1, digit_indices] = True
    mask[2, digit_indices] = True
    mask[3, dot_indices] = True
    for pos in range(4, seq_len):
        mask[pos, digit_indices] = True
    return mask


def apply_position_mask(logits: torch.Tensor, position_mask: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.clone()
    invalid_mask = ~position_mask.unsqueeze(0).to(logits.device)
    masked_logits = masked_logits.masked_fill(invalid_mask, -1e9)
    return masked_logits


def compute_metrics(records: list[dict]) -> dict:
    token_correct = 0
    token_total = 0
    exact_match = 0
    abs_errors = []
    sq_errors = []

    for record in records:
        pred_text = record["pred_text"]
        target_text = record["target_text"]
        token_correct += sum(1 for p, t in zip(pred_text, target_text) if p == t)
        token_total += len(target_text)
        exact_match += int(pred_text == target_text)

        pred_value = record["pred_value"]
        target_value = record["y_value"]
        if not math.isnan(pred_value):
            abs_errors.append(abs(pred_value - target_value))
            sq_errors.append((pred_value - target_value) ** 2)

    return {
        "token_accuracy": token_correct / max(token_total, 1),
        "exact_match_accuracy": exact_match / max(len(records), 1),
        "mae": float(np.mean(abs_errors)) if abs_errors else float("nan"),
        "mse": float(np.mean(sq_errors)) if sq_errors else float("nan"),
    }


def evaluate(model, dataloader, criterion, device, index_to_token: dict[int, str], position_mask: torch.Tensor) -> tuple[float, dict, list[dict]]:
    model.eval()
    losses = []
    records = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            target_labels = batch["target_labels"].to(device)
            logits = model(input_ids)
            logits = apply_position_mask(logits, position_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_labels.reshape(-1))
            losses.append(loss.item())

            pred_indices = logits.argmax(dim=-1)
            pred_texts = decode_predictions(pred_indices, index_to_token)
            for input_text, target_text, pred_text, x_value, y_value in zip(
                batch["input_texts"],
                batch["target_texts"],
                pred_texts,
                batch["x_values"].tolist(),
                batch["y_values"].tolist(),
            ):
                records.append(
                    {
                        "input_text": input_text,
                        "target_text": target_text,
                        "pred_text": pred_text,
                        "x_value": x_value,
                        "y_value": y_value,
                        "pred_value": safe_float(pred_text),
                    }
                )
    return float(np.mean(losses)), compute_metrics(records), records


def plot_loss_curve(history: list[dict], out_path: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot([item["epoch"] for item in history], [item["train_loss"] for item in history], label="Train")
    plt.plot([item["epoch"] for item in history], [item["id_loss"] for item in history], label="ID")
    plt.plot([item["epoch"] for item in history], [item["ood_loss"] for item in history], label="OOD")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Serialized sin(x) Cross Entropy Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_mse_curve(history: list[dict], out_path: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot([item["epoch"] for item in history], [item["id_metrics"]["mse"] for item in history], label="ID MSE")
    plt.plot([item["epoch"] for item in history], [item["ood_metrics"]["mse"] for item in history], label="OOD MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Serialized sin(x) Numeric MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_predictions(
    id_records: list[dict],
    ood_records: list[dict],
    out_path: str,
    id_left: float,
    id_right: float,
    ood_left: float,
    ood_right: float,
    epoch: int | None = None,
) -> None:
    import matplotlib.pyplot as plt

    all_records = list(id_records) + list(ood_records)
    xs = np.array([item["x_value"] for item in all_records], dtype=np.float64)
    ys = np.array([item["y_value"] for item in all_records], dtype=np.float64)
    pred = np.array([item["pred_value"] for item in all_records], dtype=np.float64)
    order = np.argsort(xs)
    xs, ys, pred = xs[order], ys[order], pred[order]

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.axvspan(ood_left, id_left, color="#f4d7d7", alpha=0.45, label="OOD range")
    ax.axvspan(id_left, id_right, color="#d9ecff", alpha=0.45, label="ID range")
    ax.axvspan(id_right, ood_right, color="#f4d7d7", alpha=0.45)

    ax.axvline(id_left, color="#555555", linestyle="--", linewidth=1)
    ax.axvline(id_right, color="#555555", linestyle="--", linewidth=1)

    ax.plot(xs, ys, color="#1f4e79", linewidth=2.5, label="Ground truth sin(x)")
    ax.plot(xs, pred, color="#c0392b", linewidth=1.8, label="Model prediction")

    ax.text((ood_left + id_left) / 2, 1.15, "OOD", ha="center", va="center", fontsize=11)
    ax.text((id_left + id_right) / 2, 1.15, "ID", ha="center", va="center", fontsize=11)
    ax.text((id_right + ood_right) / 2, 1.15, "OOD", ha="center", va="center", fontsize=11)

    ax.set_xlim(ood_left, ood_right)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = "FANformer on Serialized sin(x)"
    if epoch is not None:
        title = f"{title} (epoch {epoch})"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def append_timestamp(output_dir: str) -> str:
    normalized = output_dir.rstrip("/\\")
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    return f"{normalized}_{timestamp}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Appendix H: serialized sin(x) with Qwen2.5Embedding backbones")
    parser.add_argument("--output_dir", type=str, default="./outputs/appendix_h_fanformer_sin")
    parser.add_argument("--pretrained_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen2.5Embedding-FANformer",
        choices=["Qwen2.5Embedding-FANformer", "Qwen2.5Embedding-Transformer"],
    )
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument(
        "--norm_first",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=True,
        help="Whether to use pre-norm in Transformer/FANformer blocks.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--train_size", type=int, default=12000)
    parser.add_argument("--id_test_size", type=int, default=2000)
    parser.add_argument("--ood_test_size", type=int, default=2000)
    parser.add_argument("--id_left", type=float, default=-3 * math.pi)
    parser.add_argument("--id_right", type=float, default=3 * math.pi)
    parser.add_argument("--ood_left", type=float, default=-6 * math.pi)
    parser.add_argument("--ood_right", type=float, default=6 * math.pi)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--plot_every", type=int, default=0, help="Save prediction curve every N epochs. 0 disables intermediate plots.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.plot_every < 0:
        raise ValueError("--plot_every must be >= 0.")

    set_seed(args.seed)
    args.output_dir = append_timestamp(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = load_tokenizer(args.pretrained_name)
    allowed_tokens = ["+", "-", "."] + [str(i) for i in range(10)]
    allowed_token_ids = tokenizer.convert_tokens_to_ids(allowed_tokens)
    if any(token_id is None or token_id < 0 for token_id in allowed_token_ids):
        raise ValueError("Failed to resolve all output tokens in Qwen tokenizer.")
    label_to_index = {token_id: idx for idx, token_id in enumerate(allowed_token_ids)}
    index_to_token = {idx: token for idx, token in enumerate(allowed_tokens)}
    position_mask = build_position_mask(index_to_token)

    train_samples, id_samples, ood_samples = build_datasets(
        args.train_size,
        args.id_test_size,
        args.ood_test_size,
        args.id_left,
        args.id_right,
        args.ood_left,
        args.ood_right,
    )
    train_dataset = SerializedSinDataset(train_samples, tokenizer, label_to_index)
    id_dataset = SerializedSinDataset(id_samples, tokenizer, label_to_index)
    ood_dataset = SerializedSinDataset(ood_samples, tokenizer, label_to_index)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    id_loader = DataLoader(id_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    ood_loader = DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model = get_model_by_name(
        args.model_name,
        pretrained_name=args.pretrained_name,
        output_dim=len(allowed_tokens),
        num_layers=args.layers,
        num_heads=args.num_heads,
        norm_first=args.norm_first,
        freeze_emb=True,
        causal=False,
    ).to(args.device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    final_payload = None

    config = vars(args).copy()
    config["allowed_tokens"] = allowed_tokens
    save_json(os.path.join(args.output_dir, "config.json"), config)
    save_json(
        os.path.join(args.output_dir, "dataset_preview.json"),
        {
            "train_samples": [train_samples[i].__dict__ for i in range(min(3, len(train_samples)))],
            "id_samples": [id_samples[i].__dict__ for i in range(min(3, len(id_samples)))],
            "ood_samples": [ood_samples[i].__dict__ for i in range(min(3, len(ood_samples)))],
        },
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            input_ids = batch["input_ids"].to(args.device)
            target_labels = batch["target_labels"].to(args.device)
            logits = model(input_ids)
            logits = apply_position_mask(logits, position_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_labels.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        should_eval = (
            epoch % args.eval_every == 0
            or (args.plot_every > 0 and epoch % args.plot_every == 0)
            or epoch == args.epochs
        )
        if not should_eval:
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}")
            continue

        id_loss, id_metrics, id_records = evaluate(model, id_loader, criterion, args.device, index_to_token, position_mask)
        ood_loss, ood_metrics, ood_records = evaluate(model, ood_loader, criterion, args.device, index_to_token, position_mask)
        history_item = {
            "epoch": epoch,
            "train_loss": train_loss,
            "id_loss": id_loss,
            "ood_loss": ood_loss,
            "id_metrics": id_metrics,
            "ood_metrics": ood_metrics,
        }
        history.append(history_item)
        print(
            f"Epoch {epoch}: train={train_loss:.6f} "
            f"id_mse={id_metrics['mse']:.6f} ood_mse={ood_metrics['mse']:.6f} "
            f"id_exact={id_metrics['exact_match_accuracy']:.4f} ood_exact={ood_metrics['exact_match_accuracy']:.4f}"
        )
        final_payload = {
            "epoch": epoch,
            "id_records": id_records,
            "ood_records": ood_records,
            "id_metrics": id_metrics,
            "ood_metrics": ood_metrics,
        }
        if (args.plot_every > 0 and epoch % args.plot_every == 0) or epoch == args.epochs:
            plot_predictions(
                id_records,
                ood_records,
                os.path.join(args.output_dir, f"prediction_curve_epoch{epoch}.png"),
                args.id_left,
                args.id_right,
                args.ood_left,
                args.ood_right,
                epoch=epoch,
            )

    if not history:
        id_loss, id_metrics, id_records = evaluate(model, id_loader, criterion, args.device, index_to_token, position_mask)
        ood_loss, ood_metrics, ood_records = evaluate(model, ood_loader, criterion, args.device, index_to_token, position_mask)
        history.append(
            {
                "epoch": args.epochs,
                "train_loss": train_loss,
                "id_loss": id_loss,
                "ood_loss": ood_loss,
                "id_metrics": id_metrics,
                "ood_metrics": ood_metrics,
            }
        )
        final_payload = {
            "epoch": args.epochs,
            "id_records": id_records,
            "ood_records": ood_records,
            "id_metrics": id_metrics,
            "ood_metrics": ood_metrics,
        }

    save_json(os.path.join(args.output_dir, "metrics_history.json"), history)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "last_model.pt"))
    save_json(
        os.path.join(args.output_dir, "last_eval.json"),
        {
            "epoch": final_payload["epoch"],
            "id_metrics": final_payload["id_metrics"],
            "ood_metrics": final_payload["ood_metrics"],
            "id_records_preview": final_payload["id_records"][:50],
            "ood_records_preview": final_payload["ood_records"][:50],
        },
    )
    plot_loss_curve(history, os.path.join(args.output_dir, "cross_entropy_loss_curve.png"))
    plot_mse_curve(history, os.path.join(args.output_dir, "mse_curve.png"))
    plot_predictions(
        final_payload["id_records"],
        final_payload["ood_records"],
        os.path.join(args.output_dir, "prediction_curve.png"),
        args.id_left,
        args.id_right,
        args.ood_left,
        args.ood_right,
        epoch=final_payload["epoch"],
    )


if __name__ == "__main__":
    main()
