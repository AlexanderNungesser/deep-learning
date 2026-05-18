"""Erzeuge Trainingskurven-Plots pro Architektur aus TensorBoard-Logs.

Pro Modell wird ein Plot mit zwei Subplots gespeichert:
- links: Accuracy (Training + Validation)
- rechts: Loss (Training + Validation)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
RANKING_CSV = Path(__file__).resolve().parent / "architecture_ranking.csv"
LOGS_ROOT = WORKSPACE_ROOT / "logs"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def _extract_series(log_dir: Path, tag_candidates: Iterable[str]) -> tuple[list[int], list[float]]:
    """Liest einen Zeitverlauf fuer den ersten verfuegbaren Tag aus einem TB-Logordner."""
    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    tags = accumulator.Tags()

    scalar_tags = set(tags.get("scalars", []))
    tensor_tags = set(tags.get("tensors", []))

    for tag in tag_candidates:
        if tag in scalar_tags:
            events = accumulator.Scalars(tag)
            epochs = [event.step + 1 for event in events]
            values = [float(event.value) for event in events]
            return epochs, values

        if tag in tensor_tags:
            events = accumulator.Tensors(tag)
            epochs = [event.step + 1 for event in events]
            values = [float(tf.make_ndarray(event.tensor_proto).reshape(-1)[0]) for event in events]
            return epochs, values

    raise ValueError(f"Keiner der erwarteten Tags gefunden in {log_dir}: {list(tag_candidates)}")


def _plot_single_model(rank: int, variant: str, model_name: str, fold: int) -> Path:
    train_dir = LOGS_ROOT / model_name / f"fold{fold}" / "train"
    val_dir = LOGS_ROOT / model_name / f"fold{fold}" / "validation"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Fehlende TensorBoard-Ordner fuer {model_name}: {train_dir} / {val_dir}")

    train_acc_x, train_acc = _extract_series(train_dir, ["epoch_accuracy", "accuracy"])
    val_acc_x, val_acc = _extract_series(val_dir, ["epoch_accuracy", "accuracy", "evaluation_accuracy_vs_iterations"])

    train_loss_x, train_loss = _extract_series(train_dir, ["epoch_loss", "loss"])
    val_loss_x, val_loss = _extract_series(val_dir, ["epoch_loss", "loss", "evaluation_loss_vs_iterations"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

    axes[0].plot(train_acc_x, train_acc, label="Train", linewidth=2)
    axes[0].plot(val_acc_x, val_acc, label="Validation", linewidth=2)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(train_loss_x, train_loss, label="Train", linewidth=2)
    axes[1].plot(val_loss_x, val_loss, label="Validation", linewidth=2)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"Rank {rank} | {variant} | {model_name}", fontsize=12)
    fig.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_DIR / f"rank_{rank:02d}_{variant}_{model_name}.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path


def main() -> None:
    if not RANKING_CSV.exists():
        raise FileNotFoundError(f"Ranking-Datei nicht gefunden: {RANKING_CSV}")

    df = pd.read_csv(RANKING_CSV)
    generated = []

    for _, row in df.iterrows():
        rank = int(row["Rang"])
        variant = str(row["Variante"])
        model_name = str(row["Modell"])
        fold = int(row["Fold mit Best Val"])

        output_path = _plot_single_model(rank, variant, model_name, fold)
        generated.append(output_path)
        print(f"[OK] {model_name} -> {output_path}")

    print(f"\nFertig. {len(generated)} Plot(s) gespeichert in: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
