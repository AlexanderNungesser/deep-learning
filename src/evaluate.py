import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
 
from data import LABEL_DICT, NUM_CLASSES
from generator import RAMDataGenerator
 
 
# ---------------------------------------------------------------------------
# Test-Evaluierung
# ---------------------------------------------------------------------------
 
def evaluate_models(
    fold_models: list[tf.keras.Model],
    images_raw: np.ndarray,
    labels: np.ndarray,
    X_test: np.ndarray,
    batch_size: int = 16,
) -> tuple[list[float], tf.keras.Model, int]:
    """
    Evaluiert alle Fold-Modelle auf dem gemeinsamen Test-Set.
 
    Parameters
    ----------
    fold_models : list[tf.keras.Model]  — Ausgabe von run_training()
    images_raw  : np.ndarray
    labels      : np.ndarray
    X_test      : np.ndarray            — Test-Indizes aus prepare_splits()
    batch_size  : int
 
    Returns
    -------
    test_accuracies : list[float]       — Accuracy je Modell
    best_model      : tf.keras.Model    — Modell mit hoechster Test-Accuracy
    best_fold_idx   : int               — 0-basierter Index des besten Modells
    """
    test_gen = RAMDataGenerator(images_raw, labels, X_test, batch_size=batch_size)
 
    test_accuracies = []
 
    print("Starte Evaluierung auf dem Test-Set...")
 
    for i, model in enumerate(fold_models):
        print(f"\n  Fold {i + 1} / {len(fold_models)}:")
        loss, acc = model.evaluate(test_gen, verbose=1)
        test_accuracies.append(acc)
        print(f"  --> Test Accuracy: {acc:.4f} ({acc * 100:.2f} %)")
 
    best_fold_idx = int(np.argmax(test_accuracies))
    best_model    = fold_models[best_fold_idx]
 
    print(f"\n{'=' * 58}")
    print(f"Durchschnittliche Test Accuracy : {np.mean(test_accuracies):.4f}")
    print(f"Standardabweichung              : {np.std(test_accuracies):.4f}")
    print(f"Bestes Modell                   : Fold {best_fold_idx + 1}")
    print(f"  Test Accuracy                 : {test_accuracies[best_fold_idx]:.4f} "
          f"({test_accuracies[best_fold_idx] * 100:.2f} %)")
    print(f"{'=' * 58}")
 
    return test_accuracies, best_model, best_fold_idx
 
 
# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------
 
def plot_confusion_matrix(
    best_model: tf.keras.Model,
    images_raw: np.ndarray,
    labels: np.ndarray,
    X_test: np.ndarray,
    best_fold_idx: int,
    batch_size: int = 16,
    save_path: str | None = None,
) -> None:
    """
    Erstellt und zeigt die Confusion Matrix des besten Modells.
 
    Parameters
    ----------
    best_model    : tf.keras.Model  — Ausgabe von evaluate_models()
    images_raw    : np.ndarray
    labels        : np.ndarray
    X_test        : np.ndarray      — Test-Indizes aus prepare_splits()
    best_fold_idx : int             — fuer den Diagrammtitel
    batch_size    : int
    save_path     : str | None      — wenn angegeben, wird die Figure gespeichert
                                      z.B. "./../reports/milestone-1/confusion_matrix.png"
    """
    test_gen = RAMDataGenerator(images_raw, labels, X_test, batch_size=batch_size)
 
    predictions = best_model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = labels[X_test]
 
    cm = confusion_matrix(y_true, y_pred)
    class_names = [LABEL_DICT[i] for i in range(NUM_CLASSES)]
 
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
 
    plt.title(f"Confusion Matrix — Bestes Modell (Fold {best_fold_idx + 1})")
    plt.ylabel("Echte Klasse")
    plt.xlabel("Vorhergesagte Klasse")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
 
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Confusion Matrix gespeichert: {save_path}")
 
    plt.show()
 
 
# ---------------------------------------------------------------------------
# Trainings-History
# ---------------------------------------------------------------------------
 
def plot_training_history(
    fold_histories: list[tf.keras.callbacks.History | None],
    save_dir: str | None = None,
) -> None:
    """
    Plottet Trainings- und Validierungs-Accuracy sowie Loss-Kurven 
    fuer alle Folds in einem gemeinsamen Bild.
 
    Folds, bei denen das Modell geladen wurde (history == None),
    werden automatisch uebersprungen.
 
    Parameters
    ----------
    fold_histories : list       — Ausgabe von run_training()
    save_dir       : str | None — Ordner zum Speichern des Plots
    """
    # Nur Folds mit echten History-Objekten
    valid = [(i, h) for i, h in enumerate(fold_histories) if h is not None]
 
    if not valid:
        print("Keine Trainings-History verfuegbar (alle Modelle wurden geladen).")
        return
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
    for fold_idx, history in valid:
        # Fallback, falls die Metrik 'acc' statt 'accuracy' heißt
        acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
        val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'

        ### Subplot 1: Accuracy ###
        axes[0].plot(
            history.history[acc_key], 
            linestyle='-', 
            label=f"Fold {fold_idx + 1} Train"
        )
        axes[0].plot(
            history.history[val_acc_key], 
            linestyle='-', 
            label=f"Fold {fold_idx + 1} Val"
        )
 
        ### Subplot 2: Loss ###
        axes[1].plot(
            history.history["loss"], 
            linestyle='-', 
            label=f"Fold {fold_idx + 1} Train"
        )
        axes[1].plot(
            history.history["val_loss"],     
            linestyle='-', 
            label=f"Fold {fold_idx + 1} Val"
        )
 
    # Titel und Achsenbeschriftungen anpassen
    axes[0].set_title("Accuracy pro Fold (Train vs. Val)")
    axes[0].set_xlabel("Epoche")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
 
    axes[1].set_title("Loss pro Fold (Train vs. Val)")
    axes[1].set_xlabel("Epoche")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
 
    plt.tight_layout()
 
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "training_history.png")
        plt.savefig(path, dpi=150)
        print(f"Training History gespeichert: {path}")
 
    plt.show()
 