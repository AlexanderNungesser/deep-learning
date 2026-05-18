import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    TensorBoard,
    EarlyStopping,
    ReduceLROnPlateau,
)
 
from generator import RAMDataGenerator
 
 
# ---------------------------------------------------------------------------
# Einzelner Fold
# ---------------------------------------------------------------------------
 
def train_fold(
    fold_idx: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    images_raw: np.ndarray,
    labels: np.ndarray,
    logs_dir: str,
    model_factory: callable,
    batch_size: int = 16,
    epochs: int = 100,
    early_stopping_patience: int = 8,
    reduce_lr_patience: int = 3,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Baut ueber `model_factory` ein frisches Modell und trainiert es auf einem Fold.
 
    Warum `model_factory` statt eines fertigen Modell-Objekts?
    Weil jeder Fold mit komplett zurueckgesetzten Gewichten starten muss.
    Eine factory ist eine Funktion, die bei jedem Aufruf ein neues Modell
    erzeugt — so wird garantiert, dass kein Fold Wissen aus dem vorherigen erbt.
 
    Parameters
    ----------
    fold_idx              : int        — 0-basierter Index (fuer Dateinamen)
    train_indices         : np.ndarray — Absolute Indizes des Trainingssets
    val_indices           : np.ndarray — Absolute Indizes des Validationssets
    images_raw            : np.ndarray — Alle Rohbilder (uint8) im RAM
    labels                : np.ndarray — Alle Klassenindizes
    logs_dir              : str        — Pfad fuer Checkpoints und TensorBoard
    model_factory         : callable   — Funktion ohne Argumente, die ein
                                         frisches kompiliertes Modell liefert
    batch_size            : int
    epochs                : int        — Maximale Trainingsepochen
    early_stopping_patience : int      — Epochen ohne Verbesserung bis Abbruch
    reduce_lr_patience    : int        — Epochen ohne Verbesserung bis LR-Halbierung
 
    Returns
    -------
    fold_model : tf.keras.Model
    history    : tf.keras.callbacks.History
    """
    # Frisches Modell fuer diesen Fold — jeder Fold startet bei Null
    fold_model = model_factory()
 
    train_gen = RAMDataGenerator(images_raw, labels, train_indices, batch_size=batch_size)
    val_gen   = RAMDataGenerator(images_raw, labels, val_indices,   batch_size=batch_size)
 
    callbacks = [
        ModelCheckpoint(
            filepath=f"{logs_dir}/checkpoint_fold{fold_idx + 1}.keras",
            save_best_only=True,
            monitor="val_accuracy",
            verbose=0,
        ),
        TensorBoard(
            log_dir=f"{logs_dir}/fold{fold_idx + 1}",
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
 
    history = fold_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
    )
 
    return fold_model, history
 
 
# ---------------------------------------------------------------------------
# Trainingsschleife ueber alle Folds
# ---------------------------------------------------------------------------
 
def run_training(
    folds: list[tuple[np.ndarray, np.ndarray]],
    images_raw: np.ndarray,
    labels: np.ndarray,
    logs_dir: str,
    model_factory: callable,
    batch_size: int = 16,
    epochs: int = 100,
    early_stopping_patience: int = 8,
    reduce_lr_patience: int = 3,
    load_models: bool = False,
) -> tuple[list[tf.keras.Model], list]:
    """
    Iteriert ueber alle Folds und trainiert bzw. laedt ein Modell pro Fold.
 
    Die Schleife ist vollstaendig blind gegenueber USE_KFOLD: sie iteriert
    einfach ueber `folds`, egal ob die Liste 1 oder k Eintraege hat.
 
    Parameters
    ----------
    folds         : list  — Ausgabe von prepare_splits() aus src/data.py
    images_raw    : np.ndarray
    labels        : np.ndarray
    logs_dir      : str         — Pfad fuer Checkpoints und TensorBoard-Logs
    model_factory : callable    — Funktion ohne Argumente, liefert frisches Modell.
                                  Wird pro Fold einmal aufgerufen.
                                  Beim Laden (load_models=True) wird sie nicht
                                  benoetigt, muss aber trotzdem uebergeben werden.
    batch_size    : int
    epochs        : int
    early_stopping_patience : int
    reduce_lr_patience      : int
    load_models   : bool  — True = lade Checkpoint falls vorhanden,
                            False = trainiere immer neu
 
    Returns
    -------
    fold_models    : list[tf.keras.Model]
    fold_histories : list[History | None]  — None bei geladenen Modellen
    """
    os.makedirs(logs_dir, exist_ok=True)
 
    fold_models    = []
    fold_histories = []
    total          = len(folds)
 
    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        checkpoint_path = f"{logs_dir}/checkpoint_fold{fold_idx + 1}.keras"
 
        print(f"\n{'=' * 58}")
 
        if load_models and os.path.exists(checkpoint_path):
            print(f"Fold {fold_idx + 1} / {total}: Lade gespeichertes Modell")
            model   = tf.keras.models.load_model(checkpoint_path)
            history = None
            print(f"  Checkpoint geladen: {checkpoint_path}")
        else:
            print(f"Fold {fold_idx + 1} / {total}: Starte Training")
            model, history = train_fold(
                fold_idx=fold_idx,
                train_indices=train_indices,
                val_indices=val_indices,
                images_raw=images_raw,
                labels=labels,
                logs_dir=logs_dir,
                model_factory=model_factory,
                batch_size=batch_size,
                epochs=epochs,
                early_stopping_patience=early_stopping_patience,
                reduce_lr_patience=reduce_lr_patience,
            )
 
        fold_models.append(model)
        fold_histories.append(history)
 
    print(f"\n{'=' * 58}")
    print(f"Alle {total} Fold(s) abgeschlossen.")
 
    return fold_models, fold_histories