import h5py
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

LABEL_DICT = {
    0: "Disturbed Galaxies",
    1: "Merging Galaxies",
    2: "Round Smooth Galaxies",
    3: "In-between Round Smooth Galaxies",
    4: "Cigar Shaped Smooth Galaxies",
    5: "Barred Spiral Galaxies",
    6: "Unbarred Tight Spiral Galaxies",
    7: "Unbarred Loose Spiral Galaxies",
    8: "Edge-on Galaxies without Bulge",
    9: "Edge-on Galaxies with Bulge",
}
 
NUM_CLASSES = len(LABEL_DICT)

INPUT_SHAPE = (256, 256, 3)

def load_data(data_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Lädt das komplette Galaxy10 DECals Dataset aus einer HDF5-Datei in den RAM.
 
    Parameters
    ----------
    data_path : str
        Pfad zur Galaxy10_DECals.h5-Datei, z.B. "./../data/Galaxy10_DECals.h5"
 
    Returns
    -------
    images_raw : np.ndarray, shape (N, 256, 256, 3), dtype uint8
        Rohe Pixelwerte im Bereich [0, 255]. Bewusst uint8, um RAM zu sparen —
        die Skalierung auf [0.0, 1.0] passiert erst im Generator, Batch für Batch.
    labels : np.ndarray, shape (N,), dtype int
        Klassenindizes im Bereich [0, 9], noch *kein* One-Hot-Encoding.
    """
    with h5py.File(data_path, "r") as f:
        images_raw = np.array(f["images"])
        labels = np.array(f["ans"])
 
    print(f"Dataset geladen: {len(labels):,} Bilder, Shape {images_raw.shape}")
    return images_raw, labels

def print_class_distribution(labels: np.ndarray) -> dict[int, int]:
    """
    Gibt eine Tabelle der Klassenverteilung aus und liefert sie als Dict zurück.
 
    Parameters
    ----------
    labels : np.ndarray
        Array der Klassenindizes (wie von load_data() zurückgegeben).
 
    Returns
    -------
    counts : dict[int, int]
        Mapping von Klassenindex → Anzahl der Samples.
    """
    counts = {}
 
    print(f"\n{'Klasse':<8} {'Beschreibung':<40} {'Anzahl':>7}")
    print("-" * 58)
 
    for cls_id, cls_name in LABEL_DICT.items():
        count = int(np.sum(labels == cls_id))
        counts[cls_id] = count
        print(f"{cls_id:<8} {cls_name:<40} {count:>7}")
 
    print("-" * 58)
    print(f"{'GESAMT':<48} {len(labels):>7}")
 
    return counts

def show_class_distribution(counts: dict[int, int]) -> None:
    """
    Zeigt die Klassenverteilung als Balkendiagramm an.
 
    Parameters
    ----------
    counts : dict[int, int]
        Mapping von Klassenindex → Anzahl der Samples, z.B. von print_class_distribution().
    """
    import matplotlib.pyplot as plt
 
    classes = list(counts.keys())
    frequencies = list(counts.values())
 
    plt.figure(figsize=(12, 6))
    plt.bar(classes, frequencies, color="skyblue")
    plt.xticks(classes, [LABEL_DICT[c] for c in classes], rotation=45, ha="right")
    plt.xlabel("Klassen")
    plt.ylabel("Anzahl der Bilder")
    plt.title("Klassenverteilung im Galaxy10 DECals Dataset")
    plt.tight_layout()
    plt.show()

def prepare_splits(
    labels: np.ndarray,
    use_kfold: bool = True,
    k: int = 5,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """
    Teilt den Datensatz in einen fixen Test-Split und Train/Val-Folds auf.
 
    Das Test-Set wird *immer* einmalig stratifiziert abgetrennt und bleibt
    für alle Folds identisch — so sind die Ergebnisse direkt vergleichbar.
 
    Anschließend gibt es zwei Modi, gesteuert über `use_kfold`:
 
    USE_KFOLD = True  → StratifiedKFold mit k Folds
                        Jeder Fold nutzt einen anderen Teil als Validation.
                        Ideal, um die Modellstabilität zu messen und
                        sicherzustellen, dass das Ergebnis nicht vom
                        zufälligen Split abhängt.
 
    USE_KFOLD = False → Einfacher stratifizierter Train/Val-Split
                        Schneller, gut für schnelle Experimente oder wenn
                        Rechenzeit knapp ist. Liefert nur einen einzigen Fold.
 
    Beide Modi liefern *dieselbe Datenstruktur* zurück (Liste von Fold-Tuples),
    sodass train.py keinerlei Fallunterscheidung braucht — es iteriert immer
    über dieselbe Liste, die bei use_kfold=False eben nur ein Element hat.
 
    Parameters
    ----------
    labels       : np.ndarray  — Klassenindizes aus load_data()
    use_kfold    : bool        — True = k-fache Kreuzvalidierung, False = einfacher Split
    k            : int         — Anzahl der Folds (nur relevant wenn use_kfold=True)
    val_size     : float       — Anteil Validation (nur relevant wenn use_kfold=False)
    test_size    : float       — Anteil Test-Set (immer aktiv)
    random_state : int         — Seed für Reproduzierbarkeit
 
    Returns
    -------
    folds  : list[tuple[np.ndarray, np.ndarray]]
        Liste von (train_indices, val_indices)-Paaren.
        Laenge k bei use_kfold=True, Laenge 1 bei use_kfold=False.
    X_test : np.ndarray  — Absolute Indizes des Test-Sets
    y_test : np.ndarray  — Zugehoerige Labels des Test-Sets
    """
    all_indices = np.arange(len(labels))
 
    # Test-Set einmalig und fix abtrennen
    X_temp, X_test, y_temp, y_test = train_test_split(
        all_indices,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )
 
    print(f"\nSplit-Modus : {'K-Fold (k=' + str(k) + ')' if use_kfold else 'Einfacher Train/Val-Split'}")
    print(f"Train + Val : {len(X_temp):,} Samples")
    print(f"Test        : {len(X_test):,} Samples")
 
    folds = []
 
    if use_kfold:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
 
        for fold_idx, (train_pos, val_pos) in enumerate(skf.split(X_temp, y_temp)):
            # train_pos / val_pos sind Positionen innerhalb von X_temp,
            # keine absoluten Indizes — daher die Umrechnung:
            train_indices = X_temp[train_pos]
            val_indices = X_temp[val_pos]
            folds.append((train_indices, val_indices))
            print(f"  Fold {fold_idx + 1}: Train {len(train_indices):,} | Val {len(val_indices):,}")
    else:
        # val_size bezieht sich auf den Train+Val-Block
        X_train, X_val, _, _ = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size,
            stratify=y_temp,
            random_state=random_state,
        )
        folds.append((X_train, X_val))
        print(f"  Train {len(X_train):,} | Val {len(X_val):,}")
 
    return folds, X_test, y_test