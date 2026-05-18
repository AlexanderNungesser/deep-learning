from tensorflow.keras.utils import Sequence, to_categorical
import math
import numpy as np
from data import NUM_CLASSES

class RAMDataGenerator(Sequence):
    """
    Keras-kompatibler Batch-Generator, der Bilder aus einem vorgeladenen
    NumPy-Array holt und on-the-fly normalisiert.
 
    Das Kernprinzip: Wir halten die Rohdaten als uint8 im RAM (platzsparend)
    und wandeln *nur* den aktuellen Batch in float32 um, bevor er ans Modell
    geht. Das spart gegenüber einer vollständigen float32-Kopie des Datasets
    rund 75 % RAM — bei 256×256×3 Bildern ist das ein erheblicher Unterschied.
 
    Parameters
    ----------
    images_array : np.ndarray
        Alle Rohbilder als uint8, direkt aus load_data().
    labels_array : np.ndarray
        Alle Klassenindizes als int, direkt aus load_data().
    indices : np.ndarray
        Die Indizes der Samples, die dieser Generator bedienen soll.
        Damit lassen sich Train/Val/Test-Splits sauber abbilden, ohne
        die Daten physisch zu duplizieren.
    batch_size : int
        Anzahl der Bilder pro Batch. Default: 32.
    num_classes : int
        Anzahl der Zielklassen für das One-Hot-Encoding. Default: 10.
 
    Example
    -------
    >>> train_gen = RAMDataGenerator(images_raw, labels, train_indices, batch_size=16)
    >>> x_batch, y_batch = train_gen[0]   # x: float32 in [0,1], y: One-Hot
    """
 
    def __init__(
        self,
        images_array: np.ndarray,
        labels_array: np.ndarray,
        indices: np.ndarray,
        batch_size: int = 32,
        num_classes: int = NUM_CLASSES,
    ):
        self.images = images_array
        self.labels = labels_array
        self.indices = indices
        self.batch_size = batch_size
        self.num_classes = num_classes
 
    def __len__(self) -> int:
        # math.ceil stellt sicher, dass auch der letzte (möglicherweise kleinere)
        # Batch noch mitgenommen wird
        return math.ceil(len(self.indices) / self.batch_size)
 
    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        # Welche absoluten Bild-Indizes gehören zu diesem Batch?
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
 
        # Bilder aus dem RAM holen (uint8) — dieser Schritt ist sehr schnell,
        # da alles bereits im Arbeitsspeicher liegt
        batch_x = self.images[batch_indices]
        batch_y = self.labels[batch_indices]
 
        # Normalisierung und One-Hot-Encoding nur für diesen kleinen Batch:
        # float32-Cast + Division kostet kaum RAM, da wir nur ~16-32 Bilder anfassen
        batch_x = batch_x.astype(np.float32) / 255.0
        batch_y = to_categorical(batch_y, self.num_classes)
 
        return batch_x, batch_y
 