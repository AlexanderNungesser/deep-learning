"""
models_kd.py — Student-Architekturen für Knowledge Distillation (Station 3)

WICHTIG: Alle Modelle hier verwenden KEINEN Softmax im letzten Dense-Layer.
Der Distiller in milestone-3-part2-student.ipynb erwartet rohe Logits, weil:
  1. SparseCategoricalCrossentropy(from_logits=True) → direkter Vergleich mit Labels
  2. tf.nn.softmax(logits / T) → manuelles Temperature-Scaling für KL-Divergenz

Alle Modelle erfüllen die Anforderung aus Station 3:
  "Kleines Modell = weniger Parameter als Modelle aus Station 1 & 2"
  → Limit: < 2.5 Mio. Parameter (bestes Modell Station 1: 2.49M)

Parameterprogression (verifiziert via .count_params()):
  S1_NANO  :   64k  ← Sehr kleines Baseline-Modell
  S2_SMALL :  251k
  S3_MEDIUM:  619k
  S4_BASE  : 1.10M
  S5_LARGE : 1.24M
  S6_XL    : 1.84M
  S7_MAX   : 2.03M  ← Größtes erlaubtes Student-Modell
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    BatchNormalization,
    Dropout,
    Dense,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomTranslation,
)
from tensorflow.keras.optimizers import Adam

from data import NUM_CLASSES, INPUT_SHAPE


# ===========================================================================
# STUDENT S1 — NANO  (~64k Parameter)
# Minimalste Variante: 4 Blöcke, kleine Filter (16→32→64→64)
# Baseline für: "Wie viel kann ein winziges Modell durch KD lernen?"
# ===========================================================================
STUDENT_S1_NANO = Sequential([
    # Block 1: 256×256 → 128×128
    Conv2D(16, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    # Block 2: 128×128 → 64×64
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    # Block 3: 64×64 → 32×32
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    # Block 4: 32×32 → 16×16
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    GlobalAveragePooling2D(),
    Dense(32, activation='relu'),
    Dense(NUM_CLASSES),  # Logits — kein Softmax!
], name='student_s1_nano')


# ===========================================================================
# STUDENT S2 — SMALL  (~251k Parameter)
# 4 Blöcke, größere Filter (32→64→128→128)
# ===========================================================================
STUDENT_S2_SMALL = Sequential([
    # Block 1: 256×256 → 128×128
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    # Block 2: 128×128 → 64×64
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    # Block 3: 64×64 → 32×32
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    # Block 4: 32×32 → 16×16
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    GlobalAveragePooling2D(),
    Dense(64, activation='relu'),
    Dense(NUM_CLASSES),  # Logits
], name='student_s2_small')


# ===========================================================================
# STUDENT S3 — MEDIUM  (~619k Parameter)
# 4 Blöcke (erste 3 mit Double-Conv), bis 256 Filter
# Vergleichbar mit V3/V5 aus Station 1 in der Parameterklasse
# ===========================================================================
STUDENT_S3_MEDIUM = Sequential([
    # Block 1: Double Conv, 256×256 → 128×128
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    # Block 2: Double Conv, 128×128 → 64×64
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    # Block 3: Double Conv, 64×64 → 32×32
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    # Block 4: Single Conv, 32×32 → 16×16
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(NUM_CLASSES),  # Logits
], name='student_s3_medium')


# ===========================================================================
# STUDENT S4 — BASE  (~1.10M Parameter)
# 5 Blöcke (erste 2 mit Double-Conv), bis 256 Filter
# Erster Student der >1M Parameterklasse
# ===========================================================================
STUDENT_S4_BASE = Sequential([
    # Block 1: Double Conv, 256×256 → 128×128
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    # Block 2: Double Conv, 128×128 → 64×64
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    # Block 3: Single Conv, 64×64 → 32×32
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    # Block 4: Single Conv, 32×32 → 16×16
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    # Block 5: Single Conv, 16×16 → 8×8
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(NUM_CLASSES),  # Logits
], name='student_s4_base')


# ===========================================================================
# STUDENT S5 — LARGE  (~1.24M Parameter)
# 5 Blöcke (erste 3 mit Double-Conv) + Dropout
# Wie S4, aber Block 3 mit Double-Conv für mehr Repräsentationstiefe
# ===========================================================================
STUDENT_S5_LARGE = Sequential([
    # Block 1: Double Conv, 256×256 → 128×128
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    # Block 2: Double Conv, 128×128 → 64×64
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    # Block 3: Double Conv, 64×64 → 32×32
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    # Block 4: Single Conv, 32×32 → 16×16
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    # Block 5: Single Conv, 16×16 → 8×8
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(NUM_CLASSES),  # Logits
], name='student_s5_large')


# ===========================================================================
# STUDENT S6 — XL  (~1.84M Parameter)
# 5 Blöcke (erste 4 mit Double-Conv) + Dropout
# Erste 4 Blöcke verdoppelt → viel mehr Feature-Tiefe
# ===========================================================================
STUDENT_S6_XL = Sequential([
    # Block 1: Double Conv, 256×256 → 128×128
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    # Block 2: Double Conv, 128×128 → 64×64
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    # Block 3: Double Conv, 64×64 → 32×32
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    # Block 4: Double Conv, 32×32 → 16×16
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    # Block 5: Single Conv, 16×16 → 8×8
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(NUM_CLASSES),  # Logits
], name='student_s6_xl')


# ===========================================================================
# STUDENT S7 — MAX  (~2.03M Parameter)
# Wie S6, aber mit tieferem Dense-Kopf (512→256)
# Größtmöglicher Student unter dem 2.5M-Limit
# ===========================================================================
STUDENT_S7_MAX = Sequential([
    # Block 1: Double Conv, 256×256 → 128×128
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    # Block 2: Double Conv, 128×128 → 64×64
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    # Block 3: Double Conv, 64×64 → 32×32
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    # Block 4: Double Conv, 32×32 → 16×16
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    # Block 5: Single Conv, 16×16 → 8×8
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),

    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(NUM_CLASSES),  # Logits
], name='student_s7_max')


# ===========================================================================
# REGISTRY — alle Studenten als geordnetes Dict für die Trainingsschleife
# Format: { "Anzeigename": (Modell-Objekt, "~Parameteranzahl") }
# ===========================================================================
STUDENT_VARIANTS: dict[str, tuple[Sequential, str]] = {
    "S1_Nano":   (STUDENT_S1_NANO,   "~64k"),
    "S2_Small":  (STUDENT_S2_SMALL,  "~251k"),
    "S3_Medium": (STUDENT_S3_MEDIUM, "~619k"),
    "S4_Base":   (STUDENT_S4_BASE,   "~1.10M"),
    "S5_Large":  (STUDENT_S5_LARGE,  "~1.24M"),
    "S6_XL":     (STUDENT_S6_XL,     "~1.84M"),
    "S7_Max":    (STUDENT_S7_MAX,     "~2.03M"),
}


# ===========================================================================
# BUILD-FUNKTION — analog zu build_model() in models.py, aber für KD
# Unterschiede zu build_model():
#   - Kein Softmax am Output (Logits für Distiller)
#   - input_shape wird nicht aus dem Body-Modell inferiert,
#     sondern explizit übergeben
#   - compile() mit from_logits=True (wird vom Distiller sowieso überschrieben,
#     ist aber sinnvoll für standalone-Tests ohne Distiller)
# ===========================================================================
def build_student_model(
    student_body: Sequential,
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Baut ein Student-Modell mit Augmentierung für Knowledge Distillation.

    Der Aufbau ist identisch zu build_model() aus models.py — dieselbe
    Augmentierungspipeline — aber es wird KEIN Softmax ergänzt.
    Die Ausgabe sind rohe Logits, was der Distiller-Klasse erwartet.

    Parameters
    ----------
    student_body : Sequential
        Eine der STUDENT_S*-Architekturen aus dieser Datei.
        Das Modell wird intern geclont, sodass kein Fold Gewichte teilt.
    input_shape : tuple
        (H, W, C) ohne Batch-Dimension. Default: (256, 256, 3).
    learning_rate : float
        Wird vom Distiller überschrieben, aber für standalone-Tests relevant.

    Returns
    -------
    tf.keras.Model
        Functional-API-Modell: Input → Augmentierung → Body → Logits
    """
    body = clone_model(student_body)

    inputs = Input(shape=input_shape)

    # Identische Augmentierung wie in Station 1
    x = RandomTranslation(height_factor=0.1, width_factor=0.1)(inputs)
    x = RandomFlip(mode="horizontal_and_vertical")(x)
    x = RandomRotation(factor=0.2)(x)
    x = RandomZoom(height_factor=0.1, width_factor=0.1)(x)

    x = body(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=x,
        name=student_body.name,
    )

    # Compile mit from_logits=True — wichtig für standalone-Evaluation!
    # Der Distiller überschreibt dies mit seinem eigenen optimizer.
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    return model
