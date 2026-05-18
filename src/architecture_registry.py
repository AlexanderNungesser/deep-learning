"""
Registry für alle Architektur-Varianten.

Ermöglicht einfaches Durchiterieren über alle Varianten für systematisches Testing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from models import (
    MINIMAL_MODEL,
    BEST_CNN,
    VARIANT_V1_MINIMAL_WITH_BN,
    VARIANT_V2_MINIMAL_WITH_BN_GAP,
    VARIANT_V3_DOUBLE_CONV_WITH_BN,
    VARIANT_V4_DEEPER_FILTERS,
    VARIANT_V5_DEEPER_DENSE,
    VARIANT_V6_WITH_DROPOUT_03,
    VARIANT_V7_FIVE_CONV_BLOCKS,
    VARIANT_V8_FIVE_CONV_WITH_DROPOUT,
    VARIANT_V9_WITH_512_DENSE,
    VARIANT_V10_BEST_CNN_LITE,
    VARIANT_V11_BEST_CNN_DEEPER_DENSE,
    VARIANT_V12_WITH_DENSE_BN,
)
from data import INPUT_SHAPE

# Alle Architektur-Varianten in Progression
ARCHITECTURE_VARIANTS = [
    # Basis
    ("MINIMAL_MODEL", MINIMAL_MODEL, "Baseline: 3 Conv-Schichten, Flatten"),
    
    # V1-V6: Schrittweise Verbesserungen
    ("V1", VARIANT_V1_MINIMAL_WITH_BN, "MINIMAL mit BatchNorm"),
    ("V2", VARIANT_V2_MINIMAL_WITH_BN_GAP, "V1 mit GlobalAveragePooling statt Flatten"),
    ("V3", VARIANT_V3_DOUBLE_CONV_WITH_BN, "Doppelte Conv-Layer pro Block (32-32, 64-64, 128-128)"),
    ("V4", VARIANT_V4_DEEPER_FILTERS, "V3 + 256-Filter Block"),
    ("V5", VARIANT_V5_DEEPER_DENSE, "V4 + tiefere Dense-Schichten (256→128)"),
    ("V6", VARIANT_V6_WITH_DROPOUT_03, "V5 + Dropout(0.3) Regularisierung"),
    
    # V7-V9: Richtung BEST_CNN
    ("V7", VARIANT_V7_FIVE_CONV_BLOCKS, "5 Conv-Blöcke (mit 512-Block)"),
    ("V8", VARIANT_V8_FIVE_CONV_WITH_DROPOUT, "V7 mit Dropout(0.3)"),
    ("V9", VARIANT_V9_WITH_512_DENSE, "V8 + Dense(512) für mehr Kapazität"),
    
    # BEST_CNN Baseline
    ("BEST_CNN", BEST_CNN, "Baseline: 5 Conv-Blöcke, Dense(256), Dropout(0.5)"),
    
    # V10-V12: Verbesserungen über BEST_CNN
    ("V10", VARIANT_V10_BEST_CNN_LITE, "BEST_CNN ohne 512-Block (4 Conv-Blöcke)"),
    ("V11", VARIANT_V11_BEST_CNN_DEEPER_DENSE, "BEST_CNN mit Dense(512→256) + Dropout(0.3)"),
    ("V12", VARIANT_V12_WITH_DENSE_BN, "BEST_CNN + BatchNorm in Dense-Schicht"),
]


def get_all_models():
    """Gibt alle Modelle als Tuple-Liste (name, model_obj) zurück."""
    return [(name, model) for name, model, _ in ARCHITECTURE_VARIANTS]


def get_model_descriptions():
    """Gibt alle Modelle mit Beschreibungen zurück."""
    return ARCHITECTURE_VARIANTS


def print_architecture_summary():
    """Druckt eine schöne Zusammenfassung aller Varianten."""
    print("\n" + "="*80)
    print("ARCHITEKTUR-VARIANTEN ÜBERSICHT")
    print("="*80)
    
    for idx, (name, model, description) in enumerate(ARCHITECTURE_VARIANTS, 1):
        # Modell bauen, um Parameter zu zählen
        model.build(input_shape=(None, *INPUT_SHAPE))
        params = model.count_params()
        
        print(f"\n[{idx:2d}] {name:15s} | {model.name:30s}")
        print(f"     → {description}")
        print(f"     → {params:>12,} Parameter")
    
    print("\n" + "="*80)
    print(f"Gesamt: {len(ARCHITECTURE_VARIANTS)} Varianten")
    print("="*80 + "\n")


if __name__ == "__main__":
    print_architecture_summary()
