# Architektur-Varianten Dokumentation

## Übersicht

Es wurden **14 verschiedene Architektur-Varianten** erstellt, um eine systematische Progression von `MINIMAL_MODEL` zu `BEST_CNN` und darüber hinaus zu demonstrieren. Dies ermöglicht es, die Auswirkungen verschiedener architektonischer Änderungen auf die Genauigkeit zu untersuchen.

> **Hinweis**: Hyperparameter wie Epochen, Batch-Größe und Optimizer-Einstellungen werden **nicht** variiert. Alle Änderungen betreffen ausschließlich die **Architektur-Struktur**.

---

## Architektur-Progression

```
MINIMAL_MODEL (3 Conv-Schichten)
    ↓
V1 (+ BatchNormalization)
    ↓
V2 (+ GlobalAveragePooling statt Flatten)
    ↓
V3 (+ Doppelte Conv-Layer pro Block)
    ↓
V4 (+ 256-Filter Block)
    ↓
V5 (+ Tiefere Dense-Schichten)
    ↓
V6 (+ Dropout(0.3))
    ↓
V7 (+ 512-Filter Block = 5 Conv-Blöcke)
    ↓
V8 (+ Dropout im Dense-Layer)
    ↓
V9 (+ Dense(512) für mehr Kapazität)
    ↓
BEST_CNN (Baseline mit 0.83 Accuracy)
    ↓
V10 (Lite: ohne 512-Block)
    ↓
V11 (+ Deeper Dense Stack)
    ↓
V12 (+ BatchNorm in Dense-Schicht)
```

---

## Detaillierte Varianten-Beschreibungen

### **Basis: MINIMAL_MODEL**
- **Parameter**: ~33.6M
- **Struktur**:
  - Conv2D(32) → MaxPool(2,2)
  - Conv2D(64) → MaxPool(2,2)
  - Conv2D(128) → Flatten
  - Dense(64) → Dense(NUM_CLASSES)
- **Merkmale**: Sehr einfach, keine BatchNorm, kein Dropout
- **Baseline ohne weitere Verbesserungen**

---

### **V1: MINIMAL_WITH_BN**
- **Parameter**: ~33.6M (±0.03% Diff)
- **Struktur**: MINIMAL_MODEL + BatchNormalization nach jedem Conv2D
- **Änderung**: `BatchNormalization()` hinzufügt
- **Effekt**: 
  - Stabilisiert das Training
  - Verbessert die Konvergenz
  - Kann interne Covariate Shift reduzieren

---

### **V2: MINIMAL_WITH_BN_GAP**
- **Parameter**: ~103K (↓ 99.7%)
- **Struktur**: V1 mit `GlobalAveragePooling2D()` statt `Flatten()`
- **Änderung**: 
  - Ersetzt `Flatten()` durch `GlobalAveragePooling2D()`
  - Reduziert Dense-Eingaben drastisch
- **Effekt**:
  - Drastisch weniger Parameter
  - Robuster gegen räumliche Verschiebungen
  - Potentiell besser für Generalisierung

---

### **V3: DOUBLE_CONV_WITH_BN**
- **Parameter**: ~297K
- **Struktur**: 
  - 2×Conv2D(32) + BatchNorm → MaxPool
  - 2×Conv2D(64) + BatchNorm → MaxPool
  - 2×Conv2D(128) + BatchNorm → MaxPool
  - GlobalAveragePooling2D
  - Dense(64) → Dense(NUM_CLASSES)
- **Änderung**: Doppelte Conv-Layer pro Block
- **Effekt**:
  - Tieferes Feature-Lernen
  - Besseres Capture komplexer Patterns
  - Moderat mehr Kapazität

---

### **V4: DEEPER_FILTERS**
- **Parameter**: ~619K
- **Struktur**: V3 + zusätzlicher Conv2D(256) Block
- **Änderung**: Neuer 4. Conv-Block mit 256 Filtern + Dense(128)
- **Effekt**:
  - Noch tiefere Feature-Extraktion
  - Größere Modell-Kapazität
  - Höheres Overfitting-Risiko

---

### **V5: DEEPER_DENSE**
- **Parameter**: ~685K
- **Struktur**: V4 + tiefere Dense-Schichten
- **Änderung**: Dense-Stack `Dense(256) → Dense(128)` statt nur `Dense(128)`
- **Effekt**:
  - Mehr Flexibilität in der Klassifikation
  - Bessere Nutzung der extrahierten Features

---

### **V6: WITH_DROPOUT_03**
- **Parameter**: ~685K (gleich V5)
- **Struktur**: V5 + `Dropout(0.3)` nach Dense-Layern
- **Änderung**: Regularisierung hinzugefügt
- **Effekt**:
  - Verhindert Overfitting
  - Verbessert Generalisierung
  - Moderat aggressive Regularisierung (0.3)

---

### **V7: FIVE_CONV_BLOCKS**
- **Parameter**: ~2.5M
- **Struktur**: 5 Conv-Blöcke mit doppelten Conv-Layern
  - 32-32, 64-64, 128-128, 256-256, 512
- **Änderung**: 5. Block mit Conv2D(512) hinzugefügt
- **Effekt**:
  - Deutlich tieferes Netzwerk
  - Massiv mehr Parameter
  - Hohes Risiko für Overfitting
  - Sehr hohes Lernpotential

---

### **V8: FIVE_CONV_WITH_DROPOUT**
- **Parameter**: ~2.5M (gleich V7)
- **Struktur**: V7 + `Dropout(0.3)` nach Dense(256)
- **Änderung**: Regularisierung für tiefes Netzwerk
- **Effekt**:
  - Bessere Generalisierung des tiefen Netzwerks
  - Balance zwischen Kapazität und Regularisierung

---

### **V9: WITH_512_DENSE**
- **Parameter**: ~2.75M
- **Struktur**: V8 + größerer Dense-Stack
- **Änderung**: `Dense(512) → Dense(256)` statt `Dense(256)`
- **Effekt**:
  - Maximale Dense-Kapazität
  - Potentiell bessere Klassifikation
  - Höheres Overfitting-Risiko

---

### **BEST_CNN (Baseline)**
- **Parameter**: ~2.5M
- **Accuracy**: **0.83** (Baseline)
- **Struktur**: 
  - 5 Conv-Blöcke (32-32, 64-64, 128-128, 256-256, 512)
  - GlobalAveragePooling2D
  - Dense(256) + Dropout(0.5)
  - Dense(NUM_CLASSES)
- **Merkmale**: Bewährtes Design mit guter Balance

---

### **V10: BEST_CNN_LITE**
- **Parameter**: ~1.24M (↓ 50%)
- **Struktur**: BEST_CNN ohne 512-Block (nur 4 Conv-Blöcke)
- **Änderung**: 5. Conv-Block entfernt, aber Dropout(0.3) behalten
- **Effekt**:
  - Halbierte Parameter
  - Schnelleres Training
  - Könnte leichter zu trainieren sein
  - Potentiell bessere Generalisierung

---

### **V11: BEST_CNN_DEEPER_DENSE**
- **Parameter**: ~2.75M
- **Struktur**: BEST_CNN + tieferer Dense-Stack
- **Änderung**: 
  - Dense(256) → Dense(512) → Dense(256)
  - Dropout(0.3) nach Dense(256)
  - Dropout(0.3) nach Dense(512)
- **Effekt**:
  - Maximize Dense-Verarbeitung
  - Höhere Regularisierung (0.3 statt 0.5)
  - Bessere Feature-Kombination

---

### **V12: WITH_DENSE_BN**
- **Parameter**: ~2.49M
- **Struktur**: BEST_CNN mit `BatchNormalization()` im Dense-Layer
- **Änderung**: `BatchNormalization()` nach `Dense(256)`, vor Dropout
- **Effekt**:
  - Normalisierung auch in Dense-Schichten
  - Stabilisiert Dense-Layer-Training
  - Potentiell schnellere Konvergenz

---

## Verwendung

### Alle Modelle durchiterieren (Python)

```python
from src.architecture_registry import ARCHITECTURE_VARIANTS, print_architecture_summary

# Übersicht ausgeben
print_architecture_summary()

# Alle Modelle durchiterieren
for name, model, description in ARCHITECTURE_VARIANTS:
    print(f"Training {name}: {description}")
    # Training-Code hier
    model_instance = build_model(model)
    # ...
```

### Im Notebook

```python
from src.architecture_registry import get_all_models

for model_name, model_obj in get_all_models():
    print(f"Testing {model_name}...")
    # Testing-Code
```

### Einzelnes Modell verwenden

```python
from src.models import VARIANT_V10_BEST_CNN_LITE

model = build_model(VARIANT_V10_BEST_CNN_LITE)
```

---

## Vergleichende Analysen

### Parameter-Vergleich

| Variante    | Parameter  | Rel. zu BEST_CNN | 
|------------|-----------|-----------------|
| MINIMAL    | 33.6M     | +1,248%         |
| V1         | 33.6M     | +1,248%         |
| V2         | 103K      | -95.9%          |
| V3         | 297K      | -88.1%          |
| V4         | 619K      | -75.2%          |
| V5         | 685K      | -72.5%          |
| V6         | 685K      | -72.5%          |
| V7         | 2.5M      | +0.3%           |
| V8         | 2.5M      | +0.3%           |
| V9         | 2.75M     | +10.5%          |
| **BEST_CNN** | **2.49M** | **Baseline**    |
| V10        | 1.24M     | -50.2%          |
| V11        | 2.75M     | +10.5%          |
| V12        | 2.49M     | -0.2%           |

---

## Experimentalstrategie

1. **Phase 1** (V1-V6): Schrittweise Verbesserung der Basis
   - Ziel: Optimales Gleichgewicht mit weniger Parametern finden
   
2. **Phase 2** (V7-V9): Richtung BEST_CNN
   - Ziel: Tiefere Netzwerke evaluieren
   
3. **Phase 3** (V10-V12): Verbesserungen über BEST_CNN
   - Ziel: Baseline von 0.83 Accuracy übertreffen

---

## Esperierte Ergebnisse

**Ziel**: Mindestens eine Variante mit **>0.83 Accuracy** finden.

**Hypothesen**:
- **V2**: Stark reduzierte Parameter könnten besser generalisieren → ⚠️ Zu dünn
- **V6**: Mit Dropout sollte Overfitting reduziert werden → ✓ Vielversprechend
- **V10**: 50% weniger Parameter, könnte schneller + besser konvergieren → ✓ Vielversprechend
- **V11**: Mehr Dense-Verarbeitung könnte Features besser nutzen → ✓ Vielversprechend
- **V12**: Dense BatchNorm könnte Training stabilisieren → ✓ Vielversprechend

---

## Weitere Optimierungsmöglichkeiten

Falls keine Variante >0.83 erreicht, könnten nächste Iterationen sein:
- Residual Connections (ResNet-Stil)
- Attention Mechanisms
- Ensembles aus mehreren Varianten
- Fine-tuning mit zusätzlicher Augmentation
- Learning Rate Scheduling

---

*Dokumentation erstellt: Mai 2026*
*Projekt: Galaxy10 DECals Klassifikation - Milestone 1*
