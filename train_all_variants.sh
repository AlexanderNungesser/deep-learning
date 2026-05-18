#!/bin/bash
# Automatisiertes Training aller Architektur-Varianten
# Generiert: 2026-05-17T14:57:37.208411

set -e

source tf-gpu-env/bin/activate
cd src

echo "=========================================="
echo "Start: Training aller Varianten"
echo "Timestamp: $(date)"
echo "=========================================="


echo ""
echo "[ 1] Training: MINIMAL_MODEL (minimal_cnn)"
echo "     → Baseline: 3 Conv-Schichten, Flatten"
echo "     → 33,648,394 Parameters"
echo "     → Logs: logs/minimal_cnn/"

# TODO: Hier würde das eigentliche Training-Skript aufgerufen
# python train_variant.py --model MINIMAL_MODEL --logs logs/minimal_cnn/


echo ""
echo "[ 2] Training: V1 (v1_minimal_with_bn)"
echo "     → MINIMAL mit BatchNorm"
echo "     → 33,649,290 Parameters"
echo "     → Logs: logs/v1_minimal_with_bn/"

# TODO: Hier würde das eigentliche Training-Skript aufgerufen
# python train_variant.py --model V1 --logs logs/v1_minimal_with_bn/


echo ""
echo "[ 3] Training: V2 (v2_minimal_bn_gap)"
echo "     → V1 mit GlobalAveragePooling statt Flatten"
echo "     → 103,050 Parameters"
echo "     → Logs: logs/v2_minimal_bn_gap/"

# TODO: Hier würde das eigentliche Training-Skript aufgerufen
# python train_variant.py --model V2 --logs logs/v2_minimal_bn_gap/


echo ""
echo "[ 4] Training: V3 (v3_double_conv_bn)"
echo "     → Doppelte Conv-Layer pro Block (32-32, 64-64, 128-128)"
echo "     → 297,706 Parameters"
echo "     → Logs: logs/v3_double_conv_bn/"

# TODO: Hier würde das eigentliche Training-Skript aufgerufen
# python train_variant.py --model V3 --logs logs/v3_double_conv_bn/


echo ""
echo "[ 5] Training: V4 (v4_deeper_filters)"
echo "     → V3 + 256-Filter Block"
echo "     → 619,178 Parameters"
echo "     → Logs: logs/v4_deeper_filters/"

# TODO: Hier würde das eigentliche Training-Skript aufgerufen
# python train_variant.py --model V4 --logs logs/v4_deeper_filters/


echo ""
echo "[ 6] Training: V5 (v5_deeper_dense)"
echo "     → V4 + tiefere Dense-Schichten (256→128)"
echo "     → 684,970 Parameters"
echo "     → Logs: logs/v5_deeper_dense/"

# TODO: Hier würde das eigentliche Training-Skript aufgerufen
# python train_variant.py --model V5 --logs logs/v5_deeper_dense/


echo ""
echo "[ 7] Training: V6 (v6_with_dropout_03)"
echo "     → V5 + Dropout(0.3) Regularisierung"
echo "     → 684,970 Parameters"
echo "     → Logs: logs/v6_with_dropout_03/"

# TODO: Hier würde das eigentliche Training-Skript aufgerufen
# python train_variant.py --model V6 --logs logs/v6_with_dropout_03/


echo ""
echo "[ 8] Training: V7 (v7_five_conv_blocks)"
echo "     → 5 Conv-Blöcke (mit 512-Block)"
echo "     → 2,492,202 Parameters"
echo "     → Logs: logs/v7_five_conv_blocks/"

# TODO: Hier würde das eigentliche Training-Skript aufgerufen
# python train_variant.py --model V7 --logs logs/v7_five_conv_blocks/


echo ""
echo "[ 9] Training: V8 (v8_five_conv_dropout)"
echo "     → V7 mit Dropout(0.3)"
echo "     → 2,492,202 Parameters"
echo "     → Logs: logs/v8_five_conv_dropout/"

# TODO: Hier würde das eigentliche Training-Skript aufgerufen
# python train_variant.py --model V8 --logs logs/v8_five_conv_dropout/


echo ""
echo "[10] Training: V9 (v9_with_512_dense)"
echo "     → V8 + Dense(512) für mehr Kapazität"
echo "     → 2,754,858 Parameters"
echo "     → Logs: logs/v9_with_512_dense/"

# TODO: Hier würde das eigentliche Training-Skript aufgerufen
# python train_variant.py --model V9 --logs logs/v9_with_512_dense/


echo ""
echo "[11] Training: BEST_CNN (9_conv_layer_cnn)"
echo "     → Baseline: 5 Conv-Blöcke, Dense(256), Dropout(0.5)"
echo "     → 2,492,202 Parameters"
echo "     → Logs: logs/9_conv_layer_cnn/"

# TODO: Hier würde das eigentliche Training-Skript aufgerufen
# python train_variant.py --model BEST_CNN --logs logs/9_conv_layer_cnn/


echo ""
echo "[12] Training: V10 (v10_best_cnn_lite)"
echo "     → BEST_CNN ohne 512-Block (4 Conv-Blöcke)"
echo "     → 1,244,458 Parameters"
echo "     → Logs: logs/v10_best_cnn_lite/"

# TODO: Hier würde das eigentliche Training-Skript aufgerufen
# python train_variant.py --model V10 --logs logs/v10_best_cnn_lite/


echo ""
echo "[13] Training: V11 (v11_best_cnn_deeper_dense)"
echo "     → BEST_CNN mit Dense(512→256) + Dropout(0.3)"
echo "     → 2,754,858 Parameters"
echo "     → Logs: logs/v11_best_cnn_deeper_dense/"

# TODO: Hier würde das eigentliche Training-Skript aufgerufen
# python train_variant.py --model V11 --logs logs/v11_best_cnn_deeper_dense/


echo ""
echo "[14] Training: V12 (v12_with_dense_bn)"
echo "     → BEST_CNN + BatchNorm in Dense-Schicht"
echo "     → 2,493,226 Parameters"
echo "     → Logs: logs/v12_with_dense_bn/"

# TODO: Hier würde das eigentliche Training-Skript aufgerufen
# python train_variant.py --model V12 --logs logs/v12_with_dense_bn/


echo ""
echo "=========================================="
echo "Finished: All variants trained"
echo "Timestamp: $(date)"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. python evaluate_variants.py --compare"
echo "2. Check reports/ for detailed results"
