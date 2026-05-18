"""
Script zum systematischen Vergleich aller Architektur-Varianten.

Trainiert alle Varianten und erstellt einen Vergleichsbericht.
"""

import sys
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

from architecture_registry import ARCHITECTURE_VARIANTS
from data import INPUT_SHAPE, NUM_CLASSES
from models import build_model
import numpy as np


def create_comparison_config():
    """Erstellt eine Test-Konfiguration für alle Varianten."""
    
    config = {
        "timestamp": datetime.now().isoformat(),
        "variants": [],
        "hyperparameters": {
            "batch_size": 16,
            "epochs": 100,
            "learning_rate": 1e-4,
            "optimizer": "Adam",
            "loss": "categorical_crossentropy",
            "data_augmentation": ["Translation", "Flip", "Rotation", "Zoom"],
            "note": "Hyperparameter sind konstant über alle Varianten"
        }
    }
    
    for idx, (name, model_obj, description) in enumerate(ARCHITECTURE_VARIANTS):
        model_obj.build(input_shape=(None, *INPUT_SHAPE))
        params = model_obj.count_params()
        
        variant_info = {
            "id": idx + 1,
            "name": name,
            "model_name": model_obj.name,
            "description": description,
            "parameters": params,
            "logs_directory": f"logs/{model_obj.name}/"
        }
        config["variants"].append(variant_info)
    
    return config


def print_comparison_table(config):
    """Druckt eine schöne Vergleichstabelle."""
    
    print("\n" + "="*120)
    print("ARCHITEKTUR-VARIANTEN VERGLEICH")
    print("="*120)
    print(f"Zeitstempel: {config['timestamp']}")
    print(f"\nHyperparameter (konstant):")
    for key, value in config['hyperparameters'].items():
        if key != 'note':
            print(f"  {key:20s}: {value}")
    print(f"  {config['hyperparameters']['note']}")
    
    print("\n" + "-"*120)
    print(f"{'ID':>3} | {'Name':^15} | {'Modell':^30} | {'Parameter':>15} | Beschreibung")
    print("-"*120)
    
    for variant in config['variants']:
        params_str = f"{variant['parameters']:,}"
        print(
            f"{variant['id']:3d} | "
            f"{variant['name']:^15} | "
            f"{variant['model_name']:^30} | "
            f"{params_str:>15} | "
            f"{variant['description'][:50]}"
        )
    
    print("="*120 + "\n")


def create_experiment_shell_script(config, output_dir="logs/"):
    """Erstellt ein Shell-Script zum Trainieren aller Varianten nacheinander."""
    
    script_content = """#!/bin/bash
# Automatisiertes Training aller Architektur-Varianten
# Generiert: {timestamp}

set -e

source tf-gpu-env/bin/activate
cd src

echo "=========================================="
echo "Start: Training aller Varianten"
echo "Timestamp: $(date)"
echo "=========================================="

"""
    
    for variant in config['variants']:
        script_content += f"""
echo ""
echo "[{variant['id']:2d}] Training: {variant['name']} ({variant['model_name']})"
echo "     → {variant['description']}"
echo "     → {variant['parameters']:,} Parameters"
echo "     → Logs: {variant['logs_directory']}"

# TODO: Hier würde das eigentliche Training-Skript aufgerufen
# python train_variant.py --model {variant['name']} --logs {variant['logs_directory']}

"""
    
    script_content += """
echo ""
echo "=========================================="
echo "Finished: All variants trained"
echo "Timestamp: $(date)"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. python evaluate_variants.py --compare"
echo "2. Check reports/ for detailed results"
"""
    
    script_content = script_content.format(timestamp=config['timestamp'])
    
    script_path = Path(__file__).parent / "train_all_variants.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    script_path.chmod(0o755)
    print(f"✓ Training-Script erstellt: {script_path}")
    
    return script_path


def save_config_json(config, output_file="variant_comparison_config.json"):
    """Speichert die Konfiguration als JSON."""
    
    output_path = Path(__file__).parent / output_file
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Konfiguration gespeichert: {output_path}")
    
    return output_path


def save_config_csv(config, output_file="variant_comparison_config.csv"):
    """Speichert die Konfiguration als CSV."""
    
    data = []
    for variant in config['variants']:
        data.append({
            'ID': variant['id'],
            'Name': variant['name'],
            'Model_Name': variant['model_name'],
            'Parameters': variant['parameters'],
            'Logs_Dir': variant['logs_directory'],
            'Description': variant['description']
        })
    
    df = pd.DataFrame(data)
    output_path = Path(__file__).parent / output_file
    df.to_csv(output_path, index=False)
    
    print(f"✓ Konfiguration (CSV) gespeichert: {output_path}")
    
    return output_path


def main():
    print("\n" + "="*120)
    print("ARCHITEKTUR-VARIANTEN VERGLEICH - INITIALISIERUNG")
    print("="*120)
    
    # Konfiguration erstellen
    config = create_comparison_config()
    
    # Tabelle ausgeben
    print_comparison_table(config)
    
    # Dateien speichern
    json_file = save_config_json(config)
    csv_file = save_config_csv(config)
    script_file = create_experiment_shell_script(config)
    
    # Zusammenfassung
    print("\nZUSAMMENFASSUNG:")
    print(f"  → {len(config['variants'])} Architektur-Varianten definiert")
    print(f"  → Konfiguration: {json_file}, {csv_file}")
    print(f"  → Training-Script: {script_file}")
    
    print("\nNACHSTES VORGEHEN:")
    print(f"  1. Modifiziere {script_file} mit deinem Training-Code")
    print(f"  2. Führe aus: bash {script_file}")
    print(f"  3. Vergleiche Ergebnisse: python compare_results.py")
    
    print("\n" + "="*120 + "\n")


if __name__ == "__main__":
    main()
