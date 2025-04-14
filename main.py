import os
import subprocess
import argparse
from pathlib import Path

# --- Argument CLI pour choisir le modèle ---
parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["rf", "xgb"], default="rf", help="Choisir le modèle à utiliser")
args = parser.parse_args()
model = args.model

# --- Base path ---
BASE_DIR = Path(__file__).resolve().parent
scripts_folder = os.path.join(BASE_DIR, "scripts")

# --- Scripts communs ---
scripts = [
    "download_past_matches.py",
    "upcoming_matches.py",
    "data_processing.py"
]

# --- Ajout du bon script d'entraînement ---
if model == "rf":
    scripts.append("train_rf_model.py --cutoff 2025-01-01")
else:
    scripts.append("train_xgb_optimised.py --cutoff 2025-01-01")

# --- Ajout des prédictions selon le modèle ---
scripts.append(f"predict_matches.py --type past --model {model}")
scripts.append(f"predict_matches.py --type future --model {model}")

# --- Exécution ---
print(f"\nLancement du pipeline avec le modèle : {model.upper()}\n")

for cmd in scripts:
    parts = cmd.split()
    script = parts[0]
    args = parts[1:] if len(parts) > 1 else []

    script_path = os.path.join(scripts_folder, script)
    print(f"Exécution de {script} {' '.join(args)}...")
    result = subprocess.run(["python", script_path] + args, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"{script} terminé avec succès\n")
    else:
        print(f"Erreur dans {script} :\n{result.stderr}")
        break

print("Pipeline terminé.")
