import pandas as pd
import joblib
import argparse
from xgboost import XGBClassifier  # type: ignore
import json
from pathlib import Path

# --- CLI ---
parser = argparse.ArgumentParser()
parser.add_argument("--cutoff", type=str, help="Date limite pour l'entraînement (format YYYY-MM-DD)")
args = parser.parse_args()

# --- Constantes ---
FEATURES = [
    'Last3_Goals_Diff', 'Last3_GoalsConceded_Diff',
    'Last3_ShotsOnTarget_Diff', 'Last3_ShotAccuracy_Diff',
    'Last3_Corners_Diff', 'CodeHomeTeam', 'CodeAwayTeam'
]
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "dataset_final.xlsx"
PARAMS_PATH = BASE_DIR / "config" / "best_xgb_params.json"
MODEL_PATH = BASE_DIR / "models" / "model_xgb.pkl"

# --- Données ---
data = pd.read_excel(DATA_PATH, engine="openpyxl")
data["MatchDate"] = pd.to_datetime(data["MatchDate"], errors="coerce")

# --- Sépare passé/futur à la date de coupure
cutoff_date = pd.to_datetime(args.cutoff) if args.cutoff else pd.Timestamp.today()
past_matches = data[data["MatchDate"] < cutoff_date].sort_values("MatchDate")
future_matches = data[data["MatchDate"] >= cutoff_date].sort_values("MatchDate")

# --- Sauvegarde les fichiers pour cohérence du pipeline
past_matches_path = BASE_DIR / 'data' / 'processed' / 'past_matches.xlsx'
future_matches_path = BASE_DIR / 'data' / 'processed' / 'future_matches.xlsx'
past_matches.to_excel(past_matches_path, index=False)
future_matches.to_excel(future_matches_path, index=False)

# --- Entraînement
X_train = past_matches[FEATURES].fillna(0)
y_train = past_matches["FTR"].map({'H': 0, 'D': 0, 'A': 1})

# --- Charge les meilleurs hyperparamètres
with open(PARAMS_PATH, "r") as f:
    best_params = json.load(f)

model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss", random_state=42)
model.fit(X_train, y_train)

# --- Sauvegarde du modèle
joblib.dump(model, MODEL_PATH)
print("Modèle XGBoost entraîné avec les meilleurs paramètres et sauvegardé.")
