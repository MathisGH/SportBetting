import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import argparse
from pathlib import Path

# Paramètres CLI
parser = argparse.ArgumentParser()
parser.add_argument("--cutoff", type=str, default=None, help="Date limite pour entraîner (format YYYY-MM-DD)")
args = parser.parse_args()

# Charge les données
BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / 'data' / 'processed' / 'dataset_final.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# Convertit les dates
data["MatchDate"] = pd.to_datetime(data["MatchDate"], errors="coerce")

# Sélection des features
features = [
    'Last3_Goals_Diff', 'Last3_GoalsConceded_Diff',
    'Last3_ShotsOnTarget_Diff', 'Last3_ShotAccuracy_Diff',
    'Last3_Corners_Diff', 'CodeHomeTeam', 'CodeAwayTeam'
]

# Date de coupure pour séparation passé/futur
cutoff_date = pd.to_datetime(args.cutoff) if args.cutoff else pd.Timestamp.today()

# Sépare les matchs passés et futurs (à cette date précise)
past_matches = data[data["MatchDate"] < cutoff_date].sort_values("MatchDate")
future_matches = data[data["MatchDate"] >= cutoff_date].sort_values("MatchDate")

# Enregistre les deux fichiers pour cohérence du pipeline
past_matches_path = BASE_DIR / 'data' / 'processed' / 'past_matches.xlsx'
future_matches_path = BASE_DIR / 'data' / 'processed' / 'future_matches.xlsx'
past_matches.to_excel(past_matches_path, index=False)
future_matches.to_excel(future_matches_path, index=False)

# Crée X et y pour l'entraînement
X_train = past_matches[features].fillna(0)
y_train = past_matches["FTR"].map({'H': 0, 'D': 0, 'A': 1})

# Entraîne le modèle
rf = RandomForestClassifier(
    max_depth=20,
    max_features='log2',
    min_samples_leaf=8,
    min_samples_split=3,
    n_estimators=484,
    random_state=1
)
rf.fit(X_train, y_train)

# Sauvegarde le modèle
model_path = BASE_DIR / 'models' / 'model_rf.pkl'
joblib.dump(rf, model_path)
print("Modèle Random Forest entraîné avec succès et sauvegardé.")
