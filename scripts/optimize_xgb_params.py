import optuna # type: ignore
import pandas as pd
import json
from xgboost import XGBClassifier # type: ignore
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

# Charge les données
BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / 'data' / 'processed' / 'dataset_final.xlsx'
data = pd.read_excel(file_path, engine="openpyxl")
data["MatchDate"] = pd.to_datetime(data["MatchDate"], errors="coerce")

# Filtre les données avant une date de cutoff (pour éviter les fuites de données)
cutoff_date = pd.to_datetime("2025-01-01")
train_data = data[data["MatchDate"] < cutoff_date].copy()
test_data = data[(data["MatchDate"] >= cutoff_date) & (data["MatchDate"] < pd.Timestamp.today())].copy()

# Features et target
features = [
    "Last3_Goals_Diff", "Last3_GoalsConceded_Diff", "Last3_ShotsOnTarget_Diff",
    "Last3_ShotAccuracy_Diff", "Last3_Corners_Diff", "CodeHomeTeam", "CodeAwayTeam"
]
X_train = train_data[features].fillna(0)
y_train = train_data["FTR"].map({"H": 0, "D": 0, "A": 1})
X_test = test_data[features].fillna(0)
y_test = test_data["FTR"].map({"H": 0, "D": 0, "A": 1})

# Calcul du ratio pour l'équilibrage
class_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.01, 10, log=True),
        "lambda": trial.suggest_float("lambda", 0.01, 10, log=True),
        "alpha": trial.suggest_float("alpha", 0.01, 10, log=True),
        "scale_pos_weight": class_ratio
    }

    model = XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

print("Optimisation des hyperparamètres XGBoost en cours...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("\nMeilleurs hyperparamètres trouvés:")
print(study.best_params)

# Sauvegarde dans un fichier JSON
os.makedirs("config", exist_ok=True)
config_path = os.path.join("config", "best_xgb_params.json")
with open(config_path, "w") as f:
    json.dump(study.best_params, f, indent=4)

print("\nParamètres sauvegardés dans config/best_xgb_params.json")
