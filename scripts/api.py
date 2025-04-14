from fastapi import FastAPI # type: ignore
import pandas as pd
import joblib
from pathlib import Path

app = FastAPI()

# Charge le modèle et les données
BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "processed" / "future_matches.xlsx"
model_rf = joblib.load(BASE_DIR / "models" / "model_rf.pkl") #Random Forest
model_xgb = joblib.load(BASE_DIR / "models" / "model_xgb.pkl") #XGBoost

model = model_rf

future_matches = pd.read_excel(data_path, engine="openpyxl")

# Sélection des features utilisées par le modèle
features = [
    'Last3_Goals_Diff',             # Différence des buts marqués sur les 3 derniers matchs
    'Last3_GoalsConceded_Diff',      # Différence des buts encaissés sur les 3 derniers matchs
    'Last3_ShotsOnTarget_Diff',      # Différence des tirs cadrés sur les 3 derniers matchs
    'Last3_ShotAccuracy_Diff',       # Différence de la précision des tirs sur les 3 derniers matchs
    'Last3_Corners_Diff',            # Différence des corners obtenus sur les 3 derniers matchs
    'CodeHomeTeam',                  # Code de l'équipe à domicile
    'CodeAwayTeam'                   # Code de l'équipe à l'extérieur
]



@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API de prédiction de matchs !"}

@app.get("/predict")
def predict():
    if future_matches.empty:
        return {"error": "Aucun match à prédire"}

    X_future = future_matches[features].fillna(0)
    predictions = model.predict(X_future)

    # Ajoute les prédictions aux matchs
    future_matches["Prediction"] = predictions

    return future_matches[["MatchDate", "HomeTeam", "AwayTeam", "Prediction"]].to_dict(orient="records")

