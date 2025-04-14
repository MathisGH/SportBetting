import pandas as pd
import joblib
import argparse
from datetime import datetime
from pathlib import Path

FEATURES = [
    'Last3_Goals_Diff',
    'Last3_GoalsConceded_Diff',
    'Last3_ShotsOnTarget_Diff',
    'Last3_ShotAccuracy_Diff',
    'Last3_Corners_Diff',
    'CodeHomeTeam',
    'CodeAwayTeam'
]

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATHS = {
    "rf": BASE_DIR / "models" / "model_rf.pkl",
    "xgb": BASE_DIR / "models" / "model_xgb.pkl"
}


def load_model(model_name):
    return joblib.load(MODEL_PATHS[model_name])


def predict_past_matches(model):
    file_path = BASE_DIR / 'data' / 'processed' / 'dataset_final.xlsx'
    df = pd.read_excel(file_path, engine="openpyxl")
    df["MatchDate"] = pd.to_datetime(df["MatchDate"], errors="coerce")

    # Filtre uniquement les matchs déjà joués en 2025
    df_2025 = df[(df["MatchDate"].dt.year == 2025) & (df["MatchDate"] < pd.Timestamp.today())].copy()

    if df_2025.empty:
        print("Aucun match de 2025 joué trouvé dans dataset_final.xlsx.")
        return

    X = df_2025[FEATURES].fillna(0)
    df_2025["Prediction"] = model.predict(X)
    df_2025["Correct"] = df_2025["Prediction"] == df_2025["FTR_encoded"]

    output_path = BASE_DIR / 'data' / 'processed' / 'predictions_passees.xlsx'
    df_2025.to_excel(output_path, index=False)

    print(f"Prédictions passées sauvegardées dans {output_path}")
    print(f"{df_2025['Correct'].sum()} bonnes prédictions sur {len(df_2025)} matchs ({df_2025['Correct'].mean():.2%})")



def predict_future_matches(model):
    file_path = BASE_DIR / 'data' / 'processed' / 'dataset_final.xlsx'
    df = pd.read_excel(file_path, engine='openpyxl')
    future_matches = df[df['MatchDate'] >= pd.Timestamp.today()].sort_values('MatchDate')

    X = future_matches[FEATURES].fillna(0)
    future_matches["Predicted_Result"] = model.predict(X)
    future_matches["Predicted_Result"] = future_matches["Predicted_Result"].map({1: "AwayWin", 0: "Draw/HomeWin"})

    output_path = BASE_DIR / 'data' / "processed" / 'predictions.xlsx'
    future_matches.to_excel(output_path, index=False)

    print(f"Prédictions enregistrées dans : {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Générer des prédictions pour les matchs passés ou futurs.")
    parser.add_argument("--type", choices=["past", "future"], required=True, help="Type de prédiction : past ou future")
    parser.add_argument("--model", choices=["rf", "xgb"], default="rf", help="Modèle à utiliser")
    args = parser.parse_args()

    model = load_model(args.model)

    if args.type == "past":
        predict_past_matches(model)
    else:
        predict_future_matches(model)


if __name__ == "__main__":
    main()