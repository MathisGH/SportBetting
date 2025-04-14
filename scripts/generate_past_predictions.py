import pandas as pd
import joblib
from datetime import datetime

# Charger les matchs passés
file_past = r"C:\Users\Mathi\OneDrive\Bureau\Projets\Sport betting\data\Données finales\past_matches.xlsx"
df_past = pd.read_excel(file_past, engine="openpyxl")

# Filtrer les matchs de 2025
df_past["MatchDate"] = pd.to_datetime(df_past["MatchDate"], errors="coerce")
df_past_2025 = df_past[df_past["MatchDate"].dt.year == 2025].copy()

if df_past_2025.empty:
    print("Aucun match de 2025 trouvé dans les données passées.")
    exit()

# Charger le modèle
model_rf = joblib.load(r"C:\Users\Mathi\OneDrive\Bureau\Projets\Sport betting\Modèles\model_rf.pkl") #Random Forest
model_xgb = joblib.load(r"C:\Users\Mathi\OneDrive\Bureau\Projets\Sport betting\Modèles\model_xgb.pkl") #XGBoost
model_xgb_opti = joblib.load(r"C:\Users\Mathi\OneDrive\Bureau\Projets\Sport betting\Modèles\model_xgb_optimisé.pkl") #XGBoost optimisé

model = model_rf # A changer selon ...

# Sélection des features
features = [
    'Last3_Goals_Diff',             # Différence des buts marqués sur les 3 derniers matchs
    'Last3_GoalsConceded_Diff',      # Différence des buts encaissés sur les 3 derniers matchs
    'Last3_ShotsOnTarget_Diff',      # Différence des tirs cadrés sur les 3 derniers matchs
    'Last3_ShotAccuracy_Diff',       # Différence de la précision des tirs sur les 3 derniers matchs
    'Last3_Corners_Diff',            # Différence des corners obtenus sur les 3 derniers matchs
    'CodeHomeTeam',                  # Code de l'équipe à domicile
    'CodeAwayTeam'                   # Code de l'équipe à l'extérieur
]

X_test_2025 = df_past_2025[features]

# Faire les prédictions
X_test_2025 = X_test_2025.fillna(0)  # Remplace les NaN par 0

df_past_2025["Prediction"] = model.predict(X_test_2025)

# Comparer aux résultats réels
df_past_2025["Correct"] = df_past_2025["Prediction"] == df_past_2025["FTR_encoded"]

# Sauvegarder dans un fichier Excel
file_predictions = r"C:\Users\Mathi\OneDrive\Bureau\Projets\Sport betting\data\Données finales\predictions_passées.xlsx"
df_past_2025.to_excel(file_predictions, index=False)

print(f"Prédictions passées sauvegardées dans {file_predictions}")
print(f"{df_past_2025['Correct'].sum()} bonnes prédictions sur {len(df_past_2025)} matchs ({df_past_2025['Correct'].mean():.2%})")
