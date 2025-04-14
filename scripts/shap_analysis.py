import shap # type: ignore
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from pathlib import Path

# Charge le modèle entraîné
BASE_DIR = Path(__file__).resolve().parent.parent
model_rf = joblib.load(BASE_DIR / 'models' / 'model_rf.pkl') #Random Forest
model_xgb = joblib.load(BASE_DIR / 'models' / 'model_xgb.pkl') #XGBoost

model = model_rf # A changer selon ...

# Charge les données de test
file_past = BASE_DIR / 'data' / 'processed' / 'past_matches.xlsx'
data = pd.read_excel(file_past, engine="openpyxl")

features = [
    'Last3_Goals_Diff',             # Différence des buts marqués sur les 3 derniers matchs
    'Last3_GoalsConceded_Diff',      # Différence des buts encaissés sur les 3 derniers matchs
    'Last3_ShotsOnTarget_Diff',      # Différence des tirs cadrés sur les 3 derniers matchs
    'Last3_ShotAccuracy_Diff',       # Différence de la précision des tirs sur les 3 derniers matchs
    'Last3_Corners_Diff',            # Différence des corners obtenus sur les 3 derniers matchs
    'CodeHomeTeam',                  # Code de l'équipe à domicile
    'CodeAwayTeam'                   # Code de l'équipe à l'extérieur
]


X_test = data[features].copy()
X_test = X_test.fillna(0)

# Crée l'explainer SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Affiche l'importance globale des features
plt.figure()
shap.summary_plot(shap_values, X_test)
plt.savefig("shap_summary_plot.png")

# Analyse l'impact d'une feature spécifique
plt.figure()
shap.dependence_plot("Last3_Goals_Diff", shap_values, X_test)
plt.savefig("shap_Last3_Goals_Diff_plot.png")

# Explication détaillée d'une prédiction individuelle (première ligne de X_test)
plt.figure()
shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], X_test.iloc[0, :], matplotlib=True)
plt.savefig("shap_force_plot.png")

print("Analyse SHAP terminée, graphiques sauvegardés")
