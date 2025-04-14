import streamlit as st  # type: ignore
import pandas as pd
import joblib
import os
import plotly.express as px  # type: ignore
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
FUTURE_PATH = BASE_DIR / "data" / "processed" / "future_matches.xlsx"
PAST_PATH = BASE_DIR / "data" / "processed" / "past_matches.xlsx"
MODEL_PATHS = {
    "rf": BASE_DIR / "models" / "model_rf.pkl",
    "xgb": BASE_DIR / "models" / "model_xgb.pkl"
}
FEATURES = [
    'Last3_Goals_Diff', 'Last3_GoalsConceded_Diff',
    'Last3_ShotsOnTarget_Diff', 'Last3_ShotAccuracy_Diff',
    'Last3_Corners_Diff', 'CodeHomeTeam', 'CodeAwayTeam'
]

# --- SIDEBAR : choix du modèle ---
model_display_names = {"Random Forest": "rf", "XGBoost": "xgb"}

st.sidebar.title("Paramètres")
selected_label = st.sidebar.selectbox("Modèle utilisé", list(model_display_names.keys()))
model_choice = model_display_names[selected_label]  # "rf" ou "xgb"

model = joblib.load(MODEL_PATHS[model_choice])

# --- Bouton mise à jour ---
if st.sidebar.button("Mettre à jour les données"):
    with st.spinner("Mise à jour en cours..."):
        os.system(f'cmd /c "python main.py --model {model_choice.lower().replace(" ", "_")}"')
    st.success("Données mises à jour avec succès !")
    st.experimental_rerun()

# =============================
#       PRÉDICTIONS FUTURES
# =============================
st.title("Prédictions des prochains matchs de Ligue 1")

df_future = pd.read_excel(FUTURE_PATH, engine="openpyxl")
df_future = df_future.dropna(subset=FEATURES)
df_future["Prediction"] = model.predict(df_future[FEATURES])

teams = sorted(set(df_future["HomeTeam"]).union(df_future["AwayTeam"]))
selected_team = st.selectbox("Filtrer par équipe :", ["Toutes"] + teams)

filtered_future = df_future if selected_team == "Toutes" else df_future[
    (df_future["HomeTeam"] == selected_team) | (df_future["AwayTeam"] == selected_team)
]

st.dataframe(filtered_future[["MatchDate", "HomeTeam", "AwayTeam", "Prediction"]],
             height=600, use_container_width=True)

# =============================
#       PRÉDICTIONS PASSÉES
# =============================
st.header("Performance des Prédictions Passées")

# Nouveau chemin vers les prédictions sauvegardées
PRED_PAST_PATH = BASE_DIR / "data" / "processed" / "predictions_passees.xlsx"

try:
    df_pred_past = pd.read_excel(PRED_PAST_PATH, engine="openpyxl")
except FileNotFoundError:
    st.warning("Aucune prédiction passée trouvée. Lance d'abord le script de prédiction.")
    df_pred_past = None

if df_pred_past is not None and not df_pred_past.empty:
    st.subheader("Tableau des Prédictions depuis 2025")
    st.dataframe(df_pred_past[["MatchDate", "HomeTeam", "AwayTeam", "Prediction", "FTR_encoded", "Correct"]],
                 height=600, use_container_width=True)

    accuracy = df_pred_past["Correct"].mean()
    st.subheader("Taux de réussite")
    st.metric("Précision du modèle", f"{accuracy:.2%}")

    fig = px.histogram(df_pred_past, x="Correct", color="Correct",
                       title="Distribution des prédictions correctes",
                       labels={"Correct": "Prédiction correcte"})
    st.plotly_chart(fig)
else:
    st.warning("Aucun match de 2025 trouvé dans les prédictions sauvegardées.")
