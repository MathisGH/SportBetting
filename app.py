import streamlit as st  # type: ignore
import pandas as pd
import joblib
import os
import plotly.express as px  # type: ignore
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import plotly.figure_factory as ff


# CONFIGURATION 
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

# SIDEBAR : choix du modèle
model_display_names = {"Random Forest": "rf", "XGBoost": "xgb"}

st.sidebar.title("Paramètres")
selected_label = st.sidebar.selectbox("Modèle utilisé", list(model_display_names.keys()))
model_choice = model_display_names[selected_label]  # "rf" ou "xgb"

model = joblib.load(MODEL_PATHS[model_choice])
st.sidebar.write(f"Modèle chargé : {model_choice}") # Pour vérifier le modèle chargé

# Bouton mise à jour
if st.sidebar.button("Mettre à jour les données"):
    with st.spinner("Mise à jour en cours..."):
        os.system(f'cmd /c "python main.py --model {model_choice.lower().replace(" ", "_")}"')
    st.success("Données mises à jour avec succès !")
    st.rerun()

# PRÉDICTIONS FUTURES
st.title("Prédictions des prochains matchs de Ligue 1")

df_future = pd.read_excel(FUTURE_PATH, engine="openpyxl")
df_future = df_future.dropna(subset=FEATURES)
df_future["Prediction"] = model.predict(df_future[FEATURES])
st.write("Extrait des prédictions :", df_future["Prediction"].value_counts())


teams = sorted(set(df_future["HomeTeam"]).union(df_future["AwayTeam"]))
selected_team = st.selectbox("Filtrer par équipe :", ["Toutes"] + teams)

filtered_future = df_future if selected_team == "Toutes" else df_future[
    (df_future["HomeTeam"] == selected_team) | (df_future["AwayTeam"] == selected_team)
]

st.dataframe(filtered_future[["MatchDate", "HomeTeam", "AwayTeam", "Prediction"]],
             height=600, use_container_width=True)

# PRÉDICTIONS PASSÉES
st.header("Performance des Prédictions Passées")

# Nouveau chemin dynamique selon le modèle choisi
PRED_PAST_PATH = BASE_DIR / "data" / "processed" / f"predictions_passees_{model_choice}.xlsx"

try:
    df_pred_past = pd.read_excel(PRED_PAST_PATH, engine="openpyxl")
except FileNotFoundError:
    st.warning("Aucune prédiction passée trouvée pour ce modèle. Lance d'abord le script de prédiction.")
    df_pred_past = None

if df_pred_past is not None and not df_pred_past.empty:
    st.subheader("Tableau des Prédictions depuis 2025")
    st.dataframe(df_pred_past[["MatchDate", "HomeTeam", "AwayTeam", "Prediction", "FTR_encoded", "Correct"]],
                 height=600, use_container_width=True)

    y_true = df_pred_past["FTR_encoded"]
    y_pred = df_pred_past["Prediction"]

    st.subheader("Métriques de classification")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    st.metric("Accuracy", f"{acc:.2%}")
    st.metric("Precision", f"{prec:.2%}")
    st.metric("Recall", f"{rec:.2%}")
    st.metric("F1-Score", f"{f1:.2%}")

    st.subheader("Matrice de confusion")
    cm = confusion_matrix(y_true, y_pred)
    cm_labels = ["Draw/HomeWin", "AwayWin"]
    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=cm_labels,
        y=cm_labels,
        colorscale='Blues',
        showscale=True,
        hoverinfo="z"
    )
    fig_cm.update_layout(xaxis_title="Prédiction", yaxis_title="Réel")
    st.plotly_chart(fig_cm)

