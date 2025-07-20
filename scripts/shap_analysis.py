import shap  # type: ignore
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Model
model_rf = joblib.load(BASE_DIR / 'models' / 'model_rf.pkl')
model_xgb = joblib.load(BASE_DIR / 'models' / 'model_xgb.pkl')

model = model_xgb # à choisir

# Data
file_past = BASE_DIR / 'data' / 'processed' / 'past_matches.xlsx'
data = pd.read_excel(file_past, engine="openpyxl")

# Sélection des features
features = [
    'Last3_Goals_Diff',
    'Last3_GoalsConceded_Diff',
    'Last3_ShotsOnTarget_Diff',
    'Last3_ShotAccuracy_Diff',
    'Last3_Corners_Diff',
    'CodeHomeTeam',
    'CodeAwayTeam'
]

X_test = data[features].copy().fillna(0)
print("X_test shape:", X_test.shape)

# Explainer et calcul des valeurs SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Gérer les formats différents de shap_values (différent entre rf et xgb)
if isinstance(shap_values, list):
    # Multi-classe (rf)
    shap_values_to_use = shap_values[1]
else:
    # Binaire (xgb)
    shap_values_to_use = shap_values

assert shap_values_to_use.shape == X_test.shape, f"Shape mismatch: {shap_values_to_use.shape} vs {X_test.shape}"


# Répertoire de sauvegarde des figures
fig_dir = BASE_DIR / 'outputs' / 'figures'
fig_dir.mkdir(parents=True, exist_ok=True)

# Summary plot
plt.figure()
shap.summary_plot(shap_values_to_use, X_test, show=False)
plt.savefig(fig_dir / "shap_summary_plot.png", bbox_inches='tight')
plt.close()

# Dependence plot
plt.figure()
shap.dependence_plot("Last3_Goals_Diff", shap_values_to_use, X_test, interaction_index=None, show=False)
plt.savefig(fig_dir / "shap_dependence_plot.png", bbox_inches='tight')
plt.close()

if isinstance(explainer.expected_value, list):
    expected_value_to_use = explainer.expected_value[1]
else:
    expected_value_to_use = explainer.expected_value

force_plot_html = shap.force_plot(
    expected_value_to_use,
    shap_values_to_use[0, :],
    X_test.iloc[0, :],
    matplotlib=False
)

shap.save_html(str(fig_dir / "shap_force_plot.html"), force_plot_html)

print("Analyse SHAP terminée. Graphiques sauvegardés dans :", fig_dir)
