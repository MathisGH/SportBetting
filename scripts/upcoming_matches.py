from shutil import move
import requests
import pandas as pd
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Supprime l'ancien dataset fusionné s'il existe
dataset_final_path = BASE_DIR / "data" / "processed" / "dataset_final.xlsx"
if os.path.exists(dataset_final_path):
    os.remove(dataset_final_path)

# Dossier de stockage des fichiers
folder_path = BASE_DIR / "data" / "raw"
output_file = os.path.join(folder_path, "prochains_matchs.xlsx")

# API Football-Data.org (Ligue 1 uniquement)
API_TOKEN = "ad122d7c45d84b839143b7593a4daaa2"
URL = "https://api.football-data.org/v4/competitions/FL1/matches"

headers = {"X-Auth-Token": API_TOKEN}
response = requests.get(URL, headers=headers)

if response.status_code == 200:
    data = response.json()
    matches = data["matches"]
    
    # Filtre les matchs à venir et structure les données
    df_futurs = pd.DataFrame([
        {
            "Date": match["utcDate"],
            "HomeTeam": match["homeTeam"]["name"],
            "AwayTeam": match["awayTeam"]["name"],
            "Competition": match["competition"]["name"],
            "Status": match["status"]
        }
        for match in matches if match["status"] == "SCHEDULED"
    ])

    # Sauvegarde en Excel avec noms de colonnes standardisés
    df_futurs.to_excel(output_file, index=False)
    print(f"Prochains matchs de Ligue 1 enregistrés dans : {output_file}")

else:
    print(f"Erreur lors de la récupération ({response.status_code})")

# Charge les fichiers historiques et futurs
historical_file = os.path.join(folder_path, "merged_data.xlsx")

if os.path.exists(historical_file):
    df_historique = pd.read_excel(historical_file)
    df_futurs = pd.read_excel(output_file)

    # Correction des formats de date
    def clean_dates(df, column, format_old=None):
        df[column] = df[column].astype(str)  # Convertir en string

        # Cas des nouvelles dates (format API "AAAA-MM-JJTHH:MM:SSZ")
        df[column] = df[column].apply(lambda x: x.split("T")[0] if "T" in x else x)

        # Cas des anciennes dates (format "JJ/MM/AAAA")
        df[column] = pd.to_datetime(df[column], format=format_old, errors="coerce")

        return df

    # Applique la conversion avec les bons formats
    df_historique = clean_dates(df_historique, "Date", format_old="%d/%m/%Y")
    df_futurs = clean_dates(df_futurs, "Date", format_old="%Y-%m-%d")

    # Fusionne les datasets
    df_final = pd.concat([df_historique, df_futurs], ignore_index=True)
    df_final = df_final.dropna(subset=["Date"])
    df_final = df_final.sort_values(['Date'])

    # Sauvegarde du fichier fusionné
    final_file = os.path.join(folder_path, "dataset_final.xlsx")
    df_final.to_excel(final_file, index=False)

    print(f"\nDonnées historiques et prochains matchs fusionnés : {final_file}")

# Nettoyage des fichiers temporaires
os.remove(output_file)  # Supprimer le fichier des prochains matchs
move(final_file, dataset_final_path)  # Déplacer le fichier final dans le dossier définitif
