import os
import pandas as pd
import requests
from pathlib import Path

# Dossier où enregistrer les fichiers
BASE_DIR = Path(__file__).resolve().parent.parent # Chemin vers la racine du projet
folder_path = BASE_DIR / "data" # Chemin vers le dossier data

# URL du fichier CSV de la saison actuelle (Ligue 1, saison 24/25)
season_csv_url = "https://www.football-data.co.uk/mmz4281/2425/F1.csv"
season_csv_path = os.path.join(folder_path, "current_season.csv")

# Télécharge le CSV de la saison en cours
def download_current_season():
    response = requests.get(season_csv_url)
    if response.status_code == 200: # Vérifie que la requête a réussi
        with open(season_csv_path, "wb") as f: # Ouvre un fichier en mode écriture binaire
            f.write(response.content) # Écrit le contenu du fichier téléchargé dans current_season.csv
        print(f"Fichier de la saison actuelle téléchargé : {season_csv_path}")
    else:
        print(f"Erreur lors du téléchargement ({response.status_code})")

# Convertit les fichiers CSV en Excel et fusionne avec les fichiers existants
def convert_and_merge_csv_to_excel(folder_path):
    converted_folder = os.path.join(folder_path, "converted_excel")
    os.makedirs(converted_folder, exist_ok=True) # Évite une erreur si le dossier existe déjà

    merged_data = []  # Stocke les fichiers pour fusion

    # Convertit tous les fichiers CSV en Excel et fusionne
    for file_name in os.listdir(folder_path): # Parcourt tous les fichiers du dossier
        if file_name.endswith(".csv"): # Vérifie que c'est un fichier CSV
            csv_file_path = os.path.join(folder_path, file_name)
            excel_file_path = os.path.join(converted_folder, file_name.replace(".csv", ".xlsx"))

            try:
                data = pd.read_csv(csv_file_path, delimiter=",")

                # Ajoute une colonne indiquant la source (nom du fichier)
                data['SourceFile'] = file_name

                # Sauvegarde en Excel
                data.to_excel(excel_file_path, index=False)
                print(f"Converti : {file_name} -> {excel_file_path}")

                # Ajoute les données pour fusion
                merged_data.append(data)

            except Exception as e:
                print(f"Erreur lors de la conversion de {file_name} : {e}")

    # Inclut aussi les fichiers Excel des saisons précédentes
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".xlsx") and file_name != "merged_data.xlsx":
            excel_file_path = os.path.join(folder_path, file_name)
            try:
                data = pd.read_excel(excel_file_path)
                merged_data.append(data)
                print(f"Ajouté à la fusion : {file_name}")
            except Exception as e:
                print(f"Erreur lors de la lecture de {file_name} : {e}")

    # Fusionne toutes les données et sauvegarde dans un seul fichier Excel
    if merged_data:
        merged_df = pd.concat(merged_data, ignore_index=True)
        merged_file_path = os.path.join(folder_path, "merged_data.xlsx")
        merged_df.to_excel(merged_file_path, index=False)
        print(f"Fichier fusionné créé : {merged_file_path}")


download_current_season()  # Télécharge la saison actuelle
convert_and_merge_csv_to_excel(folder_path)  # Converti et fusionne

# On déplace current_season pour ne pas qu'il dérange plus tard
current_season_path = BASE_DIR / "data" / "current_season.csv"
os.remove(current_season_path)