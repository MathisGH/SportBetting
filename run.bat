@echo off
:: Appelle main.py avec un argument dynamique
:: Le modèle doit être passé manuellement via Streamlit (rf ou xgb)

:: Exemple d'appel : run.bat rf
python "main.py" --model %1
