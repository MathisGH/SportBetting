import pandas as pd
import os
from pathlib import Path

# --- Chargement ---
BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / 'data' / "processed" / 'dataset_final.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# --- Renommage des colonnes ---
data.rename(columns={
    'Div': 'LeagueDivision', 'Date': 'MatchDate', 'Time': 'MatchTime', 
    'HomeTeam': 'HomeTeam', 'AwayTeam': 'AwayTeam', 'FTHG': 'HomeGoals',
    'FTAG': 'AwayGoals', 'Res': 'Result', 'HTHG': 'HomeHTGoals', 
    'HTAG': 'AwayHTGoals', 'HTR': 'HalfTimeResult', 'HS': 'HomeShots', 
    'AS': 'AwayShots', 'HST': 'HomeShotsOnTarget', 'AST': 'AwayShotsOnTarget', 
    'HHW': 'HomeHitWoodwork', 'AHW': 'AwayHitWoodwork', 'HC': 'HomeCorners', 
    'AC': 'AwayCorners', 'HF': 'HomeFouls', 'AF': 'AwayFouls',
    'HFKC': 'HomeFreeKicksConceded', 'AFKC': 'AwayFreeKicksConceded', 
    'HO': 'HomeOffsides', 'AO': 'AwayOffsides', 'HY': 'HomeYellowCards', 
    'AY': 'AwayYellowCards', 'HR': 'HomeRedCards', 'AR': 'AwayRedCards', 
    'HBP': 'HomeBookingsPoints', 'ABP': 'AwayBookingsPoints'
}, inplace=True)

# --- Nettoyage ---
data["MatchDate"] = pd.to_datetime(data["MatchDate"], errors="coerce", utc=True).dt.tz_localize(None)
columns_to_drop = [
    'MatchTime', 'LeagueDivision', 'HalfTimeResult', 'MaxCAHA', 'AvgCAHH', 'AvgCAHA', 'PCAHH', 'PCAHA', 'MaxCAHH',
    'B365CAHH', 'B365CAHA', 'AHCh', 'MaxC<2.5', 'AvgC>2.5', 'AvgC<2.5', 'PC>2.5', 'PC<2.5', 'MaxC>2.5',
    'AvgCA', 'B365C>2.5', 'B365C<2.5', 'MaxCA', 'AvgCH', 'AvgCD', 'VCCA', 'MaxCH', 'MaxCD', 'WHCA', 'VCCH',
    'VCCD', 'PSCA', 'WHCH', 'WHCD', 'IWCA', 'PSCH', 'PSCD', 'BWCA', 'IWCH', 'IWCD', 'B365CA', 'BWCH',
    'BWCD', 'B365CH', 'B365CD', '1XBH', '1XBD', '1XBA', 'BFE>2.5', 'BFE<2.5', 'BFEAHH', 'BFEAHA', 'BFCH',
    'BFCD', 'BFCA', '1XBCH', '1XBCD', '1XBCA', 'BFECH', 'BFECD', 'BFECA', 'BFEC>2.5', 'BFEC<2.5', 'BFECAHH',
    'BFECAHA', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'BFH', 'BFD', 'BFA', 'PSH', 'PSD', 'PSA',
    'WHH', 'WHD', 'WHA', 'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA', 'BFEH', 'BFED', 'BFEA', 'B365>2.5',
    'B365<2.5', 'P>2.5', 'P<2.5', 'PAHH', 'Max>2.5', 'Max<2.5', 'Avg>2.5', 'Avg<2.5', 'AHh', 'B365AHH',
    'B365AHA', 'HomeFouls', 'AwayFouls', 'PAHA', 'MaxAHH', 'MaxAHA', 'AvgAHH', 'AvgAHA', 'SourceFile',
    'Competition', 'Status', 'HomeYellowCards', 'AwayYellowCards', 'HomeRedCards', 'AwayRedCards'
]
data.drop(columns=columns_to_drop, errors='ignore', inplace=True)

# --- Normalisation des noms d'équipes ---
team_map = { 'Olympique de Marseille': 'Marseille', 'Olympique Lyonnais': 'Lyon', 'AS Saint-Étienne': 'St Etienne',
    'FC Nantes': 'Nantes', 'Stade Brestois 29': 'Brest', 'OGC Nice': 'Nice', 'Racing Club de Lens': 'Lens',
    'Le Havre AC': 'Le Havre', 'Stade de Reims': 'Reims', 'AJ Auxerre': 'Auxerre', 'Toulouse FC': 'Toulouse',
    'AS Monaco FC': 'Monaco', 'RC Strasbourg Alsace': 'Strasbourg', 'Angers SCO': 'Angers', 'Lille OSC': 'Lille',
    'Montpellier HSC': 'Montpellier', 'Stade Rennais FC 1901': 'Rennes', 'Paris Saint-Germain FC': 'Paris SG',
    'Troyes': 'Troyes', 'Clermont': 'Clermont', 'Lorient': 'Lorient', 'Metz': 'Metz', 'Ajaccio': 'Ajaccio', 'Bordeaux': 'Bordeaux'
}
data['HomeTeam'] = data['HomeTeam'].map(team_map).fillna(data['HomeTeam'])
data['AwayTeam'] = data['AwayTeam'].map(team_map).fillna(data['AwayTeam'])

# --- Encodages ---
data['CodeHomeTeam'] = data['HomeTeam'].astype('category').cat.codes
data['CodeAwayTeam'] = data['AwayTeam'].astype('category').cat.codes
data['FTR_encoded'] = data['FTR'].map({'H': 0, 'D': 0, 'A': 1})

# --- Création des colonnes nécessaires aux features ---
data['HomeGoalsConceded'] = data['AwayGoals']
data['AwayGoalsConceded'] = data['HomeGoals']
data['HomeShotAccuracy'] = data['HomeShotsOnTarget'] / data['HomeShots']
data['AwayShotAccuracy'] = data['AwayShotsOnTarget'] / data['AwayShots']
data[['HomeShotAccuracy', 'AwayShotAccuracy']] = data[['HomeShotAccuracy', 'AwayShotAccuracy']].fillna(0)

# Feature Engineering
def add_recent_form_features(df):
    for feature in ['Goals', 'GoalsConceded', 'ShotsOnTarget', 'ShotAccuracy', 'Corners']:
        home_series = df.groupby('CodeHomeTeam')[f'Home{feature}'].apply(
            lambda x: x.shift(1).rolling(3).mean()
        ).reset_index(level=0, drop=True)

        away_series = df.groupby('CodeAwayTeam')[f'Away{feature}'].apply(
            lambda x: x.shift(1).rolling(3).mean()
        ).reset_index(level=0, drop=True)

        df[f'Last3_{feature}_Diff'] = home_series.subtract(away_series, fill_value=0)

    return df


data = add_recent_form_features(data)

# Nettoyage final
data.drop(columns=['HomeGoals', 'AwayGoals', 'HomeGoalsConceded', 'AwayGoalsConceded', 
                   'HomeShotsOnTarget', 'AwayShotsOnTarget', 'HomeShotAccuracy', 'AwayShotAccuracy', 
                   'HomeCorners', 'AwayCorners'], errors='ignore', inplace=True)

data.drop_duplicates(inplace=True)
data.fillna(0, inplace=True)

# Sauvegarde
os.remove(file_path)
data.to_excel(file_path, index=False)
print("Données nettoyées et enrichies sauvegardées !")
