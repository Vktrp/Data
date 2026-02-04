import pandas as pd
import numpy as np
import pickle
import holidays

def load_ai_model():
    """Charge les modèles multi-horizon sauvegardés."""
    try:
        with open('models_multi_horizon.pkl', 'rb') as f:
            models = pickle.load(f)
        return models
    except FileNotFoundError:
        return None

def predict_next_7_days_with_ai(df_admissions, models_dict, scenario="Aucun", impact_adm=0):
    """
    Prédit les 7 prochains jours en utilisant les MÊMES features que l'entraînement.
    """
    if models_dict is None:
        # Fallback si pas de modèle : moyenne simple
        mean_adm = df_admissions['nb_admissions'].tail(30).mean()
        return [mean_adm * (1 + impact_adm/100)] * 7
    
    # 1. Préparation des données (Copie propre)
    df = df_admissions.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # =========================================================================
    # RECONSTRUCTION EXACTE DES FEATURES (Comme dans model_prediction.py)
    # =========================================================================
    
    fr_holidays = holidays.France()

    # Jours fériés
    df["is_holiday"] = df["date"].isin(fr_holidays).astype(int)
    df["veille_holiday"] = df["date"].isin(
        [d - pd.Timedelta(days=1) for d in fr_holidays if d in df["date"].values]
    ).astype(int)
    df["lendemain_holiday"] = df["date"].isin(
        [d + pd.Timedelta(days=1) for d in fr_holidays if d in df["date"].values]
    ).astype(int)
    
    # GESTION COVID (Indispensable pour que le modèle s'y retrouve)
    df["is_covid"] = 0
    df.loc[(df["date"] >= "2020-03-01") & (df["date"] <= "2021-06-30"), "is_covid"] = 1

    # Temporel
    df["jour_semaine"] = df["date"].dt.dayofweek
    df["mois"] = df["date"].dt.month
    df["jour_annee"] = df["date"].dt.dayofyear
    df["semaine_annee"] = df["date"].dt.isocalendar().week.astype(int)
    df["annee"] = df["date"].dt.year  # <--- Manquait avant
    
    df["weekend"] = (df["jour_semaine"] >= 5).astype(int)
    df["lundi"] = (df["jour_semaine"] == 0).astype(int)
    df["decembre"] = (df["mois"] == 12).astype(int)
    df["hiver"] = df["mois"].isin([11, 12, 1, 2]).astype(int)
    
    # Cycles
    df["sin_annee"] = np.sin(2 * np.pi * df["jour_annee"] / 365)
    df["cos_annee"] = np.cos(2 * np.pi * df["jour_annee"] / 365)
    df["sin_semaine"] = np.sin(2 * np.pi * df["jour_semaine"] / 7)
    df["cos_semaine"] = np.cos(2 * np.pi * df["jour_semaine"] / 7)
    
    # Lags (Mémoire) - DOIT INCLURE 364/365
    for lag in [1, 7, 14, 21, 28, 364, 365]:
        df[f"lag_{lag}"] = df["nb_admissions"].shift(lag)
    
    # Moyennes Glissantes
    for w in [7, 14, 30, 60]:
        df[f"roll_mean_{w}"] = df["nb_admissions"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"] = df["nb_admissions"].shift(1).rolling(w).std()
    
    # Tendance
    df["trend_7_30"] = df["roll_mean_7"] - df["roll_mean_30"]
    # Note : trend_30_60 a été retiré du modèle final, on ne le met pas ici.
    
    # Nettoyage des NaN créés par les lags
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # LISTE EXACTE DES FEATURES ATTENDUES PAR LE MODÈLE
    features = [
        "jour_semaine", "semaine_annee", "weekend", "lundi",
        "decembre", "hiver", "annee",
        "is_holiday", "veille_holiday", "lendemain_holiday",
        "is_covid",  # <--- La feature qui manquait
        "sin_annee", "cos_annee", "sin_semaine", "cos_semaine",
        "lag_1", "lag_7", "lag_14", "lag_21", "lag_28", "lag_364", "lag_365",
        "roll_mean_7", "roll_mean_14", "roll_mean_30", "roll_mean_60",
        "roll_std_7", "roll_std_30",
        "trend_7_30"
    ]
    
    # On prend la toute dernière ligne disponible (Aujourd'hui) pour prédire demain
    if len(df) == 0:
        return [0] * 7 # Sécurité si dataframe vide
        
    last_row = df.iloc[-1]
    
    # Boucle de prédiction pour les 7 modèles (J+1 à J+7)
    predictions = []
    for h in range(1, 8):
        if h in models_dict:
            # Prédiction brute
            model = models_dict[h]
            # Reshape pour que sklearn soit content (1 ligne, N colonnes)
            feat_values = last_row[features].values.reshape(1, -1)
            
            try:
                pred = model.predict(feat_values)[0]
            except Exception as e:
                # Fallback en cas d'erreur interne du modèle
                pred = last_row["roll_mean_7"]

            # =====================================================
            # APPLICATION DU SCÉNARIO DE CRISE (SIMULATION)
            # =====================================================
            # On applique le pourcentage d'impact choisi dans le dashboard
            if scenario != "Aucun":
                pred = pred * (1 + impact_adm / 100)
            
            predictions.append(max(pred, 0)) # Pas de prédiction négative
        else:
            # Si le modèle J+h n'existe pas, on met la moyenne
            predictions.append(last_row["roll_mean_7"])
            
    return predictions

def get_ai_prediction_single_day(df_admissions, models_dict, horizon=1):
    """Wrapper pour récupérer juste un jour spécifique"""
    preds = predict_next_7_days_with_ai(df_admissions, models_dict)
    if preds and len(preds) >= horizon:
        return preds[horizon-1]
    return 0