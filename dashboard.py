import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import plotly.graph_objects as go

"""
DASHBOARD COMPLET - VERSION FINALE
Avec Simulation de Scénarios + Recommandations Automatiques
"""

st.set_page_config(
    page_title="Pitié-Salpêtrière", 
    layout="wide",
    page_icon="🏥"
)

# Style minimaliste
st.markdown("""
    <style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        padding: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏥 Pitié-Salpêtrière - Dashboard de Prédiction</div>', 
            unsafe_allow_html=True)

# =============================================================================
# CHARGEMENT
# =============================================================================

@st.cache_data
def load_data():
    try:
        adm = pd.read_csv("admissions_daily.csv", parse_dates=["date"])
        beds = pd.read_csv("beds.csv", parse_dates=["date"])
        pred = pd.read_csv("previsions_future.csv", parse_dates=["date"])
        return adm, beds, pred
    except FileNotFoundError as e:
        st.error(f"Fichier manquant: {e}")
        st.stop()

df_adm, df_beds, df_pred = load_data()

# =============================================================================
# SIDEBAR ÉPURÉE
# =============================================================================

mode = st.sidebar.radio(
    "Capacité hospitalière:",
    ("Mode Normal", "Plan Blanc Activé")
)

# Capacité selon le mode
if "Plan Blanc" in mode:
    capacite = 2500
    st.sidebar.error(" **PLAN BLANC ACTIF**")
    st.sidebar.markdown("""
    **Mesures d'urgence :**
    - Capacité : 1800 → 2500 lits
    - Rappel du personnel
    - Déprogrammation opérations
    """)
else:
    capacite = 1800
    st.sidebar.success("Fonctionnement normal")

st.sidebar.markdown("---")

nb_jours = st.sidebar.slider(
    "Historique (jours):",
    min_value=30, max_value=365, value=90, step=30
)

show_confidence = st.sidebar.checkbox("Intervalles de confiance", value=True)

# =============================================================================
# 🔬 SIMULATION DE SCÉNARIOS (NOUVEAU)
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.header("Simulation de Crise")

scenario = st.sidebar.selectbox(
    "Tester un scénario:",
    ["Aucun", "Épidémie", "Grève", "Canicule", "Grand froid", "Accident massif"]
)

# Variables pour la simulation
simulation_active = False
adm_simulees = pred_j1 = 0

if scenario != "Aucun":
    st.sidebar.warning(f"Scénario **{scenario}** activé")
    
    # Paramètres par défaut selon scénario
    if scenario == "Épidémie":
        default_adm, default_staff = 40, -20
    elif scenario == "Grève":
        default_adm, default_staff = -10, -40
    elif scenario == "Canicule":
        default_adm, default_staff = 25, -10
    elif scenario == "Grand froid":
        default_adm, default_staff = 30, -15
    else:
        default_adm, default_staff = 60, 0
    
    impact_admissions = st.sidebar.slider(
        "Impact sur admissions (%):",
        min_value=-50, max_value=100, value=default_adm, step=5
    )
    
    impact_staff = st.sidebar.slider(
        "Impact sur personnel (%):",
        min_value=-50, max_value=50, value=default_staff, step=5
    )
    
    duree_crise = st.sidebar.slider(
        "Durée de la crise (jours):",
        min_value=1, max_value=30, value=7, step=1
    )
    
    simulation_active = st.sidebar.button("Lancer la simulation", type="primary")

# =============================================================================
# KPI
# =============================================================================

st.header("Indicateurs")

last_day_adm = int(df_adm.iloc[-1]['nb_admissions'])
last_day_beds = int(df_beds.iloc[-1]['lits_occupees'])

lits_dispo = capacite - last_day_beds
taux_occupation = (last_day_beds / capacite) * 100

pred_j1 = int(df_pred.iloc[0]['pred_admissions'])
pred_min = int(df_pred.iloc[0].get('pred_min', pred_j1 - 30))
pred_max = int(df_pred.iloc[0].get('pred_max', pred_j1 + 30))

adm_j7 = int(df_adm.iloc[-8]['nb_admissions'])
delta_adm = last_day_adm - adm_j7

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Admissions hier",
        f"{last_day_adm}",
        delta=f"{delta_adm:+d} vs J-7",
        delta_color="inverse"
    )

with col2:
    st.metric(
        f"Lits disponibles (/{capacite})",
        f"{lits_dispo}",
        delta=f"{taux_occupation:.1f}% occupés",
        delta_color="inverse" if taux_occupation > 90 else "normal"
    )

with col3:
    st.metric(
        "Prévision IA J+1",
        f"{pred_j1}",
        delta=f"[{pred_min}-{pred_max}]",
        delta_color="off"
    )

with col4:
    beds_prevues = last_day_beds + (pred_j1 - last_day_adm)
    risque = (beds_prevues / capacite) * 100
    
    if risque > 95:
        alerte = "🔴"
        color = "inverse"
    elif risque > 85:
        alerte = "🟠"
        color = "inverse"
    else:
        alerte = "🟢"
        color = "normal"
    
    st.metric("Risque J+1", alerte, delta=f"{risque:.1f}%", delta_color=color)

st.markdown("---")

# =============================================================================
# AFFICHAGE RÉSULTATS SIMULATION
# =============================================================================

if simulation_active:
    st.header(f"Résultats Simulation : {scenario}")
    
    # Calculs simulation
    adm_simulees = pred_j1 * (1 + impact_admissions/100)
    staff_dispo = 110 * (1 + impact_staff/100)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Admissions prévues J+1",
            f"{int(adm_simulees)}",
            delta=f"{impact_admissions:+d}%",
            delta_color="inverse"
        )
    
    with col2:
        ratio_crise = adm_simulees / staff_dispo
        st.metric(
            "Patients par infirmier",
            f"{ratio_crise:.1f}",
            delta=f"Tension {'élevée' if ratio_crise > 4 else 'normale'}",
            delta_color="inverse" if ratio_crise > 4 else "normal"
        )
    
    with col3:
        impact_total = adm_simulees * duree_crise * 0.15  
        beds_necessaires = last_day_beds + impact_total
        taux_sim = (beds_necessaires / capacite) * 100
        st.metric(
            "Taux occupation prévu",
            f"{taux_sim:.1f}%",
            delta=f"{'🔴 CRITIQUE' if taux_sim > 90 else '🟠 URGENT' if taux_sim > 70 else '🟡 ATTENTION' if taux_sim > 60 else '🟢 NORMAL'}",
            delta_color="inverse" if taux_sim > 70 else "normal"
        )
    
    # Recommandations automatiques
    st.subheader("Recommandations Automatiques")
    
    if taux_sim > 90:
        st.error("🚨 **ACTION URGENTE** : Activer le Plan Blanc immédiatement")
        st.markdown("- Rappeler le personnel en congé")
        st.markdown("- Déprogrammer opérations non-urgentes")
        st.markdown("- Ouvrir lits de débordement")
        st.markdown("- Activer cellule de crise")
    elif taux_sim > 80:
        st.error("🔴 **SATURATION IMMINENTE** : Préparation Plan Blanc urgente")
        st.markdown("- Prévenir personnel de réserve (alerte)")
        st.markdown("- Vérifier lits débordement")
        st.markdown("- Reporter admissions programmées")
    elif taux_sim > 70:
        st.warning("⚠️ **TENSION CONFIRMÉE** : Mobilisation préventive")
        st.markdown("- Renforcer effectifs progressivement")
        st.markdown("- Accélérer les sorties")
        st.markdown("- Limiter admissions programmées")
    elif taux_sim > 60:
        st.info("💡 **SURVEILLANCE RENFORCÉE** : Anticiper hausse")
        st.markdown("- Surveiller quotidiennement")
        st.markdown("- Optimiser gestion des sorties")
        st.markdown("- Communiquer avec ville")
    else:
        st.success("✅ **SITUATION NORMALE** : Capacité suffisante")
    
    # Graphique projection
    st.subheader("Projection sur la durée de la crise")
    
    fig_sim, ax_sim = plt.subplots(figsize=(12, 5))
    
    jours = np.arange(duree_crise)
    admissions_proj = [adm_simulees * (1 - 0.02*j) for j in jours]
    beds_proj = [beds_necessaires + sum(admissions_proj[:i+1]) * 0.3 for i in jours]
    
    ax_sim.plot(jours, beds_proj, marker='o', linewidth=2, 
                label='Lits occupés projetés', color='#E74C3C')
    ax_sim.axhline(capacite, color='red', linestyle='--', linewidth=2, 
                   label=f'Capacité ({capacite})')
    ax_sim.fill_between(jours, beds_proj, capacite, 
                        where=(np.array(beds_proj) > capacite),
                        color='red', alpha=0.3, label='Dépassement')
    
    ax_sim.set_xlabel('Jour de la crise', fontsize=11)
    ax_sim.set_ylabel('Lits occupés', fontsize=11)
    ax_sim.set_title(f'Projection sur {duree_crise} jours - {scenario}', 
                     fontsize=13, fontweight='bold')
    ax_sim.legend()
    ax_sim.grid(True, alpha=0.3)
    plt.tight_layout()
    
    st.pyplot(fig_sim)
    
    # Coût estimé
    st.subheader("💰 Coût Estimé")
    
    jours_saturation = sum(1 for b in beds_proj if b > capacite)
    cout_saturation = jours_saturation * 50000
    cout_rappels = int(staff_dispo * 0.2) * 500
    cout_total = cout_saturation + cout_rappels
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Jours saturation", f"{jours_saturation}/{duree_crise}")
    with col2:
        st.metric("Coût saturation", f"{cout_saturation/1000:.0f}k€")
    with col3:
        st.metric("Coût total estimé", f"{cout_total/1000:.0f}k€")
    
    st.markdown("---")

# =============================================================================
# GRAPHIQUES
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Admissions", 
    "Occupation", 
    "Prévisions",
    "Recommandations"
])

# TAB 1: ADMISSIONS
with tab1:
    st.subheader("Évolution des Admissions")
    
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    df_recent = df_adm.tail(nb_jours)
    
    ax1.plot(df_recent['date'], df_recent['nb_admissions'], 
             color='#3498DB', linewidth=2.5, label='Admissions quotidiennes')
    
    rolling = df_recent['nb_admissions'].rolling(7).mean()
    ax1.plot(df_recent['date'], rolling, 
             color='#E74C3C', linestyle='--', linewidth=2, label='Moyenne 7j')
    
    ax1.set_title(f'Admissions quotidiennes ({nb_jours} derniers jours)', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Nombre d\'admissions', fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig1)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Moyenne", f"{df_recent['nb_admissions'].mean():.0f}")
    with col2:
        st.metric("Maximum", f"{df_recent['nb_admissions'].max():.0f}")
    with col3:
        st.metric("Minimum", f"{df_recent['nb_admissions'].min():.0f}")

# TAB 2: LITS
with tab2:
    st.subheader("Occupation des Lits")
    
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    df_beds_recent = df_beds.tail(nb_jours)
    
    ax2.bar(df_beds_recent['date'], df_beds_recent['lits_occupees'],
            color='#3498DB', alpha=0.7, label='Lits occupés')
    
    ax2.axhline(capacite, color='red', linestyle='-', 
                linewidth=3, label=f'Capacité actuelle: {capacite} lits')
    
    if capacite == 2500:
        ax2.axhline(1800, color='orange', linestyle='--', 
                    linewidth=2, label='Capacité normale: 1800 lits')
        ax2.fill_between(df_beds_recent['date'], 1800, 2500, 
                         alpha=0.1, color='red', label='Lits Plan Blanc')
    
    ax2.set_title(f'Occupation des lits - Capacité: {capacite}', 
                  fontsize=14, fontweight='bold')
    ax2.set_ylabel('Nombre de lits', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig2)
    
    taux_moyen = (df_beds_recent['lits_occupees'].mean() / capacite) * 100
    jours_sat = len(df_beds_recent[df_beds_recent['lits_occupees'] >= capacite])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Taux moyen", f"{taux_moyen:.1f}%")
    with col2:
        st.metric("Jours saturation", f"{jours_sat}")
    with col3:
        st.metric("Pic", f"{df_beds_recent['lits_occupees'].max()}")

# TAB 3: PRÉVISIONS
with tab3:
    st.subheader("Prévisions IA - 7 jours")
    
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    df_recent = df_adm.tail(30)
    
    ax3.plot(df_recent['date'], df_recent['nb_admissions'],
             label='Historique', color='#2C3E50', linewidth=2.5)
    
    ax3.plot(df_pred['date'], df_pred['pred_admissions'],
             label='Prévisions IA', color='#E74C3C', 
             linestyle='--', linewidth=3, marker='o', markersize=10)
    
    if show_confidence and 'pred_min' in df_pred.columns:
        ax3.fill_between(df_pred['date'], 
                        df_pred['pred_min'], 
                        df_pred['pred_max'],
                        alpha=0.3, color='#E74C3C',
                        label='Intervalle confiance')
    
    for _, row in df_pred.iterrows():
        ax3.text(row['date'], row['pred_admissions'] + 10,
                f"{row['pred_admissions']:.0f}",
                ha='center', fontsize=10, fontweight='bold', color='#E74C3C')
    
    ax3.set_title('Prévisions 7 jours', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Admissions', fontsize=11)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig3)
    
    st.subheader("📋 Détail")
    df_display = df_pred.copy()
    df_display['Jour'] = df_display['date'].dt.strftime('%A %d/%m')
    df_display['Prévision'] = df_display['pred_admissions'].apply(lambda x: f"{x:.0f}")
    st.dataframe(df_display[['Jour', 'Prévision']], width='stretch', hide_index=True)

# =============================================================================
# TAB 4: RECOMMANDATIONS AUTOMATIQUES (CORRIGÉ)
# =============================================================================

with tab4:
    st.subheader("Recommandations Automatiques Basées sur l'IA")
    
    if scenario != "Aucun":
        st.info(f"Analyse basée sur le scénario **{scenario}**")
        
        # Calculer les données de simulation (même sans bouton cliqué)
        adm_simulees_calc = pred_j1 * (1 + impact_admissions/100)
        staff_dispo_calc = 110 * (1 + impact_staff/100)
        impact_total_calc = adm_simulees_calc * duree_crise * 0.15
        beds_necessaires_calc = last_day_beds + impact_total_calc
        taux_sim_calc = (beds_necessaires_calc / capacite) * 100
        
        # Utiliser les données de SIMULATION
        stress_level = taux_sim_calc
        tendance_7j = impact_admissions
        pct_tendance = impact_admissions
        jours_avant_sat = max(1, (capacite - beds_necessaires_calc) / adm_simulees_calc) if adm_simulees_calc > 0 else 999
        
        pred_j1_analyse = adm_simulees_calc
        last_day_adm_analyse = last_day_adm
    else:
        st.info("📊 Analyse basée sur les données réelles actuelles")
        
        # Utiliser les données RÉELLES
        stress_level = (last_day_beds / capacite) * 100
        tendance_7j = (df_adm.tail(7)['nb_admissions'].mean() - 
                       df_adm.tail(14).head(7)['nb_admissions'].mean())
        pct_tendance = (tendance_7j / df_adm.tail(14).head(7)['nb_admissions'].mean()) * 100
        jours_avant_sat = (capacite - last_day_beds) / df_adm.tail(7)['nb_admissions'].mean()
        
        pred_j1_analyse = pred_j1
        last_day_adm_analyse = last_day_adm
    
    # Analyse situation
    st.markdown("### Analyse de la Situation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Tendance 7 jours",
            f"{pct_tendance:+.1f}%",
            delta="Hausse" if pct_tendance > 0 else "Baisse"
        )
    
    with col2:
        st.metric(
            "Niveau de stress",
            f"{stress_level:.0f}%",
            delta="Élevé" if stress_level > 85 else "Normal"
        )
    
    with col3:
        st.metric(
            "Jours avant saturation",
            f"{int(jours_avant_sat)}",
            delta="Critique" if jours_avant_sat < 3 else "OK"
        )
    
    st.markdown("---")
    st.markdown("### Recommandations Stratégiques")
    
    # Génération recommandations (AVEC données simulées si actif)
    recommandations = []
    
    # Seuils conformes aux standards hospitaliers français
    if stress_level > 90:
        recommandations.append({
            'niveau': 'CRITIQUE',
            'titre': 'Activation Plan Blanc Immédiate',
            'actions': [
                'Rappeler tout le personnel disponible',
                'Ouvrir les 700 lits de débordement',
                'Déprogrammer opérations non-urgentes',
                'Activer cellule de crise',
                'Prévenir ARS et hôpitaux voisins'
            ],
            'delai': 'IMMÉDIAT'
        })
    elif stress_level > 80:
        recommandations.append({
            'niveau': 'URGENT',
            'titre': 'Saturation Imminente - Préparation Plan Blanc',
            'actions': [
                f'Taux occupation: {stress_level:.0f}% (critique à 90%)',
                'Prévenir personnel de réserve (mise en alerte)',
                'Vérifier disponibilité lits de débordement',
                'Reporter admissions programmées non-urgentes',
                'Renforcer effectifs du lendemain'
            ],
            'delai': '24H'
        })
    elif stress_level > 70:
        recommandations.append({
            'niveau': 'URGENT',
            'titre': 'Tension Hospitalière Confirmée',
            'actions': [
                f'Taux occupation: {stress_level:.0f}% (seuil tension: 70%)',
                'Mobilisation préventive du personnel',
                'Accélérer les sorties possibles',
                'Limiter les admissions programmées',
                'Surveillance quotidienne renforcée'
            ],
            'delai': '48H'
        })
    elif stress_level > 60:
        recommandations.append({
            'niveau': 'ATTENTION',
            'titre': 'Surveillance Renforcée Recommandée',
            'actions': [
                f'Taux occupation: {stress_level:.0f}% (seuil surveillance: 60%)',
                'Surveiller évolution quotidienne',
                'Anticiper possible hausse',
                'Optimiser la gestion des sorties',
                'Communiquer avec médecins de ville'
            ],
            'delai': '72H'
        })
    
    if pct_tendance > 15:
        recommandations.append({
            'niveau': 'ATTENTION',
            'titre': 'Tendance Haussière Forte',
            'actions': [
                f'Hausse de {pct_tendance:.0f}% détectée',
                'Anticiper +20% semaine prochaine',
                'Renforcer effectifs progressivement',
                'Vérifier stocks critiques'
            ],
            'delai': '48H'
        })
    
    if pred_j1_analyse > last_day_adm_analyse * 1.3:
        recommandations.append({
            'niveau': 'ATTENTION',
            'titre': 'Pic Prévu',
            'actions': [
                f'Prévision: {int(pred_j1_analyse)} admissions (+{((pred_j1_analyse/last_day_adm_analyse-1)*100):.0f}%)',
                'Mobiliser +15% personnel',
                'Libérer lits dès aujourd\'hui',
                'Préparer zone d\'attente supplémentaire'
            ],
            'delai': '12H'
        })
    
    if jours_avant_sat < 3:
        recommandations.append({
            'niveau': 'URGENT',
            'titre': 'Risque Saturation Imminent',
            'actions': [
                f'Saturation dans {int(jours_avant_sat)} jours',
                'Accélérer les sorties (patients stables)',
                'Négocier transferts avec hôpitaux voisins',
                'Communiquer avec médecins de ville'
            ],
            'delai': '48H'
        })
    
    if not recommandations:
        recommandations.append({
            'niveau': 'OK',
            'titre': 'Situation Sous Contrôle',
            'actions': [
                'Surveillance quotidienne',
                'Effectifs standards suffisants',
                'Pas d\'action particulière'
            ],
            'delai': 'Routine'
        })
    
    # Affichage
    for i, reco in enumerate(recommandations, 1):
        if reco['niveau'] == 'CRITIQUE':
            st.error(f"### 🚨 #{i} - {reco['titre']}")
        elif reco['niveau'] == 'URGENT':
            st.warning(f"### ⚠️ #{i} - {reco['titre']}")
        elif reco['niveau'] == 'ATTENTION':
            st.info(f"### 💡 #{i} - {reco['titre']}")
        else:
            st.success(f"### ✅ #{i} - {reco['titre']}")
        
        st.markdown(f"**⏰ Délai : {reco['delai']}**")
        for action in reco['actions']:
            st.markdown(f"- {action}")
        st.markdown("---")
    
    # Impact
    st.markdown("### Impact Économique")
    
    if stress_level > 70:
        if scenario != "Aucun":
            # Calculer impact selon niveau
            if stress_level > 90:
                facteur_impact = 0.8
            elif stress_level > 80:
                facteur_impact = 0.6
            else:
                facteur_impact = 0.4
            
            jours_sat_estime = int(duree_crise * facteur_impact) if simulation_active else 10
            cout_sans = jours_sat_estime * 50000
            cout_avec = int(jours_sat_estime * 0.3) * 50000
            economie = cout_sans - cout_avec
            
            st.markdown(f"""
            **Impact du scénario {scenario} :**
            
            | Métrique | Sans action | Avec action | Gain |
            |----------|-------------|-------------|------|
            | Jours difficiles | {jours_sat_estime} | {int(jours_sat_estime*0.3)} | -{int((1-0.3)*100)}% |
            | Taux occupation | {stress_level:.0f}% | ~65% | -{stress_level-65:.0f}% |
            | Coût total | {cout_sans/1000:.0f}k€ | {cout_avec/1000:.0f}k€ | -{economie/1000:.0f}k€ |
            
            💰 **Économie potentielle : {economie/1000:.0f}k€** sur {duree_crise} jours
            """)
        else:
            st.markdown("""
            **Si recommandations appliquées :**
            
            | Métrique | Sans action | Avec action | Gain |
            |----------|-------------|-------------|------|
            | Taux saturation | 85%+ | 65-70% | -15-20% |
            | Temps d'attente | 3-5h | 1-2h | -60% |
            | Coût journalier | 40k€ | 15k€ | -62% |
            
            💰 **Économie : 25k€/jour**
            """)
    else:
        st.success("✅ Situation optimale. Coûts standards.")

# =============================================================================
# ALERTES
# =============================================================================

st.markdown("---")

if lits_dispo < 100:
    st.error(f"🚨 **ALERTE CRITIQUE** : {lits_dispo} lits disponibles → Plan Blanc immédiat")
elif lits_dispo < 300:
    st.error(f"🔴 **SATURATION IMMINENTE** : {lits_dispo} lits disponibles")
elif taux_occupation > 70:
    st.warning(f"⚠️ **TENSION** : {taux_occupation:.0f}% occupation → Surveiller")
elif taux_occupation > 60:
    st.info(f"💡 **SURVEILLANCE** : {taux_occupation:.0f}% occupation")
elif pred_j1 > last_day_adm * 1.2:
    st.warning(f"📈 **HAUSSE PRÉVUE** : +{((pred_j1/last_day_adm - 1) * 100):.0f}% demain")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; font-size: 12px;'>
    Dashboard propulsé par ML (Gradient Boosting) | Pitié-Salpêtrière
</div>
""", unsafe_allow_html=True)