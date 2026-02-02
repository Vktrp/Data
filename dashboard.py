import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

"""
DASHBOARD ÉPURÉ - VERSION FINALE
Focus : Graphiques + ML + Interface claire
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
        st.error(f"❌ Fichier manquant: {e}")
        st.stop()

df_adm, df_beds, df_pred = load_data()

# =============================================================================
# SIDEBAR ÉPURÉE
# =============================================================================

st.sidebar.header("⚙️ Paramètres")

# Mode hospitalier avec VRAIE différence
mode = st.sidebar.radio(
    "Capacité hospitalière:",
    ("Mode Normal", "Plan Blanc Activé"),
    help="Le Plan Blanc augmente la capacité de 1800 à 2500 lits"
)

# Capacité selon le mode
if "Plan Blanc" in mode:
    capacite = 2500
    st.sidebar.error("🚨 **PLAN BLANC ACTIF**")
    st.sidebar.markdown("""
    **Mesures d'urgence :**
    - Capacité : 1800 → 2500 lits
    - Rappel du personnel
    - Déprogrammation opérations
    """)
else:
    capacite = 1800
    st.sidebar.success("✅ Fonctionnement normal")

st.sidebar.markdown("---")

# Paramètres d'affichage UTILES
nb_jours = st.sidebar.slider(
    "Historique (jours):",
    min_value=30, max_value=365, value=90, step=30
)

show_confidence = st.sidebar.checkbox("Intervalles de confiance", value=True)

# =============================================================================
# KPI
# =============================================================================

st.header("📊 Indicateurs")

# Calculs avec VRAIE occupation
last_day_adm = int(df_adm.iloc[-1]['nb_admissions'])
last_day_beds = int(df_beds.iloc[-1]['lits_occupees'])

# CRITIQUE : Calculer selon la capacité choisie
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
    # Afficher la capacité utilisée
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
    # Calcul du risque SELON la capacité
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
# GRAPHIQUES
# =============================================================================

tab1, tab2, tab3 = st.tabs(["📊 Admissions", "🛏️ Occupation", "🔮 Prévisions"])

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

# TAB 2: LITS (avec ligne Plan Blanc)
with tab2:
    st.subheader("Occupation des Lits")
    
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    df_beds_recent = df_beds.tail(nb_jours)
    
    ax2.bar(df_beds_recent['date'], df_beds_recent['lits_occupees'],
            color='#3498DB', alpha=0.7, label='Lits occupés')
    
    # Ligne capacité SELON le mode
    ax2.axhline(capacite, color='red', linestyle='-', 
                linewidth=3, label=f'Capacité actuelle: {capacite} lits')
    
    # Si Plan Blanc, montrer aussi la capacité normale
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
    st.subheader("🔮 Prévisions IA - 7 jours")
    
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
    
    # Tableau simple
    st.subheader("📋 Détail")
    df_display = df_pred.copy()
    df_display['Jour'] = df_display['date'].dt.strftime('%A %d/%m')
    df_display['Prévision'] = df_display['pred_admissions'].apply(lambda x: f"{x:.0f}")
    st.dataframe(df_display[['Jour', 'Prévision']], width='stretch', hide_index=True)

# =============================================================================
# ALERTES (seulement si pertinent)
# =============================================================================

st.markdown("---")

if lits_dispo < 100:
    st.error(f"🚨 **ALERTE CRITIQUE** : Seulement {lits_dispo} lits disponibles ! → Activer Plan Blanc immédiatement")
elif lits_dispo < 300:
    st.warning(f"⚠️ **TENSION** : {lits_dispo} lits disponibles → Surveiller de près")
elif pred_j1 > last_day_adm * 1.2:
    st.warning(f"📈 **HAUSSE PRÉVUE** : +{((pred_j1/last_day_adm - 1) * 100):.0f}% demain")

# Footer épuré
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; font-size: 12px;'>
    Dashboard propulsé par Machine Learning (Gradient Boosting) | Pitié-Salpêtrière
</div>
""", unsafe_allow_html=True)