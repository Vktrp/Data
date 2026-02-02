import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

"""
DASHBOARD STREAMLIT - VERSION CORRIGÉE
Avec affichage des événements optimisé
"""

st.set_page_config(
    page_title="Pitié-Salpêtrière", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏥"
)

# Style CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 38px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏥 Pilotage des Urgences - Pitié-Salpêtrière</div>', 
            unsafe_allow_html=True)

# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================

@st.cache_data
def load_data():
    """Charge tous les fichiers CSV"""
    try:
        adm = pd.read_csv("admissions_daily.csv", parse_dates=["date"])
        beds = pd.read_csv("beds.csv", parse_dates=["date"])
        pred = pd.read_csv("previsions_future.csv", parse_dates=["date"])
        
        # Debug
        st.sidebar.success(f"✅ Admissions: {len(adm)} jours")
        st.sidebar.success(f"✅ Lits: {len(beds)} jours")
        st.sidebar.success(f"✅ Prévisions: {len(pred)} jours")
        
        staff = None
        stocks = None
        
        if os.path.exists("staff.csv"):
            staff = pd.read_csv("staff.csv", parse_dates=["date"])
            st.sidebar.success(f"✅ Staff: {len(staff)} jours")
        
        if os.path.exists("stocks.csv"):
            stocks = pd.read_csv("stocks.csv", parse_dates=["date"])
            st.sidebar.success(f"✅ Stocks: {len(stocks)} jours")
        
        return adm, beds, pred, staff, stocks
        
    except FileNotFoundError as e:
        st.error(f"❌ Fichier manquant: {e}")
        st.info("💡 Exécutez d'abord: python3 RUN_ALL.py")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erreur de chargement: {e}")
        st.stop()

# Charger les données
with st.spinner("Chargement des données..."):
    df_adm, df_beds, df_pred, df_staff, df_stocks = load_data()

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.header("🎛️ Gestion de Crise")

mode_hopital = st.sidebar.radio(
    "Mode de fonctionnement:",
    ("🟢 Standard (1800 lits)", "🔴 Plan Blanc (2500 lits)")
)

capacite_actuelle = 2500 if "Plan Blanc" in mode_hopital else 1800

if "Plan Blanc" in mode_hopital:
    st.sidebar.error("🚨 PLAN BLANC ACTIVÉ")

st.sidebar.markdown("---")

st.sidebar.header("📊 Paramètres")
nb_jours_historique = st.sidebar.slider(
    "Jours d'historique:",
    min_value=30, max_value=365, value=260, step=10
)

show_confidence = st.sidebar.checkbox("Intervalles de confiance", value=True)
show_events = st.sidebar.checkbox("Marquer les événements", value=True)

# =============================================================================
# KPI PRINCIPAUX
# =============================================================================

st.header("📊 Indicateurs Clés")

# Calculs
try:
    last_day_adm = int(df_adm.iloc[-1]['nb_admissions'])
    last_day_beds = int(df_beds.iloc[-1]['lits_occupees'])
    lits_dispo = capacite_actuelle - last_day_beds
    taux_occupation = (last_day_beds / capacite_actuelle) * 100
    
    pred_j1 = int(df_pred.iloc[0]['pred_admissions'])
    pred_min = int(df_pred.iloc[0].get('pred_min', pred_j1 - 10))
    pred_max = int(df_pred.iloc[0].get('pred_max', pred_j1 + 10))
    
    adm_j7 = int(df_adm.iloc[-8]['nb_admissions'])
    delta_adm = last_day_adm - adm_j7
    
except Exception as e:
    st.error(f"Erreur calcul KPI: {e}")
    st.stop()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Admissions Hier",
        f"{last_day_adm} patients",
        delta=f"{delta_adm:+d} vs J-7",
        delta_color="inverse"
    )

with col2:
    st.metric(
        "Lits Disponibles",
        f"{lits_dispo}",
        delta=f"{taux_occupation:.1f}% occupés",
        delta_color="inverse" if lits_dispo < 100 else "normal"
    )

with col3:
    st.metric(
        "Prévision IA J+1",
        f"{pred_j1} patients",
        delta=f"[{pred_min}-{pred_max}]",
        delta_color="off"
    )

with col4:
    beds_prevues = last_day_beds + (pred_j1 - last_day_adm)
    risque = (beds_prevues / capacite_actuelle) * 100
    
    if risque > 95:
        alerte = "🔴 CRITIQUE"
    elif risque > 85:
        alerte = "🟠 ÉLEVÉ"
    else:
        alerte = "🟢 NORMAL"
    
    st.metric("Risque Saturation J+1", alerte, delta=f"{risque:.1f}%")

st.markdown("---")

# =============================================================================
# GRAPHIQUES
# =============================================================================

st.header("📈 Analyse et Prévisions")

tab1, tab2, tab3 = st.tabs(["📊 Admissions", "🛏️ Lits", "🔮 Prévisions"])

# TAB 1: ADMISSIONS (CORRIGÉ)
with tab1:
    st.subheader("Évolution des Admissions")
    
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    df_recent = df_adm.tail(nb_jours_historique)
    
    # Courbe
    ax1.plot(df_recent['date'], df_recent['nb_admissions'], 
             label='Admissions', color='#3498DB', linewidth=2.5)
    
    # Moyenne mobile
    rolling = df_recent['nb_admissions'].rolling(7).mean()
    ax1.plot(df_recent['date'], rolling, 
             label='Moyenne 7j', color='#E74C3C', linestyle='--', linewidth=2)
    
    # Événements (OPTIMISÉ - marque seulement les débuts de période)
    if show_events:
        # Grouper les événements consécutifs
        df_recent['event_group'] = (
            (df_recent['event'] != df_recent['event'].shift()) & 
            (df_recent['event'] != 'none') & 
            (df_recent['event'].notna())
        )
        
        events_debut = df_recent[df_recent['event_group']]
        
        if not events_debut.empty:
            # Marquer seulement les débuts de période
            for _, row in events_debut.iterrows():
                ax1.axvline(x=row['date'], color='red', alpha=0.3, linestyle=':', linewidth=2)
                ax1.scatter([row['date']], [row['nb_admissions']], 
                           color='red', s=200, zorder=5, marker='*')
                # Annoter SEULEMENT les débuts
                ax1.text(row['date'], row['nb_admissions'] + 5, 
                        row['event'].upper(), 
                        ha='left', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8))
    
    ax1.set_title(f'Admissions quotidiennes ({nb_jours_historique} derniers jours)', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Admissions', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig1)
    
    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Moyenne", f"{df_recent['nb_admissions'].mean():.1f}")
    with col2:
        st.metric("Maximum", f"{df_recent['nb_admissions'].max():.0f}")
    with col3:
        st.metric("Minimum", f"{df_recent['nb_admissions'].min():.0f}")

# TAB 2: LITS
with tab2:
    st.subheader("Occupation des Lits")
    
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    df_beds_recent = df_beds.tail(nb_jours_historique)
    
    ax2.bar(df_beds_recent['date'], df_beds_recent['lits_occupees'],
            color='#3498DB', alpha=0.7, label='Lits occupés')
    
    ax2.axhline(capacite_actuelle, color='red', linestyle='-', 
                linewidth=3, label=f'Capacité ({capacite_actuelle})')
    
    ax2.set_title(f'Occupation des lits', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Nombre de lits', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig2)
    
    # Stats
    taux_moyen = (df_beds_recent['lits_occupees'].mean() / capacite_actuelle) * 100
    jours_sat = len(df_beds_recent[df_beds_recent['lits_occupees'] >= capacite_actuelle])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Taux moyen", f"{taux_moyen:.1f}%")
    with col2:
        st.metric("Jours saturation", f"{jours_sat}")
    with col3:
        st.metric("Pic", f"{df_beds_recent['lits_occupees'].max()} lits")

# TAB 3: PRÉVISIONS
with tab3:
    st.subheader("🔮 Prévisions 7 jours")
    
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
                        label='Intervalle')
    
    # Annotations
    for _, row in df_pred.iterrows():
        ax3.text(row['date'], row['pred_admissions'] + 3,
                f"{row['pred_admissions']:.0f}",
                ha='center', fontsize=10, fontweight='bold', color='#E74C3C')
    
    ax3.set_title('Prévisions 7 jours', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Admissions', fontsize=11)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig3)
    
    # Tableau
    st.subheader("📋 Détail")
    df_display = df_pred.copy()
    df_display['Jour'] = df_display['date'].dt.strftime('%A %d/%m')
    df_display['Prévision'] = df_display['pred_admissions'].apply(lambda x: f"{x:.0f} patients")
    st.dataframe(df_display[['Jour', 'Prévision']], width='stretch', hide_index=True)

# =============================================================================
# ALERTES
# =============================================================================

st.markdown("---")
st.header("🔔 Alertes")

alertes = []

if lits_dispo < 100:
    st.error(f"🛏️ **SATURATION**: Seulement {lits_dispo} lits ! → Activer Plan Blanc")
elif lits_dispo < 200:
    st.warning(f"🛏️ **TENSION**: {lits_dispo} lits disponibles → Préparer Plan Blanc")

if pred_j1 > last_day_adm * 1.2:
    st.warning(f"📈 **HAUSSE PRÉVUE**: +{((pred_j1/last_day_adm - 1) * 100):.0f}% demain → Renforcer équipes")

if not alertes and lits_dispo >= 200:
    st.success("✅ Aucune alerte - Situation sous contrôle")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <em>Pitié-Salpêtrière Data Hub | Powered by AI</em>
</div>
""", unsafe_allow_html=True)