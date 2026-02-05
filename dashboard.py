import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from PIL import Image
from ai_prediction_functions import load_ai_model, predict_next_7_days_with_ai


st.set_page_config(
    page_title="Pitié-Salpêtrière", 
    layout="wide",
)

# Configuration matplotlib dark
plt.rcParams.update({
    'axes.facecolor': '#0E1117',
    'figure.facecolor': '#0E1117',
    'grid.color': '#2a2d33',
    'text.color': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.edgecolor': '#30363D'
})

# CSS Personnalisé
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    section[data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    h1, h2, h3 { color: #58A6FF !important; }
    
    .stTabs [data-baseweb="tab-list"] { 
        gap: 8px;
        padding: 5px 0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 35px;
        background-color: #161B22;
        border: 1px solid #30363D;
        color: #8b949e;
        border-radius: 6px 6px 0 0;
        font-size: 13px;
        padding: 0 15px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb;
        border-bottom: 3px solid #58a6ff;
        color: white;
        font-weight: 600;
    }
    
    div[data-testid="stMetric"] {
        background-color: #0d1117;
        border: 1px solid #30363D;
        padding: 10px;
        border-radius: 8px;
    }
    div[data-testid="stMetricLabel"] p { color: #8b949e !important; }
    div[data-testid="stMetricValue"] div { color: #f0f6fc !important; }
    </style>
""", unsafe_allow_html=True)

st.title("Pitié-Salpêtrière • Dashboard")

# =============================================================================
# CHARGEMENT
# =============================================================================

@st.cache_data
def load_data():
    try:
        # On charge tout
        adm = pd.read_csv("admissions_daily.csv", parse_dates=["date"])
        beds = pd.read_csv("beds.csv", parse_dates=["date"])
        pred = pd.read_csv("previsions_future.csv", parse_dates=["date"])
        
        # Nouveaux fichiers
        stocks = pd.read_csv("stocks.csv", parse_dates=["date"])
        staff = pd.read_csv("staff.csv", parse_dates=["date"])
        
        
        return adm, beds, pred, stocks, staff
    except FileNotFoundError as e:
        st.error(f"Fichier manquant: {e}")
        st.stop()

# On récupère les 5 DataFrames
df_adm, df_beds, df_pred, df_stocks, df_staff = load_data()

ai_models = load_ai_model()

# =============================================================================
# SIDEBAR
# =============================================================================


mode = st.sidebar.radio(
    "Capacité:",
    ("Mode Normal", "Plan Blanc Activé")
)

capacite = 2500 if "Plan Blanc" in mode else 1800

if "Plan Blanc" in mode:
    st.sidebar.error("**PLAN BLANC ACTIF**")
else:
    st.sidebar.success("Normal")

show_confidence = st.sidebar.checkbox("Intervalles confiance", value=True)

# =============================================================================
# ONGLETS PRINCIPAUX
# =============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Vue d'Ensemble",
    "Admissions & Lits",
    "Prévisions IA",
    "Simulation & Recommandations",
    "Analyses Opérationnelles",
    "Analyses Statistiques"
])

# Calculs KPI
last_day_adm = int(df_adm.iloc[-1]['nb_admissions'])
last_day_beds = int(df_beds.iloc[-1]['lits_occupees'])
lits_dispo = capacite - last_day_beds
taux_occupation = (last_day_beds / capacite) * 100
pred_j1 = int(df_pred.iloc[0]['pred_admissions'])
pred_min = int(df_pred.iloc[0].get('pred_min', pred_j1 - 30))
pred_max = int(df_pred.iloc[0].get('pred_max', pred_j1 + 30))
adm_j7 = int(df_adm.iloc[-8]['nb_admissions'])
delta_adm = last_day_adm - adm_j7

# =============================================================================
# TAB 1: VUE D'ENSEMBLE
# =============================================================================

with tab1:
    st.header("Indicateurs Clés")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Admissions Hier", f"{last_day_adm}", delta=f"{delta_adm:+d} vs J-7")
    
    with col2:
        st.metric(f"Lits Disponibles /{capacite}", f"{lits_dispo}",
                 delta=f"{taux_occupation:.1f}% occupés",
                 delta_color="inverse" if taux_occupation > 90 else "normal")
    
    with col3:
        st.metric("Prévision IA J+1", f"{pred_j1}", delta=f"[{pred_min}-{pred_max}]")
    
    with col4:
        beds_prevues = last_day_beds + (pred_j1 - last_day_adm)
        risque = (beds_prevues / capacite) * 100
        alerte = "🔴" if risque > 95 else "🟠" if risque > 85 else "🟢"
        st.metric("Risque J+1", alerte, delta=f"{risque:.1f}%")
    
    st.markdown("---")
    
    # Alertes
    st.subheader("Alertes")
    
    if lits_dispo < 100:
        st.error(f"🔴 **CRITIQUE** : {lits_dispo} lits disponibles → Plan Blanc immédiat")
    elif lits_dispo < 300:
        st.warning(f"🟠 **ATTENTION** : {lits_dispo} lits disponibles")
    elif taux_occupation > 70:
        st.info(f"🟡 **SURVEILLANCE** : {taux_occupation:.0f}% occupation")
    else:
        st.success("Situation normale")
    
    if pred_j1 > last_day_adm * 1.2:
        st.warning(f"**HAUSSE PRÉVUE** : +{((pred_j1/last_day_adm - 1) * 100):.0f}% demain")

# =============================================================================
# TAB 2: ADMISSIONS & LITS
# =============================================================================

with tab2:
    st.header("Admissions & Lits & Stock")
    
    # Sélecteur de période
    nb_jours = st.select_slider(
        "Historique à afficher :",
        options=[30, 60, 90, 180, 365, "Tout"],
        value=90
    )
    
    if nb_jours != "Tout":
        df_recent = df_adm.tail(nb_jours)
        df_beds_recent = df_beds.tail(nb_jours)
        df_staff_recent = df_staff.tail(nb_jours)
    else:
        df_recent = df_adm
        df_beds_recent = df_beds
        df_staff_recent = df_staff

    # --- GRAPHIQUE 1 : FLUX D'ADMISSIONS ---
    st.subheader("1. Flux d'Admissions Quotidien")
    
    fig_adm = go.Figure()
    fig_adm.add_trace(go.Scatter(x=df_recent['date'], y=df_recent['nb_admissions'],
                                 mode='lines', name='Admissions', line=dict(color='#3498DB', width=2)))
    fig_adm.add_trace(go.Scatter(x=df_recent['date'], y=df_recent['nb_admissions'].rolling(7).mean(),
                                 mode='lines', name='Moyenne 7j', line=dict(color='#E74C3C', width=2, dash='dash')))
    
    fig_adm.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_adm, use_container_width=True)

    st.markdown("---")

    # --- GRAPHIQUE 2 : OCCUPATION DES LITS ---
    st.subheader("2. Taux d'Occupation des Lits")
    
    fig_beds = go.Figure()
    
    fig_beds.add_trace(go.Bar(
        x=df_beds_recent['date'], 
        y=df_beds_recent['lits_occupees'],
        name='Lits Occupés',
        marker_color='#3498DB',
        opacity=0.7
    ))
    
    fig_beds.add_trace(go.Scatter(
        x=df_beds_recent['date'], 
        y=[capacite] * len(df_beds_recent),
        mode='lines',
        name='Capacité Max',
        line=dict(color='#E74C3C', width=3)
    ))

    #
    if capacite > 2000:
        fig_beds.add_trace(go.Scatter(
            x=df_beds_recent['date'],
            y=[1800] * len(df_beds_recent),
            mode='lines',
            name='Capacité Standard (1800)',
            line=dict(color='#F1C40F', width=2, dash='dot')
        ))

    fig_beds.update_layout(
        title=f"Occupation vs Capacité ({capacite} lits)",
        yaxis_title="Nombre de lits",
        template="plotly_dark",
        height=400,
        hovermode="x unified"
    )
    st.plotly_chart(fig_beds, use_container_width=True)
    
    taux_moyen = (df_beds_recent['lits_occupees'].mean() / capacite) * 100
    pic_recent = df_beds_recent['lits_occupees'].max()
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Occupation Moyenne", f"{taux_moyen:.1f}%")
    c2.metric("Pic d'occupation", f"{pic_recent} lits")
    c3.metric("Marge de sécurité min", f"{capacite - pic_recent} lits")

    st.markdown("---")

    # --- GRAPHIQUE 3 : RESSOURCES HUMAINES ---
    st.subheader("3. Effectifs Présents")

    fig_staff = px.area(
        df_staff_recent, 
        x='date', 
        y=['medecins', 'infirmiers', 'aides_soignants'],
        color_discrete_map={
            'medecins': '#e74c3c', 
            'infirmiers': '#3498db', 
            'aides_soignants': '#2ecc71'
        },
        template="plotly_dark"
    )
    fig_staff.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_staff, use_container_width=True)

# =============================================================================
# TAB 3: PRÉVISIONS IA
# =============================================================================

with tab3:
    st.header("Prévisions Intelligence Artificielle")
    
    st.markdown("---")
    
    # Graphique prévisions
    st.subheader("Prévisions 7 jours")
    
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    df_recent_prev = df_adm.tail(30)
    
    ax3.plot(df_recent_prev['date'], df_recent_prev['nb_admissions'],
             label='Historique', color='#2C3E50', linewidth=2.5)
    
    ax3.plot(df_pred['date'], df_pred['pred_admissions'],
             label='Prévisions IA', color='#E74C3C', 
             linestyle='--', linewidth=3, marker='o', markersize=10)
    
    if show_confidence and 'pred_min' in df_pred.columns:
        ax3.fill_between(df_pred['date'], df_pred['pred_min'], df_pred['pred_max'],
                        alpha=0.3, color='#E74C3C', label='Intervalle confiance')
    
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
    
    st.markdown("---")
    
    # Tableau détail
    st.subheader("Détail des Prévisions")
    df_display = df_pred.copy()
    df_display['Jour'] = df_display['date'].dt.strftime('%A %d/%m')
    df_display['Prévision'] = df_display['pred_admissions'].apply(lambda x: f"{x:.0f}")
    st.dataframe(df_display[['Jour', 'Prévision']], hide_index=True, use_container_width=True)
        
# =============================================================================
# TAB 4: SIMULATION & RECOMMANDATIONS
# =============================================================================

with tab4:
    st.header("Simulation de Crise & Recommandations")
    
    # Configuration simulation
    st.subheader("Configuration du Scénario")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        scenario = st.selectbox(
            "Scénario:",
            ["Aucun", "Épidémie", "Grève", "Canicule", "Grand froid", "Accident massif"]
        )
    
    if scenario == "Épidémie":
        default_adm = 40
        default_staff = -10
        default_duree = 21
    elif scenario == "Grève":
        default_adm = 0
        default_staff = -50
        default_duree = 3
    elif scenario == "Canicule":
        default_adm = 25
        default_staff = -5
        default_duree = 10
    elif scenario == "Grand froid":
        default_adm = 30
        default_staff = -5
        default_duree = 14
    elif scenario == "Accident massif":
        default_adm = 60
        default_staff = 0
        default_duree = 2
    else:
        default_adm = 0
        default_staff = 0
        default_duree = 7
    
    with col2:
        impact_admissions = st.slider(
            "Impact admissions (%):",
            -50, 200, default_adm, 5
        )
    
    with col3:
        impact_staff = st.slider(
            "Impact personnel (%):",
            -90, 50, default_staff, 5,
            help="Si négatif, réduit la capacité réelle de l'hôpital"
        )
    
    with col4:
        duree_crise = st.slider(
            "Durée (jours):",
            1, 30, 8, 1
        )
    
    simulation_active = st.button("Lancer la Simulation", type="primary")
    last_date_str = df_adm.iloc[-1]['date'].strftime('%d/%m/%Y')
    st.markdown("---")
    
    if scenario != "Aucun" or simulation_active:
        # 1. CALCULS INTELLIGENTS
        
        # A. Prédictions IA (Admissions)
        if ai_models:
            predictions_ai = predict_next_7_days_with_ai(
                df_admissions=df_adm,
                models_dict=ai_models,
                scenario=scenario,
                impact_adm=impact_admissions
            )
            adm_simulees_base = np.mean(predictions_ai)
        else:
            adm_simulees_base = pred_j1 * (1 + impact_admissions/100)
            
        facteur_reduction_capacite = 1 + (impact_staff / 100)
        if facteur_reduction_capacite < 0.2: facteur_reduction_capacite = 0.2
        
        capacite_operationnelle = int(capacite * facteur_reduction_capacite)
        
        # C. Projection sur la durée
        jours = np.arange(duree_crise)
        admissions_proj = [adm_simulees_base * (1 + np.random.normal(0, 0.05)) for _ in jours]
        

        beds_proj = []
        current_beds = last_day_beds
        
        for adm_du_jour in admissions_proj:
            sorties_estimees = current_beds * 0.15
            if impact_staff < 0:
                sorties_estimees *= (1 + impact_staff/100) 
            
            current_beds = current_beds - sorties_estimees + adm_du_jour
            beds_proj.append(current_beds)
        
        # --- AFFICHAGE DES RÉSULTATS ---
        
        st.subheader(f"Analyse : Scénario {scenario} ({duree_crise} jours)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Flux Patients / Jour", f"{int(adm_simulees_base)}",
                     delta=f"{impact_admissions:+d}% vol.")
        
        with col2:
            st.metric("Capacité Staffée", f"{capacite_operationnelle} lits",
                     delta=f"{capacite_operationnelle - capacite} lits",
                     delta_color="inverse",
                     help="Nombre de lits réellement gérables avec le staff disponible")
        
        with col3:
            # Le taux d'occupation est calculé sur la capacité RÉDUITE
            taux_reel = (beds_proj[-1] / capacite_operationnelle) * 100
            st.metric("Tension Réelle", f"{taux_reel:.0f}%",
                     delta="CRITIQUE" if taux_reel > 100 else "Élevée")

        st.markdown("---")
        
        # Graphique Projection
        st.subheader("Projection de la Saturation (Impact Staff inclus)")
        
        fig_sim, ax_sim = plt.subplots(figsize=(12, 5))
        
        # Courbe des lits occupés
        ax_sim.plot(jours, beds_proj, marker='o', linewidth=3, 
                    label='Patients Hospitalisés', color='#E74C3C')
        
        # Ligne de capacité PHYSIQUE
        ax_sim.axhline(capacite, color='gray', linestyle='--', alpha=0.5, 
                       label=f'Capacité Murale ({capacite})')
        
        # Ligne de capacité OPÉRATIONNELLE 
        ax_sim.axhline(capacite_operationnelle, color='#F1C40F', linestyle='-', linewidth=2, 
                       label=f'Capacité Staffée ({capacite_operationnelle})')
        
        # Zone de DANGER
        ax_sim.fill_between(jours, beds_proj, capacite_operationnelle, 
                            where=(np.array(beds_proj) > capacite_operationnelle),
                            color='#E74C3C', alpha=0.3, label='Zone de Rupture RH')
        
        ax_sim.set_xlabel('Jours de crise', fontsize=11)
        ax_sim.set_ylabel('Lits', fontsize=11)
        ax_sim.legend()
        ax_sim.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_sim)
        
        # RECOMMANDATIONS INTELLIGENTES
        st.subheader("Recommandations Stratégiques")
        
        # Calcul du nombre de jours en rupture
        jours_rupture = sum(1 for b in beds_proj if b > capacite_operationnelle)
        
        if jours_rupture > 0:
            if impact_staff < -20:
                st.error(f"""
                ALERTE MAJEURE : L'hôpital s'effondre par manque de personnel
                Avec **{impact_staff}% de staff**, votre capacité réelle tombe à **{capacite_operationnelle} lits**.
                
                **Actions Immédiates :**
                1.  **Fermeture de {capacite - capacite_operationnelle} lits** (sécurité patient).
                2. Appel à la Réserve Sanitaire Nationale.
                3. Déroutement des ambulances vers autres hôpitaux (Délestage).
                4. Primes de solidarité pour rappel staff.
                """)
            elif jours_rupture > 5:
                st.error("""
                PLAN BLANC OBLIGATOIRE
                La saturation va durer plus de 5 jours. Les équipes ne tiendront pas.
                - Déprogrammer tout le non-urgent.
                - Rappel du personnel sur repos.
                """)
            else:
                st.warning(f"""
                TENSION TEMPORAIRE ({jours_rupture} jours)
                Situation critique mais courte.
                - Heures supplémentaires.
                - Sorties anticipées.
                """)
        else:
            st.success("Le système tient le choc malgré la crise.")


        # =========================================================
        # MODE : SITUATION DE RÉFÉRENCE 
        # =========================================================
        
                
        st.subheader(f"Situation de Référence (au {last_date_str})")
        
        taux_actuel = (last_day_beds / capacite) * 100
        
        # Tendance sur les 7 derniers jours du fichier
        mean_7j = df_adm.tail(7)['nb_admissions'].mean()
        mean_prev_7j = df_adm.tail(14).head(7)['nb_admissions'].mean()
        evolution = ((mean_7j - mean_prev_7j) / mean_prev_7j) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Occupation Lits", f"{taux_actuel:.1f}%", 
                     delta=f"{last_day_beds} / {capacite}")
        
        with col2:
            st.metric("Flux Admissions", f"{int(mean_7j)} /jour", 
                     delta=f"{evolution:+.1f}% (Tendance 7j)")
            
        with col3:
            lits_restants = capacite - last_day_beds
            if mean_7j > 0:
                jours_restants = lits_restants / (mean_7j * 0.1) 
            else:
                jours_restants = 99
            
            valeur_affichée = int(jours_restants) if jours_restants < 30 else "> 30"
            st.metric("Marge de sécurité", f"{valeur_affichée} jours",
                     delta="Avant saturation théorique",
                     delta_color="normal")

        st.markdown("---")

        # DIAGNOSTIC AUTOMATIQUE 
        st.subheader("Diagnostic de la Situation")
        
        if taux_actuel > 95:
            st.error(f"""
            CRITIQUE : Hôpital Saturé à {taux_actuel:.1f}%
            La situation actuelle est intenable. 
            Il n'y a presque plus de marge de manœuvre pour absorber un imprévu.
            """)
        elif taux_actuel > 85:
            st.warning(f"""
            TENSION ÉLEVÉE : Occupation à {taux_actuel:.1f}%
            L'hôpital fonctionne à plein régime. 
            Le moindre événement (grippe, accident) fera basculer vers la saturation.
            """)
        elif taux_actuel > 60:
            st.info(f"""
            ACTIVITÉ NORMALE : Occupation à {taux_actuel:.1f}%
            Le flux est soutenu mais géré. Les équipes sont en place.
            """)
        else:
            st.success(f"""
            SITUATION CALME : Occupation à {taux_actuel:.1f}%
            Capacité d'accueil très large disponible.
            """)

# =============================================================================
# TAB 5: GESTION DES STOCKS & LOGISTIQUE
# =============================================================================

with tab5:
    st.header("Pilotage Logistique & Stocks")
    
    last_row = df_stocks.iloc[-1]
    
    items_stock = {
        "Masques": "masques",
        "Blouses": "blouses",
        "Respirateurs": "respirateurs",
        "Tests PCR": "tests",
        "Gel Hydro (L)": "gel"
    }
    
    st.subheader("État des lieux instantané")
    cols = st.columns(len(items_stock))
    
    for i, (label, col_name) in enumerate(items_stock.items()):
        stock_actuel = last_row[col_name]
        seuil = last_row[f"seuil_{col_name}"]
        
        if stock_actuel < seuil:
            etat = "🔴 CRITIQUE"
            delta_color = "inverse"
        elif stock_actuel < seuil * 1.2:
            etat = "🟠 BAS"
            delta_color = "normal"
        else:
            etat = "🟢 OK"
            delta_color = "normal"
            
        with cols[i]:
            st.metric(label, f"{stock_actuel:,}", delta=etat, delta_color="off")

    st.markdown("---")

    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.subheader("Focus Matériel")
        choix_item_label = st.selectbox("Sélectionner un stock à analyser :", list(items_stock.keys()))
        item_col = items_stock[choix_item_label]
        
        current_val = last_row[item_col]
        threshold_val = last_row[f"seuil_{item_col}"]
        max_val_histo = df_stocks[item_col].max()
        

        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = current_val,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Niveau actuel", 'font': {'size': 20}},
            delta = {'reference': threshold_val, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge = {
                'axis': {'range': [0, max_val_histo * 1.1], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#3498DB"}, # Bleu
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "#333",
                'steps': [
                    {'range': [0, threshold_val], 'color': 'rgba(231, 76, 60, 0.6)'}, # Rouge (Zone Critique)
                    {'range': [threshold_val, threshold_val*1.5], 'color': 'rgba(241, 196, 15, 0.4)'} # Jaune (Zone Tampon)
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold_val
                }
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # --- B. ESTIMATION AUTONOMIE  ---
        st.markdown("##### Estimation d'Autonomie")
        
        df_last_30 = df_stocks.tail(30)
        diffs = df_last_30[item_col].diff()
        conso_moyenne = abs(diffs[diffs < 0].mean())
        
        if conso_moyenne > 0:
            jours_restants = current_val / conso_moyenne
            
            if jours_restants < 5:
                st.error(f"RUPTURE IMMINENTE\n\nStock épuisé dans **{jours_restants:.1f} jours au rythme actuel.")
            elif jours_restants < 15:
                st.warning(f"ATTENTION\n\nAutonomie estimée : **{int(jours_restants)} jours.")
            else:
                st.success(f"CONFORTABLE\n\nAutonomie estimée : **{int(jours_restants)} jours.")
                
            st.caption(f"Consommation moyenne : {int(conso_moyenne)} unités/jour")
        else:
            st.info("Pas de consommation détectée récemment.")

    with col_right:
        st.subheader("Historique & Livraisons")
        
        # --- C. GRAPHIQUE INTELLIGENT ---
        df_chart = df_stocks.tail(180)
        
        fig_line = go.Figure()
        
        fig_line.add_trace(go.Scatter(
            x=df_chart['date'], 
            y=df_chart[item_col],
            mode='lines', 
            name='Stock disponible',
            fill='tozeroy',
            line=dict(color='#3498DB', width=2)
        ))
        
        # 2. Ligne de seuil critique
        fig_line.add_trace(go.Scatter(
            x=df_chart['date'], 
            y=df_chart[f"seuil_{item_col}"],
            mode='lines', 
            name='Seuil Sécurité',
            line=dict(color='#E74C3C', width=2, dash='dash')
        ))
        
        # 3. Marqueurs de Réapprovisionnement
        cmd_col = f"cmd_{item_col}"
        if cmd_col in df_stocks.columns:
            commandes = df_chart[df_chart[cmd_col] == 1]
            
            if not commandes.empty:
                fig_line.add_trace(go.Scatter(
                    x=commandes['date'], 
                    y=commandes[item_col],
                    mode='markers', 
                    name='Livraison Reçue',
                    marker=dict(color='#2ECC71', size=12, symbol='star', line=dict(width=2, color='white'))
                ))

        fig_line.update_layout(
            title=f"Évolution {choix_item_label} (6 derniers mois)",
            xaxis_title="Date",
            yaxis_title="Quantité en stock",
            template="plotly_dark",
            hovermode="x unified",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_line, use_container_width=True)
        
        with st.expander("Voir l'historique complet (2014-2024)"):
            fig_full = go.Figure()
            fig_full.add_trace(go.Scatter(x=df_stocks['date'], y=df_stocks[item_col], line=dict(color='#3498DB')))
            fig_full.update_layout(title="Historique Complet", template="plotly_dark", height=300)
            st.plotly_chart(fig_full, use_container_width=True)

# =============================================================================
# TAB 6: ANALYSES STATISTIQUES
# =============================================================================

with tab6:
    st.header("Analyses Statistiques Avancées")
    
    st.markdown("""
    Méthodes statistiques pour détecter patterns et corrélations.
    """)
    
    st.markdown("---")
    
    # Graph 7
    st.subheader("Graph 7 : Heatmap Jour × Mois")
    if os.path.exists("graph7_heatmap_admissions.png"):
        img = Image.open("graph7_heatmap_admissions.png")
        st.image(img, use_container_width=True)
        with st.expander("Description"):
            st.markdown("""
            Objectif : Identifier patterns hebdomadaires et saisonniers.
                        
            Résultats observés : On observe clairement que les weekends (samedi/dimanche) 
            sont en jaune (moins d'admissions ~200), tandis que janvier-février (mois 1-2) et 
            décembre (mois 12) sont rouge foncé (pics à 300-350 admissions). Les vendredis d'hiver 
            atteignent 344 admissions.
            
            """)
    else:
        st.info("graph7_heatmap_admissions.png non trouvé")
    
    st.markdown("---")
    
    # Graph 10
    st.subheader("Graph 10 : Corrélation Gravité × Durée Séjour")
    if os.path.exists("graph10_correlation.png"):
        img = Image.open("graph10_correlation.png")
        st.image(img, use_container_width=True)
        with st.expander("Description"):
            st.markdown("""
            Objectif : Quantifier relation gravité/durée hospitalisation.
            
            Chaque point = 1 patient  
            Ligne rouge = régression linéaire
            
            Résultats observés : Le coefficient r indique une corrélation positive 
            modérée. Les patients de gravité 1 restent en moyenne 5-10 jours, tandis que 
            ceux de gravité 5 peuvent rester 20-30 jours. La dispersion augmente avec la 
            gravité (plus de variabilité pour les cas graves).
            
            Utilité : Prédire durée selon gravité pour anticiper occupation lits.
            """)
    else:
        st.info("graph10_correlation.png non trouvé")

# Footer
st.markdown("---")
st.markdown("""
""", unsafe_allow_html=True)