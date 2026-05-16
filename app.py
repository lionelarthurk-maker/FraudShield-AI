# ============================================================
# DÉTECTION DE FRAUDE BANCAIRE — FraudShield AI
# Projet TPE IA — L3 ISGE-BF
# Prof : Dr Rodrique KAFANDO
# ============================================================

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(0,212,255,0.2);
    }
    .hero h1 { font-size: 2.8em; font-weight: 700; margin: 0; color: #00d4ff; }
    .hero p  { font-size: 1.1em; opacity: 0.8; margin-top: 10px; }

    .card {
        background: white;
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    .carte-fraude {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        color: white;
        font-size: 1.5em;
        font-weight: 700;
        box-shadow: 0 8px 32px rgba(255,65,108,0.3);
    }
    .carte-normale {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        color: white;
        font-size: 1.5em;
        font-weight: 700;
        box-shadow: 0 8px 32px rgba(56,239,125,0.3);
    }
    .carte-info {
        background-color: #f8f9fa;
        border-left: 4px solid #00d4ff;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        color: #333;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0f3460, #00d4ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 15px 30px !important;
        font-size: 1.1em !important;
        font-weight: 600 !important;
        width: 100% !important;
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: #888;
        font-size: 0.85em;
        border-top: 1px solid #eee;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# CHARGEMENT MODÈLE + DONNÉES
# ─────────────────────────────────────────
@st.cache_resource
def charger_modele():
    model  = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

@st.cache_data
def generer_données():
    np.random.seed(42)
    n_samples = 5000
    n_normal  = int(n_samples * 0.95)
    n_fraud   = n_samples - n_normal

    normal = pd.DataFrame({
        'montant':       np.random.exponential(scale=80,  size=n_normal),
        'heure':         np.random.randint(0, 24,          size=n_normal),
        'distance_km':   np.random.exponential(scale=5,   size=n_normal),
        'nb_trans_jour': np.random.poisson(lam=3,          size=n_normal),
        'score_risque':  np.random.beta(2, 8,              size=n_normal),
        'Fraude': 0
    })
    fraud = pd.DataFrame({
        'montant':       np.random.exponential(scale=400,          size=n_fraud),
        'heure':         np.random.choice([0,1,2,3,22,23],         size=n_fraud),
        'distance_km':   np.random.exponential(scale=200,          size=n_fraud),
        'nb_trans_jour': np.random.poisson(lam=10,                 size=n_fraud),
        'score_risque':  np.random.beta(8, 2,                      size=n_fraud),
        'Fraude': 1
    })
    df = pd.concat([normal, fraud], ignore_index=True).sample(frac=1, random_state=42)
    return df.reset_index(drop=True)

try:
    model, scaler = charger_modele()
    modele_ok = True
except FileNotFoundError:
    modele_ok = False

df = generer_données()


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ FraudShield AI")
    st.markdown("---")
    st.markdown("""
    <div class='carte-info'>
    Système de détection de fraude bancaire basé sur le <b>Machine Learning</b>.<br><br>
    Modèle : <b>Random Forest</b><br>
    Bibliothèque : <b>scikit-learn</b><br>
    Sauvegarde : <b>joblib</b>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div class='carte-info'>
    <b>Cours :</b> AI Fondamental<br>
    <b>Niveau :</b> Licence 3<br>
    <b>École :</b> ISGE-BF<br>
    <b>Prof :</b> Dr Rodrique KAFANDO
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div class='carte-info'>
    🟢 <b>0 - 30%</b> → Faible risque<br>
    🟡 <b>30 - 60%</b> → Risque modéré<br>
    🔴 <b>60 - 100%</b> → Fraude probable
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# HERO
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🛡️ FraudShield AI</h1>
    <p>Système intelligent de détection de fraude bancaire · Propulsé par Machine Learning</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# ONGLETS
# ─────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Analyse & Prédiction", "📊 Dashboard & Statistiques"])


# ══════════════════════════════════════════
# ONGLET 1 — ANALYSE
# ══════════════════════════════════════════
with tab1:

    if not modele_ok:
        st.error("❌ Modèle introuvable. Exécutez d'abord le notebook.")
        st.stop()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📋 Informations de la transaction")
    st.markdown("Renseignez les valeurs pour obtenir une analyse instantanée.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**💳 Détails du paiement**")
        montant = st.number_input("💰 Montant (FCFA)", min_value=0,
                                   max_value=10_000_000, value=25_000, step=1000)
        heure   = st.slider("🕐 Heure de la transaction", 0, 23, 14)

    with col2:
        st.markdown("**📍 Localisation & Activité**")
        distance_km   = st.number_input("📍 Distance domicile (km)", 0.0, 5000.0, 2.0, 0.5)
        nb_trans_jour = st.number_input("🔢 Transactions aujourd'hui", 0, 50, 2)

    with col3:
        st.markdown("**⚠️ Profil du compte**")
        score_risque = st.slider("⚠️ Score de risque", 0.0, 1.0, 0.1, 0.01)
        st.markdown(f"""
        <div class='carte-info'>
        💰 <b>{montant:,} FCFA</b><br>
        🕐 <b>{heure}h00</b><br>
        📍 <b>{distance_km} km</b><br>
        🔢 <b>{nb_trans_jour} transactions</b><br>
        ⚠️ Score : <b>{score_risque}</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔍 Lancer l'analyse IA"):
        with st.spinner("🤖 Analyse en cours..."):
            time.sleep(1.2)

        donnees      = np.array([[montant, heure, distance_km, nb_trans_jour, score_risque]])
        donnees_norm = scaler.transform(donnees)
        prediction   = model.predict(donnees_norm)[0]
        probabilite  = model.predict_proba(donnees_norm)[0]
        prob_fraude  = probabilite[1] * 100
        prob_normale = probabilite[0] * 100

        st.markdown("---")

        if prediction == 1:
            st.markdown(f"""
            <div class='carte-fraude'>
                🚨 FRAUDE DÉTECTÉE<br>
                <span style='font-size:0.6em'>Probabilité : {prob_fraude:.1f}%</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='carte-normale'>
                ✅ TRANSACTION NORMALE<br>
                <span style='font-size:0.6em'>Probabilité de fraude : {prob_fraude:.1f}%</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("🟢 Prob. Normale", f"{prob_normale:.1f}%")
        m2.metric("🔴 Prob. Fraude",  f"{prob_fraude:.1f}%")
        m3.metric("🎯 Décision", "FRAUDE" if prediction == 1 else "NORMALE")

        st.markdown("#### 📊 Niveau de risque")
        st.progress(int(prob_fraude))

        st.markdown("#### 🔎 Facteurs de risque")
        facteurs = {
            "💰 Montant très élevé (> 200 000 FCFA)"   : montant > 200_000,
            "🌙 Heure suspecte (22h - 4h)"              : heure <= 4 or heure >= 22,
            "📍 Transaction loin du domicile (> 100km)" : distance_km > 100,
            "🔢 Trop de transactions ce jour (> 7)"     : nb_trans_jour > 7,
            "⚠️ Score de risque élevé (> 0.6)"         : score_risque > 0.6,
        }
        col_a, col_b = st.columns(2)
        with col_a:
            for k, v in facteurs.items():
                if v: st.error(k)
        with col_b:
            for k, v in facteurs.items():
                if not v: st.success(k)

        st.markdown("#### 💡 Recommandation")
        if prediction == 1:
            st.error("🔒 Transaction **bloquée**. Le service sécurité a été notifié.")
        else:
            st.success("✅ Transaction **autorisée**. Aucune anomalie détectée.")


# ══════════════════════════════════════════
# ONGLET 2 — DASHBOARD
# ══════════════════════════════════════════
with tab2:

    st.markdown("### 📊 Dashboard — Analyse du Dataset")

    # ── KPIs ──
    total     = len(df)
    fraudes   = int(df['Fraude'].sum())
    normales  = total - fraudes

    k1, k2, k3 = st.columns(3)
    k1.metric("📦 Total transactions", f"{total:,}")
    k2.metric("✅ Transactions normales", f"{normales:,}", f"{normales/total*100:.1f}%")
    k3.metric("🚨 Fraudes détectées",    f"{fraudes:,}",  f"{fraudes/total*100:.1f}%")

    st.markdown("---")

    # ── Graphique 1 & 2 ──
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("#### Distribution Fraude / Normale")
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        colors = ['#2ecc71', '#e74c3c']
        df['Fraude'].value_counts().plot(
            kind='pie', ax=ax1,
            labels=['Normale', 'Fraude'],
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12}
        )
        ax1.set_title("Répartition des transactions", fontweight='bold', pad=15)
        ax1.set_ylabel("")
        st.pyplot(fig1)

    with col_g2:
        st.markdown("#### Importance des Features")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        importances = pd.Series(
            model.feature_importances_,
            index=['montant','heure','distance_km','nb_trans_jour','score_risque']
        ).sort_values(ascending=True)
        importances.plot(kind='barh', ax=ax2, color='#0f3460', edgecolor='white')
        ax2.set_title("Importance des paramètres", fontweight='bold', pad=15)
        ax2.set_xlabel("Score d'importance")
        st.pyplot(fig2)

    st.markdown("---")

    # ── Graphique 3 & 4 ──
    col_g3, col_g4 = st.columns(2)

    with col_g3:
        st.markdown("#### Distribution des montants")
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        df[df.Fraude==0]['montant'].hist(ax=ax3, bins=50, alpha=0.7,
                                         color='#2ecc71', label='Normale')
        df[df.Fraude==1]['montant'].hist(ax=ax3, bins=50, alpha=0.7,
                                         color='#e74c3c', label='Fraude')
        ax3.set_title("Montants : Normale vs Fraude", fontweight='bold', pad=15)
        ax3.set_xlabel("Montant (FCFA)")
        ax3.set_ylabel("Fréquence")
        ax3.legend()
        st.pyplot(fig3)

    with col_g4:
        st.markdown("#### Transactions par heure")
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        df[df.Fraude==0]['heure'].value_counts().sort_index().plot(
            ax=ax4, color='#2ecc71', label='Normale', linewidth=2)
        df[df.Fraude==1]['heure'].value_counts().sort_index().plot(
            ax=ax4, color='#e74c3c', label='Fraude', linewidth=2)
        ax4.set_title("Transactions par heure", fontweight='bold', pad=15)
        ax4.set_xlabel("Heure")
        ax4.set_ylabel("Nombre de transactions")
        ax4.legend()
        st.pyplot(fig4)

    st.markdown("---")

    # ── Heatmap ──
    st.markdown("#### Heatmap des Corrélations")
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.corr(), annot=True, fmt='.2f',
                cmap='Blues', linewidths=0.5, ax=ax5)
    ax5.set_title("Corrélations entre paramètres", fontweight='bold', pad=15)
    st.pyplot(fig5)

    st.markdown("---")

    # ── Statistiques ──
    st.markdown("#### Statistiques Descriptives")
    st.dataframe(df.describe().round(2), use_container_width=True)


# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div class="footer">
    FraudShield AI v1.0 · Projet TPE IA L3 ISGE-BF · Dr Rodrique KAFANDO · 2025
</div>
""", unsafe_allow_html=True)