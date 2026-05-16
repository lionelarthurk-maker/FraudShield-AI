# 🛡️ FraudShield AI — Détection de Fraude Bancaire

## 📌 Description du projet

Ce projet consiste à développer une application web intelligente permettant de **détecter automatiquement les transactions bancaires frauduleuses** à partir de caractéristiques de transaction.

Le modèle d'intelligence artificielle a été entraîné avec Python et Scikit-learn, puis déployé dans une application interactive avec Streamlit.

---

## 🎯 Objectifs

- Comprendre le cycle complet d'un projet IA
- Entraîner un modèle de Machine Learning
- Sauvegarder le modèle avec Joblib
- Construire une interface web avec Streamlit
- Déployer publiquement une application IA
- Réfléchir au déploiement Edge AI en contexte de faible connectivité

---

## 🛠️ Technologies utilisées

- **Python**
- **Scikit-learn** — algorithme Random Forest
- **Pandas / NumPy** — manipulation des données
- **Matplotlib / Seaborn** — visualisation
- **Joblib** — sauvegarde du modèle
- **Streamlit** — interface web interactive

---

## 📦 Données utilisées

- **Source** : Données simulées réalistes (5 000 transactions)
- **Répartition** : 95% transactions normales / 5% fraudes
- **Caractéristiques** :

| Variable | Description |
|---|---|
| `montant` | Montant de la transaction en FCFA |
| `heure` | Heure à laquelle la transaction a eu lieu |
| `distance_km` | Distance entre le domicile et le lieu de transaction |
| `nb_trans_jour` | Nombre de transactions effectuées dans la journée |
| `score_risque` | Score de risque du compte (0 = faible, 1 = élevé) |

---

## ⚙️ Fonctionnement du projet

1. **Génération des données** — création d'un dataset simulé réaliste
2. **Prétraitement** — normalisation avec StandardScaler
3. **Entraînement** — modèle Random Forest (100 arbres)
4. **Évaluation** — matrice de confusion, rapport de classification
5. **Sauvegarde** — `joblib.dump(model, 'model.pkl')`
6. **Déploiement** — application Streamlit accessible en ligne

---

## 🌐 Edge AI

Notre modèle peut fonctionner **hors connexion** directement sur un terminal bancaire, sans avoir besoin d'internet. Ceci est particulièrement utile dans les zones rurales au **Burkina Faso** où la connectivité est limitée.

---

## 🚀 Lancer l'application

```bash
pip install -r requirements.txt
streamlit run app.py# FraudShield-AI
Détection de  Fraude bancaire - Projet TPE IA L3 ISGE-BF
