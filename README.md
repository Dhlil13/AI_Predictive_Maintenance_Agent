 🤖 AI Predictive Maintenance Agent

## 📌 Présentation

Ce projet propose une implémentation d'un **assistant intelligent de maintenance prédictive** combinant :

- un système **RAG (Retrieval-Augmented Generation)** pour répondre à des questions techniques à partir d'une documentation industrielle ;
- un modèle de **Machine Learning** permettant d'estimer le risque de défaillance d'une machine ;
- un **Dashboard Streamlit** offrant une interface utilisateur interactive pour l'exploration des données, la prédiction et l'utilisation du chatbot.

Le projet a été développé dans un objectif pédagogique afin d'illustrer les principales briques de l'IA générative appliquées au domaine industriel.

---

# 🏗️ Architecture du projet

```
┌─────────────────────┐
│ Maintenance Manual │
│ (PDF) │
└──────────┬──────────┘
│
Document Chunking
│
Vector Embeddings
│
FAISS Index
│
Similarity Search
│
▼
TinyLlama (GGUF)
│
▼
Maintenance Assistant
│
▼
Streamlit Dashboard
▲
│
AI4I Predictive Maintenance Dataset
│
Machine Learning Model
│
▼
Failure Probability Prediction
```

---

# 🚀 Fonctionnalités

## 🤖 Assistant IA (RAG)

- Recherche sémantique dans un manuel de maintenance
- Génération de réponses contextualisées
- Modèle LLM exécuté entièrement en local
- Aucune dépendance à une API externe

---

## 🔮 Maintenance prédictive

Le projet intègre un modèle de Machine Learning entraîné sur le jeu de données **AI4I 2020 Predictive Maintenance Dataset** permettant de :

- prédire le risque de défaillance ;
- estimer la probabilité de panne ;
- illustrer un cas d'usage industriel de maintenance prédictive.

---

## 📊 Dashboard interactif

Le tableau de bord permet :

- d'interroger le chatbot RAG ;
- d'explorer le dataset ;
- de visualiser les statistiques descriptives ;
- d'effectuer des prédictions de défaillance à partir de paramètres saisis par l'utilisateur.

---

# 📂 Structure du projet

```
AI_Predictive_Maintenance_Agent/

│
├── backend/
│ ├── ingest.py
│ ├── rag_agent.py
│ ├── main.py
│ ├── models/
│ └── data/
│
├── datasets/
│ └── ai4i2020.csv
│
├── ml/
│ ├── train_model.py
│ ├── predict.py
│ ├── model.pkl
│ └── scaler.pkl
│
├── vectorstore/
│
├── dashboard.py
│
├── requirements.txt
│
└── README.md
```

---

# 🧠 Technologies utilisées

## IA Générative

- TinyLlama 1.1B Chat (GGUF)
- llama-cpp-python
- FAISS
- LangChain Community

## Machine Learning

- Scikit-Learn
- Pandas
- NumPy
- Joblib

## Interface

- Streamlit
- Plotly

 API

- FastAPI
- Uvicorn

---

📚 Jeu de données

Le modèle de Machine Learning est entraîné sur :

**AI4I 2020 Predictive Maintenance Dataset**

Variables principales :

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

Variable cible :

- Machine Failure


⚙️ Installation

Créer un environnement Python :

bash
conda create -n rag_pdm python=3.10
conda activate rag_pdm

Installer les dépendances :
bash
pip install -r requirements.txt


▶️ Création de la base vectorielle

Placer les documents PDF dans :
backend/data/

Puis lancer :
python backend/ingest.py

▶️ Lancer le backend

Depuis le dossier backend :

bash
uvicorn main:app --reload

Documentation FastAPI :
http://127.0.0.1:8000/docs


▶️ Lancer le Dashboard

Depuis la racine du projet :

```bash
streamlit run dashboard.py
```

---

💬 Exemples de questions

Le chatbot peut répondre à des questions telles que :

- What causes bearing overheating?
- How can excessive vibration be reduced?
- What maintenance should be performed regularly?
- What are the symptoms of bearing failure?
- How should bearings be lubricated?



🎯 Objectifs pédagogiques

Ce projet illustre :

- la construction d'un système RAG ;
- l'utilisation d'une base vectorielle FAISS ;
- l'intégration d'un modèle de langage local ;
- le développement d'un assistant IA spécialisé ;
- l'application du Machine Learning à la maintenance prédictive.

Il constitue un support de travaux pratiques pour un cours d'**IA Générative appliquée à l'industrie**.


🔄 Perspectives

Évolutions possibles :

- intégration complète ML + RAG ;
- estimation du Remaining Useful Life (RUL) ;
- prise en charge de plusieurs manuels techniques ;
- connexion à une base de données temps réel ;
- intégration de données IoT industrielles.
