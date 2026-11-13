# 🤖 RAG Multimodal & Agentique : De l'Extraction à l'Agent ReAct avec LangGraph

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph%20ReAct-orange.svg)](https://github.com/langchain-ai/langgraph)
[![ChromaDB](https://img.shields.io/badge/VectorStore-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![OpenCLIP](https://img.shields.io/badge/Embeddings-CLIP%20ViT--B--32-green.svg)](https://github.com/mlfoundations/open_clip)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

Projet complet de conception et de développement d'une application **RAG (Retrieval-Augmented Generation) Multimodale et Agentique**, capable d'extraire, d'indexer et d'interroger de manière autonome et raisonnée des contenus textuels et des figures visuelles issus de l'article scientifique de référence : ***"Attention Is All You Need"*** (Vaswani et al., 2017).


---

## 📐 Architecture du Système (MIND & BODY ReAct)

L'application découple le raisonnement de l'action selon le patron de conception **ReAct** (*Reasoning $\to$ Action $\to$ Observation*) orchestré par un graphe d'états **LangGraph** :

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AGENT IA (LANGGRAPH)                           │
│                                                                             │
│               ┌──────────────────────────────────────────────┐              │
│               │             CERVEAU (MIND - LLM)             │              │
│               │  - Raisonnement (Thought / Reasoning)         │              │
│               │  - Décomposition de requêtes complexes       │              │
│               │  - Décision d'appel d'outil (Action)         │              │
│               │  - Synthèse contextuelle multimodale         │              │
│               └──────────────────────┬───────────────────────┘              │
│                                      │                                      │
│                         Action       │    ▲ Observation                     │
│                       (Tool Call)    │    │ (Sources & Images)              │
│                                      ▼    │                                 │
│               ┌──────────────────────────────────────────────┐              │
│               │         CORPS (BODY - RETRIEVER TOOL)        │              │
│               │  - Filtrage (texte, image, tous)             │              │
│               │  - Normalisation & similarité cosinus        │              │
│               └──────────────────────┬───────────────────────┘              │
└──────────────────────────────────────┼──────────────────────────────────────┘
                                       │
                      Interrogation    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BASE VECTORIELLE (CHROMADB)                          │
│                                                                             │
│        Vecteurs Textes (512d)      │        Vecteurs Images (512d)          │
│   ┌─────────────────────────────┐  │   ┌─────────────────────────────┐      │
│   │ Fragments "Attention is..." │  │   │ Fig 1: Architecture Diagram │      │
│   │ Métadonnées : Page, SHA256  │  │   │ Fig 2: Multi-Head Attention │      │
│   └─────────────────────────────┘  │   └─────────────────────────────┘      │
│                                    │                                        │
│           Modèle d'Embedding Partagé : OpenCLIP ViT-B-32                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Les 7 Étapes Techniques Implémentées

1. **Extraction Multimodale (Textes & Images) :**
   - Utilisation de `PyMuPDF` (`fitz`) pour extraire le texte intégral ordonné page par page.
   - Extraction des figures visuelles vectorielles et matricielles, déduplication par `xref` et sauvegarde dans `resultats_pdf/images/`.
2. **Vectorisation dans un Espace Latent Conjoint (CLIP) :**
   - Modèle `ViT-B-32` (`laion2b_s34b_b79k`) projetant conjointement textes et images dans un espace partagé à $512$ dimensions.
   - Normalisation $L_2$ des vecteurs pour que le produit scalaire équivaille rigoureusement à la similarité cosinus.
   - Découpage en fragments adaptés à la fenêtre BPE de CLIP (60 tokens, chevauchement 10).
3. **Indexation et Stockage dans ChromaDB :**
   - Collection persistante HNSW configurée avec la métrique cosinus.
   - Hachage cryptographique **SHA-256** du document pour la gestion des versions et l'idempotence des réindexations.
   - Indexation de 172 fragments de texte et 3 figures d'images associées avec métadonnées (`page`, `fragment`, `document_id`, `source`).
4. **Moteur de Recherche Multimodale (Retrieval) :**
   - Fonction universelle `rechercher(texte, image, type_resultat, k)`.
   - Calcul de similarité cosinus : $\text{sim} = 1 - \text{distance}$.
   - Recherche croisée : Texte $\to$ Textes/Images et Image $\to$ Textes explicatifs.
5. **Transformation en Outil Agentique (Tool) :**
   - Décorateur `@tool` (`retriever_tool`) avec annotations strictes de type (`Literal["tous", "texte", "image"]`) et documentation pour le LLM.
6. **Agent ReAct & LangGraph :**
   - Construction du graphe d'états réactif : État `AgentState`, nœud `mind`, nœud `body` (`ToolNode`), et arête conditionnelle `tools_condition`.
   - Boucle de rétroaction continue : *Thought $\to$ Action $\to$ Observation $\to$ Synthesis*.
7. **Tests, Validation & Interface Interactive :**
   - Validation sur des questions complexes (définitions d'attention, figures, raisonnements comparatifs, détection d'absence d'information).
   - Application web interactive sous **Streamlit** avec chat documentaire et recherche par image.

---

## 📦 Installation et Prérequis

Cloner le dépôt :
```bash
git clone https://github.com/Walid-Idir/RAG-Multimodale-et-Agentique.git
cd RAG-Multimodale-et-Agentique
```

Installer les dépendances :
```bash
pip install -r requirements.txt
```

---

## 💻 Utilisation

### 1. Exécuter le Notebook Jupyter
Pour explorer le pipeline pas-à-pas avec toutes les cellules exécutées et documentées :
```bash
jupyter notebook rag_multimodal_agentique_Walid_Idir.ipynb
```

### 2. Lancer l'Application Web Streamlit
Pour interagir avec l'agent via une interface graphique moderne :
```bash
streamlit run app_streamlit.py
```
L'interface s'ouvrira à l'adresse `http://localhost:8501`.

**Fonctionnalités de l'interface :**
- 💬 **Chat interactif :** posez des questions en langage naturel.
- 🧠 **Cycle ReAct visible :** observez les étapes de pensée, les appels d'outils et les observations en direct.
- 🖼️ **Recherche par image :** glissez-déposez un schéma (ex: `architecture_transformer.png`) pour interroger la base par l'image.
- 📑 **Citations et visualisations :** affichage des extraits avec leurs numéros de page et rendu immédiat des figures associées.

---

## 📂 Structure du Répertoire

```
├── rag_multimodal_agentique_Walid_Idir.ipynb  # Notebook principal signé Walid IDIR (complet et exécuté)
├── app_streamlit.py                  # Application web interactive Streamlit
├── architecture_transformer.png      # Figure 1 du Transformer extraite du PDF
├── requirements.txt                  # Dépendances Python
├── .gitignore                        # Fichiers ignorés par Git
├── pdfs/                             # Document PDF source
│   └── attention is all you need.pdf
├── resultats_pdf/                    # Données extraites
│   ├── textes/                       # Pages au format texte
│   └── images/                       # Figures extraites (PNG)
└── chroma_pdf/                       # Base vectorielle persistante ChromaDB
```

---

## 👨‍💻 Auteur
- **Walid IDIR**
- Projet réalisé dans le cadre du module **IA & RAG Multimodal**.

