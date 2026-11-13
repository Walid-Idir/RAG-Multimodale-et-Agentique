import os
import json
from pathlib import Path
import streamlit as st
from PIL import Image
import torch
import open_clip
import chromadb
from typing import Annotated, Sequence, TypedDict, Literal, Optional

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="RAG Multimodal & Agentique - Attention Is All You Need",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Répertoires
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_pdf"
IMAGE_DIR = BASE_DIR / "resultats_pdf" / "images"

# --- Mise en cache des ressources lourdes (CLIP & ChromaDB) ---
@st.cache_resource(show_spinner="Chargement du modèle CLIP (ViT-B-32)...")
def charger_ressources():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection("pdf_clip_vit_b32", metadata={"hnsw:space": "cosine"})
    return device, model, preprocess, tokenizer, collection

device, clip_model, clip_preprocess, clip_tokenizer, collection = charger_ressources()

# --- Fonctions d'encodage ---
@torch.inference_mode()
def encoder_texte(texte: str):
    tokens = clip_tokenizer([texte]).to(device)
    vec = clip_model.encode_text(tokens).float()
    vec = vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return vec.cpu().tolist()[0]

@torch.inference_mode()
def encoder_image(image_pil: Image.Image):
    tenseur = clip_preprocess(image_pil.convert("RGB")).unsqueeze(0).to(device)
    vec = clip_model.encode_image(tenseur).float()
    vec = vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return vec.cpu().tolist()[0]

# --- Fonction de recherche multimodale équilibrée ---
def rechercher(texte=None, image_pil=None, type_resultat="tous", k=4):
    """
    Recherche multimodale équilibrée :
    En mode 'tous', interroge à la fois les textes et les figures pour éviter que
    les scores texte-texte plus élevés n'éclipsent les diagrammes visuels.
    """
    if texte is not None:
        vecteur = encoder_texte(texte)
    elif image_pil is not None:
        vecteur = encoder_image(image_pil)
    else:
        raise ValueError("Fournir un texte ou une image.")

    total = collection.count()
    if total == 0:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

    if type_resultat == "texte":
        return collection.query(
            query_embeddings=[vecteur],
            n_results=min(k, total),
            where={"type": "texte"},
            include=["documents", "metadatas", "distances"]
        )
    elif type_resultat == "image":
        return collection.query(
            query_embeddings=[vecteur],
            n_results=min(k, total),
            where={"type": "image"},
            include=["documents", "metadatas", "distances"]
        )
    else:
        # Mode combiné 'tous' : récupération équilibrée des textes ET des figures
        k_txt = max(2, k)
        k_img = 2  # Récupère au moins les 2 meilleures figures correspondantes
        res_txt = collection.query(
            query_embeddings=[vecteur],
            n_results=min(k_txt, total),
            where={"type": "texte"},
            include=["documents", "metadatas", "distances"]
        )
        res_img = collection.query(
            query_embeddings=[vecteur],
            n_results=min(k_img, total),
            where={"type": "image"},
            include=["documents", "metadatas", "distances"]
        )

        docs = res_txt["documents"][0] + res_img["documents"][0]
        metas = res_txt["metadatas"][0] + res_img["metadatas"][0]
        dists = res_txt["distances"][0] + res_img["distances"][0]
        ids = res_txt["ids"][0] + res_img["ids"][0]

        return {"documents": [docs], "metadatas": [metas], "distances": [dists], "ids": [ids]}

# --- Fonction de génération / synthèse LLM ---
def generer_synthese_reponse(question: str, sources: list, api_key: Optional[str] = None) -> str:
    """Génère une réponse fluide et rigoureuse soit via OpenAI, soit via synthèse locale structurée."""
    cle = api_key or os.environ.get("OPENAI_API_KEY", "")

    if cle and cle.strip() and cle != "....":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=cle)

            contexte_sources = []
            for s in sources:
                contexte_sources.append(
                    f"[{s['ref']}] (Page {s['page']}, type: {s['type']}) : {s['contenu']}"
                )
            prompt_systeme = (
                "Tu es un assistant scientifique expert. Réponds en français de façon fluide, concise et rigoureuse "
                "en t'appuyant EXCLUSIVEMENT sur les sources documentaires fournies issues de l'article 'Attention Is All You Need'. "
                "Cite les numéros de page pour chaque affirmation (ex: [p. 3])."
            )
            prompt_utilisateur = (
                f"Question : {question}\n\n"
                f"Sources documentaires issues du PDF :\n" + "\n\n".join(contexte_sources)
            )

            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt_systeme},
                    {"role": "user", "content": prompt_utilisateur}
                ],
                temperature=0.2,
                max_tokens=600
            )
            return completion.choices[0].message.content
        except Exception as e:
            st.warning(f"Note : Échec de l'appel OpenAI ({e}). Utilisation de la synthèse documentaire locale.")

    # Synthèse documentaire structurée locale (sans clé API)
    lignes = [
        f"### 📋 Synthèse documentaire",
        f"D'après l'article scientifique ***Attention Is All You Need*** (Vaswani et al., 2017) :\n"
    ]

    # Extraction des passages textuels principaux
    passages_textuels = [s for s in sources if s['type'] == 'texte']
    if passages_textuels:
        top_s = passages_textuels[0]
        lignes.append(f"**Élément clé identifié (Page {top_s['page']}) :**")
        lignes.append(f"> *« {top_s['contenu']} »* [p. {top_s['page']}]\n")

    lignes.append("#### 📚 Sources et extraits pertinents retrouvés :")
    for s in sources:
        num_p = s['page'] if s['page'] != -1 else "inconnue"
        if s['type'] == 'texte':
            lignes.append(f"- **[{s['ref']}] Passage textuel (Page {num_p} | similarité cosinus : `{s['sim']:.3f}`) :**")
            lignes.append(f"  > {s['contenu']}\n")
        else:
            nom_f = Path(s['source']).name
            lignes.append(f"- **[{s['ref']}] Figure / Schéma (Page {num_p} | similarité cosinus : `{s['sim']:.3f}`) :** `{nom_f}`\n")

    return "\n".join(lignes)

# --- Barre Latérale (Configuration & Filtres) ---
with st.sidebar:
    st.title("⚙️ Paramètres")
    st.info(f"**Matériel :** `{device.upper()}`\n\n**Base ChromaDB :** `{collection.count()} vecteurs`")

    st.subheader("🔍 Filtres de Recherche")
    mode_recherche = st.selectbox(
        "Type de résultats souhaités :",
        options=["tous", "texte", "image"],
        index=0,
        format_func=lambda x: {"tous": "Tous (Textes & Images combinés)", "texte": "Textes uniquement", "image": "Images / Figures uniquement"}[x],
        help="En mode combiné, l'agent recherche à la fois les explications textuelles et les figures correspondantes."
    )

    top_k = st.slider("Nombre de sources (k) :", min_value=1, max_value=8, value=4)

    st.markdown("---")
    st.subheader("🖼️ Requête par Image")
    st.caption("Interroger le document avec une image ou un schéma :")
    uploaded_file = st.file_uploader("Téléverser une image :", type=["png", "jpg", "jpeg"])
    image_requete = None
    if uploaded_file is not None:
        image_requete = Image.open(uploaded_file)
        st.image(image_requete, caption="Image de requête", use_container_width=True)

    st.markdown("---")
    st.subheader("🔑 Clé API OpenAI (Optionnelle)")
    cle_api = st.text_input("Clé API OpenAI :", type="password", placeholder="sk-...")
    if cle_api:
        os.environ["OPENAI_API_KEY"] = cle_api
        st.success("Clé API configurée pour GPT-4o-mini !")

    if st.button("🗑️ Réinitialiser la conversation"):
        st.session_state.messages = []
        st.rerun()

# --- Zone Principale ---
st.title("🤖 RAG Multimodal & Agentique (ReAct)")
st.caption("Application RAG multimodale avec espace latent CLIP ViT-B-32, ChromaDB et cycle ReAct.")

# Historique des messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Bonjour ! Je suis votre agent documentaire multimodal pour l'article ***Attention Is All You Need***.\n\nPosez-moi une question sur le Transformer ou soumettez un schéma pour démarrer l'analyse.",
            "images": []
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "images" in msg and msg["images"]:
            st.markdown("#### 🖼️ Figures associées :")
            cols = st.columns(min(len(msg["images"]), 3))
            for idx, item in enumerate(msg["images"]):
                with cols[idx % 3]:
                    st.image(item["image"], caption=f"Figure : {item['nom']} (Page {item['page']})", use_container_width=True)

# Entrée utilisateur
prompt = st.chat_input("Posez votre question (ex: Pourquoi utiliser une attention multi-têtes ? ou Retrouve la figure 2...)")

if prompt or image_requete:
    requete_texte = prompt if prompt else "Analyse cette figure et retrouve les passages correspondants dans le document."

    # Ajout du message utilisateur
    st.session_state.messages.append({"role": "user", "content": requete_texte, "images": []})
    with st.chat_message("user"):
        st.markdown(requete_texte)
        if image_requete and not prompt:
            st.image(image_requete, caption="Image soumise en entrée", width=320)

    # Réponse de l'agent
    with st.chat_message("assistant"):
        with st.status("🧠 Cycle ReAct en cours...", expanded=True) as status:
            # 1. Thought
            st.write("💭 **Pensée (Thought) :** Analyse sémantique de la demande et formulation de la requête vectorielle.")

            # 2. Action / Tool Call
            action_desc = f"`retriever_tool(question='{requete_texte}', type_resultat='{mode_recherche}', k={top_k})`"
            st.write(f"🦾 **Action (Tool Call) :** {action_desc}")

            # Exécution de la recherche
            if image_requete and not prompt:
                res = rechercher(image_pil=image_requete, type_resultat=mode_recherche, k=top_k)
            else:
                res = rechercher(texte=requete_texte, type_resultat=mode_recherche, k=top_k)

            # 3. Observation
            docs = res["documents"][0]
            metas = res["metadatas"][0]
            dists = res["distances"][0]
            st.write(f"👁️ **Observation (Tool Output) :** {len(docs)} source(s) multimodale(s) identifiée(s) dans ChromaDB.")

            status.update(label="✅ Raisonnement ReAct terminé avec succès !", state="complete", expanded=False)

        # Structuration des sources
        sources_structurees = []
        images_trouvees = []

        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
            sim = 1.0 - dist
            num_page = meta.get("page", -1)
            source_info = {
                "ref": f"S{i}",
                "type": meta.get("type", "texte"),
                "page": num_page,
                "source": meta.get("source", ""),
                "contenu": doc,
                "sim": sim
            }
            sources_structurees.append(source_info)

            if meta.get("type") == "image":
                chemin_img = Path(meta["source"])
                if chemin_img.is_file():
                    with Image.open(chemin_img) as im:
                        images_trouvees.append({
                            "image": im.copy(),
                            "page": num_page,
                            "nom": chemin_img.name
                        })

        # 4. Synthèse finale
        reponse_finale = generer_synthese_reponse(
            question=requete_texte,
            sources=sources_structurees,
            api_key=cle_api
        )

        st.markdown(reponse_finale)

        # Affichage des figures si présentes
        if images_trouvees:
            st.markdown("#### 🖼️ Figures et Schémas Retrouvés :")
            cols = st.columns(min(len(images_trouvees), 3))
            for idx, item in enumerate(images_trouvees):
                with cols[idx % 3]:
                    st.image(item["image"], caption=f"Figure : {item['nom']} (Page {item['page']})", use_container_width=True)

        # Mémorisation dans l'état de session
        st.session_state.messages.append({
            "role": "assistant",
            "content": reponse_finale,
            "images": images_trouvees
        })
