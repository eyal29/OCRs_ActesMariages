import os

import streamlit as st

from adapters.Tesseract_adapter import is_tesseract_available
from app_core.config import (
    DEFAULT_LLM_PROVIDER,
    LLM_MODEL_OPTIONS,
    SidebarConfig,
    get_api_env_var_for_provider,
    get_default_model_for_provider,
)
from app_core.setup import get_api_key_for_provider


def configure_tesseract_sidebar() -> None:
    """Affiche les paramètres de runtime Tesseract et l'état de disponibilité."""
    st.subheader("🛠️ Runtime Tesseract")
    if os.name == "nt":
        tesseract_cmd_input = st.text_input(
            "Chemin vers tesseract.exe (optionnel)",
            value=os.getenv("TESSERACT_CMD", ""),
            help="Exemple: C:/Program Files/Tesseract-OCR/tesseract.exe",
        )
        if tesseract_cmd_input.strip():
            os.environ["TESSERACT_CMD"] = tesseract_cmd_input.strip()
        if os.getenv("TESSDATA_PREFIX", "").strip():
            st.caption(f"TESSDATA_PREFIX: {os.getenv('TESSDATA_PREFIX')}")
    else:
        st.caption("ℹ️ macOS/Linux : Tesseract est détecté via le PATH système.")

    tesseract_ok, tesseract_error = is_tesseract_available()
    if tesseract_ok:
        st.success("✅ Tesseract détecté")
    else:
        if os.name == "nt":
            st.warning("⚠️ Tesseract non détecté. Renseignez le chemin de tesseract.exe ci-dessus.")
        else:
            st.warning("⚠️ Tesseract non détecté. Vérifiez son installation et le PATH système.")
        with st.expander("Détail erreur Tesseract"):
            st.code(tesseract_error or "Erreur inconnue")

    tesseract_lang_input = st.text_input(
        "Langues Tesseract (ex: fra, fra+eng, eng)",
        value=os.getenv("TESSERACT_LANG", "fra"),
        help="Ordre conseillé: fra, puis fallback automatique vers fra+eng puis eng",
    )
    if tesseract_lang_input.strip():
        os.environ["TESSERACT_LANG"] = tesseract_lang_input.strip()


def render_sidebar(enabled_techniques: list[str]) -> SidebarConfig:
    """Construit la sidebar et retourne la configuration choisie par l'utilisateur."""
    with st.sidebar:
        st.markdown("### 🧩 Control Center")

        page = st.radio(
            "Navigation",
            options=["OCR Studio", "Chatbot BDD"],
            key="app_page",
        )

        st.divider()

        # LLM — toujours visible (utile pour les deux pages)
        st.subheader("✨ LLM de correction sémantique")
        llm_provider = st.selectbox(
            "Provider",
            options=list(LLM_MODEL_OPTIONS.keys()),
            index=list(LLM_MODEL_OPTIONS.keys()).index(DEFAULT_LLM_PROVIDER),
            help="Provider LLM utilisé pour la correction et le chatbot.",
        )
        llm_model = st.selectbox(
            "Modèle",
            options=LLM_MODEL_OPTIONS[llm_provider],
            index=0,
        )
        api_env_var = get_api_env_var_for_provider(llm_provider)
        if not get_api_key_for_provider(llm_provider):
            st.error(f"❌ {api_env_var} manquante")
        else:
            st.success(f"✅ {api_env_var} détectée")

        # Éléments OCR uniquement sur la page OCR Studio
        if page == "OCR Studio":
            st.divider()

            if st.session_state["is_processing"]:
                st.info("Traitement OCR en cours...")
                if st.button("Déverrouiller la navigation"):
                    st.session_state["is_processing"] = False
                    st.rerun()

            st.subheader("📊 Moteurs OCR")
            selected_techniques = st.multiselect(
                "Sélectionner les techniques que vous souhaitez tester",
                options=enabled_techniques,
                default=enabled_techniques,
                help="Sélectionnez les techniques OCR à comparer",
            )

            st.subheader("🎛️ Paramètres généraux")
            confidence_threshold = st.slider(
                "Seuil de confiance (mots douteux)",
                min_value=0.10,
                max_value=0.99,
                value=0.60,
                step=0.01,
                help="Mots avec confiance < seuil sont marqués comme douteux",
            )

            show_detection_window = st.checkbox(
                "🔍 Fenêtre de détection externe",
                value=False,
                help="Ouvre la visualisation DocTR dans une fenêtre séparée",
            )

            configure_tesseract_sidebar()
        else:
            # Valeurs par défaut pour les champs OCR non affichés
            selected_techniques = enabled_techniques
            confidence_threshold = 0.60
            show_detection_window = False

    return {
        "page": page,
        "selected_techniques": selected_techniques,
        "confidence_threshold": confidence_threshold,
        "show_detection_window": show_detection_window,
        "llm_provider": llm_provider,
        "llm_model": llm_model or get_default_model_for_provider(llm_provider),
    }
