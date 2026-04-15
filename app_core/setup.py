import os
import streamlit as st
from dotenv import load_dotenv
from app_core.config import CUSTOM_CSS, PAGE_CONFIG, SESSION_DEFAULTS, get_api_env_var_for_provider

def initialize_app() -> None:
    """Initialise Streamlit, l'environnement et le state applicatif."""
    st.set_page_config(**PAGE_CONFIG)
    load_dotenv()
    for key, default_value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    st.session_state["is_processing"] = bool(st.session_state["is_processing"])
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def reset_ocr_state() -> None:
    """Réinitialise les artefacts OCR stockés en session."""
    st.session_state["ocr_results"] = {}
    st.session_state["ocr_final_texts"] = {}
    st.session_state["ocr_llm_outputs"] = {}
    st.session_state["ocr_structured_outputs"] = {}
    st.session_state["ocr_structured_extraction"] = None
    st.session_state["ocr_structured_source_engine"] = None
    st.session_state["ocr_segment_preview"] = None
    st.session_state["ocr_segment_boxes"] = []
    st.session_state["ocr_segment_count"] = 0
    
@st.cache_resource
def get_groq_api_key() -> str:
    """Lit la clé API Groq depuis les variables d'environnement."""
    return os.getenv("GROQ_API_KEY_CY", "")


def get_mistral_api_key() -> str:
    """Lit la clé API Mistral depuis les variables d'environnement."""
    return os.getenv("MISTRAL_API_KEY", "")


def get_api_key_for_provider(provider: str) -> str:
    """Lit la clé API correspondant au provider LLM sélectionné."""
    env_var = get_api_env_var_for_provider(provider)
    return os.getenv(env_var, "")
