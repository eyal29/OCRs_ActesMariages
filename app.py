import streamlit as st
from app_core.config import get_enabled_techniques
from app_core.upload import (
    build_input_payload,
    deduplicate_uploaded_files,
    notify_duplicates,
    render_upload_preview,
    sync_payload_to_session,
)
from app_core.pipeline.ocr_runner import run_ocr_pipeline
from app_core.setup import initialize_app
from app_core.ui.chatbot import render_chatbot_page
from app_core.ui.results import render_results
from app_core.ui.sidebar import render_sidebar


def main():
    initialize_app()
    enabled_techniques = get_enabled_techniques()
    sidebar_config = render_sidebar(enabled_techniques)

    if sidebar_config["page"] == "Chatbot BDD":
        render_chatbot_page(
            llm_provider=sidebar_config["llm_provider"],
            llm_model=sidebar_config["llm_model"],
        )
        return

    st.title("📄 OCR Studio — Comparatif manuscrits")

    col_upload, col_run = st.columns([1.35, 1], gap="large")
    with col_upload:
        st.header("📤 Ingestion des documents")
        uploaded_files = st.file_uploader(
            "Charger des images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Format supporté: PNG, JPG, JPEG",
        )

    with col_run:
        st.header("🚀 Exécution")
        st.caption("Lance l'OCR après avoir chargé au moins un fichier.")
        trigger_ocr = st.button("Lancer les OCRs", type="primary", width="stretch")

    if not uploaded_files:
        st.info("👆 Uploadez une image pour commencer l'OCR.")
        return

    unique_files, unique_files_bytes, duplicate_names = deduplicate_uploaded_files(uploaded_files)
    notify_duplicates(duplicate_names)

    payload = build_input_payload(unique_files, unique_files_bytes)
    payload = sync_payload_to_session(payload)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fichiers uniques", len(unique_files))
    col2.metric("Mode d'analyse", 'Multi' if payload['is_multi'] else 'Simple')
    col3.metric("Nom de base", payload['file_base_name'])
    col4.metric("Techniques actives", len(sidebar_config['selected_techniques']))

    st.header("🖼️ Aperçu des entrées")
    with st.expander("Voir les fichiers chargés"):
        render_upload_preview(unique_files, payload["is_multi"])
    st.divider()

    if trigger_ocr:
        run_ocr_pipeline(
            selected_techniques=sidebar_config["selected_techniques"],
            confidence_threshold=sidebar_config["confidence_threshold"],
            show_detection_window=sidebar_config["show_detection_window"],
            llm_provider=sidebar_config["llm_provider"],
            llm_model=sidebar_config["llm_model"],
        )

    render_results(file_base_name=payload["file_base_name"])


main()