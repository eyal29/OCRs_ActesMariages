import hashlib
import io
import os
from typing import Any
import img2pdf
import streamlit as st
from PIL import Image as PILImage
from streamlit_pdf_viewer import pdf_viewer as st_pdf_viewer
from app_core.config import InputPayload, UploadedFileBytes
from app_core.setup import reset_ocr_state


def deduplicate_uploaded_files(uploaded_files: list[Any]) -> tuple[list[Any], list[UploadedFileBytes], list[str]]:
    """Supprime les doublons d'upload basés sur le hash du contenu."""
    seen_hashes = set()
    duplicate_names = []
    unique_files = []
    unique_files_bytes = []

    for uploaded in uploaded_files:
        uploaded.seek(0)
        file_bytes = uploaded.read()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        if file_hash in seen_hashes:
            duplicate_names.append(uploaded.name)
            continue
        seen_hashes.add(file_hash)
        unique_files.append(uploaded)
        unique_files_bytes.append((uploaded, file_bytes))

    return unique_files, unique_files_bytes, duplicate_names


def notify_duplicates(duplicate_names):
    """Affiche une notification unique pour les fichiers uploadés en doublon."""
    if duplicate_names:
        duplicate_unique = sorted(set(duplicate_names))
        duplicate_sig = "|".join(duplicate_unique)
        if st.session_state.get("last_duplicate_toast_sig") != duplicate_sig:
            for name in duplicate_unique:
                st.toast(f"{name} déjà uploadé — ignoré", icon="⚠️")
            st.session_state["last_duplicate_toast_sig"] = duplicate_sig
    else:
        st.session_state["last_duplicate_toast_sig"] = None


def build_pdf_base_name(uploaded_files: list[Any]) -> str:
    """Construit un nom de PDF lisible à partir des noms de fichiers source."""
    stems = [os.path.splitext(file.name)[0] for file in uploaded_files]
    prefix = os.path.commonprefix(stems)
    if prefix and prefix[-1] not in ("_", "-"):
        last_sep = max(prefix.rfind("_"), prefix.rfind("-"))
        prefix = prefix[:last_sep + 1] if last_sep >= 0 else ""

    suffixes = [stem[len(prefix):] for stem in stems]
    if prefix and all(suffixes):
        return (prefix.rstrip("_-") + "_" + "-".join(suffixes))[:80]
    return "-".join(stems)[:80]


def create_pdf_from_images(unique_files_bytes: list[UploadedFileBytes]) -> bytes:
    """Assemble une liste d'images bytes en un seul PDF."""
    image_bytes_list = [file_bytes for _, file_bytes in unique_files_bytes]
    try:
        return img2pdf.convert(image_bytes_list)
    except Exception:
        pil_images = [PILImage.open(io.BytesIO(img_bytes)).convert("RGB") for img_bytes in image_bytes_list]
        buffer = io.BytesIO()
        pil_images[0].save(buffer, format="PDF", save_all=True, append_images=pil_images[1:])
        return buffer.getvalue()


def render_pdf_preview(file_bytes: bytes, file_name: str) -> None:
    """Affiche un aperçu du PDF généré et propose son téléchargement."""
    _, col_expander, _ = st.columns([1.5, 7, 1.5])
    with col_expander:
        with st.expander("📄 Aperçu du PDF assemblé", expanded=False):
            st.caption(file_name)
            st_pdf_viewer(
                input=file_bytes,
                width="100%",
                height=520,
                zoom_level="auto",
                viewer_align="center",
            )
            st.download_button(
                label="⬇️ Télécharger le PDF",
                data=file_bytes,
                file_name=file_name,
                mime="application/pdf",
                key="dl_pdf_preview",
            )


def build_input_payload(unique_files: list[Any], unique_files_bytes: list[UploadedFileBytes]) -> InputPayload:
    """Prépare le payload unique (image ou PDF) consommé par les moteurs OCR."""
    if len(unique_files) == 1:
        single_file, single_file_bytes = unique_files_bytes[0]
        return {
            "file_bytes": single_file_bytes,
            "file_name": single_file.name,
            "file_type": getattr(single_file, "type", "image/jpeg"),
            "file_base_name": os.path.splitext(single_file.name)[0],
            "signature": hashlib.sha256(single_file_bytes).hexdigest(),
            "is_multi": False,
        }

    os.makedirs("pdf", exist_ok=True)
    uploaded_signature = ":".join(hashlib.sha256(file_bytes).hexdigest() for _, file_bytes in unique_files_bytes)
    base_pdf_name = build_pdf_base_name(unique_files)
    pdf_path = os.path.join("pdf", f"{base_pdf_name}.pdf")

    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as file:
            pdf_bytes = file.read()
        st.info(f"📂 PDF existant réutilisé : `{os.path.basename(pdf_path)}`")
    else:
        pdf_bytes = create_pdf_from_images(unique_files_bytes)
        with open(pdf_path, "wb") as file:
            file.write(pdf_bytes)
        st.success(f"📄 {len(unique_files)} images fusionnées → `{os.path.basename(pdf_path)}`")

    file_name = os.path.basename(pdf_path)
    render_pdf_preview(pdf_bytes, file_name)

    return {
        "file_bytes": pdf_bytes,
        "file_name": file_name,
        "file_type": "application/pdf",
        "file_base_name": os.path.splitext(file_name)[0],
        "signature": uploaded_signature,
        "is_multi": True,
    }


def sync_payload_to_session(payload: InputPayload) -> InputPayload:
    """Synchronise le payload courant avec le session state Streamlit."""
    if st.session_state["ocr_file_signature"] != payload["signature"]:
        reset_ocr_state()
        st.session_state["ocr_file_signature"] = payload["signature"]
        st.session_state["_file_bytes"] = payload["file_bytes"]
        st.session_state["_file_name"] = payload["file_name"]
        st.session_state["_file_type"] = payload["file_type"]
        st.session_state["_file_base_name"] = payload["file_base_name"]
        return payload

    payload["file_bytes"] = st.session_state.get("_file_bytes", payload["file_bytes"])
    payload["file_name"] = st.session_state.get("_file_name", payload["file_name"])
    payload["file_type"] = st.session_state.get("_file_type", payload["file_type"])
    payload["file_base_name"] = st.session_state.get("_file_base_name", payload["file_base_name"])
    return payload


def render_upload_preview(unique_files: list[Any], is_multi: bool) -> None:
    """Affiche un aperçu du contenu uploadé."""
    if is_multi:
        st.write(f"{len(unique_files)} images chargées (fusionnées en PDF)")
    else:
        uploaded_file = unique_files[0]
        annotated_preview = st.session_state.get("ocr_segment_preview")
        segment_count = int(st.session_state.get("ocr_segment_count") or 0)

        tab_original, tab_detected = st.tabs(["Original", "Zones détectées"])

        with tab_original:
            st.image(uploaded_file, caption=uploaded_file.name, width="stretch")

        with tab_detected:
            if annotated_preview:
                st.image(
                    annotated_preview,
                    caption=f"{uploaded_file.name} — {segment_count} zone(s) détectée(s)",
                    width="stretch",
                )
            else:
                st.info("Les zones détectées apparaîtront ici après la segmentation OCR.")

        if annotated_preview:
            st.caption(
                f"Aperçu de segmentation disponible: {segment_count} zone(s) détectée(s)."
            )