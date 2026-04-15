import io
import json
import hashlib
import os
from typing import Any
import streamlit as st
from app_core.common.ocr_utils import clean_text
from app_core.config import OCR_TECHNIQUES, get_api_env_var_for_provider
from app_core.pipeline.database import save_all_extractions
from app_core.pipeline.llm_client import extract_all_marriages_data, semantic_correct_text
from app_core.pipeline.preprocessor import get_segment_preview, segment_image_bytes
from app_core.setup import get_api_key_for_provider, reset_ocr_state

# CACHE DISQUE
# Un seul fichier JSON par image 
CACHE_DIR = "Archives/ocr_cache"
CACHEABLE_TECHNIQUES = {"PaddleOCR", "EasyOCR", "DocTR", "Tesseract", "MistralOCR"}

STRUCTURED_OCR_KEY = "DocTR"


def _choose_structured_source_engine(
    results: dict,
    selected_techniques: list[str],
) -> str | None:
    # Préférence: DocTR si disponible (meilleure segmentation pour l'extraction structurée)
    if STRUCTURED_OCR_KEY in results:
        return STRUCTURED_OCR_KEY
    
    # Fallback: candidats parmi les techniques sélectionnées
    candidates = [tech for tech in selected_techniques if tech in results]
    if not candidates:
        candidates = list(results.keys())
    if not candidates:
        return None

    def score(engine_name: str) -> tuple[int, int]:
        ocr_data = results.get(engine_name, {})
        paragraph_count = len(ocr_data.get("paragraphs") or [])
        word_count = int((ocr_data.get("analysis") or {}).get("stats", {}).get("word_count") or 0)
        return paragraph_count, word_count

    return max(candidates, key=score)

def _unified_cache_path(file_signature: str, confidence_threshold: float) -> str:
    """Chemin du fichier cache JSON unifié pour une image (tous moteurs confondus)."""
    key = hashlib.sha256(
        f"{file_signature}:{confidence_threshold}".encode()
    ).hexdigest()
    return os.path.join(CACHE_DIR, f"{key}.json")


def _load_unified_cache(path: str) -> dict:
    """
    Charge le cache unifié depuis le disque.
    Retourne un dict { "meta": {...}, "engines": {...} }.
    Retourne une structure vide si le fichier est absent ou corrompu.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Validation minimale de la structure
        if isinstance(data, dict) and "engines" in data:
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {"meta": {}, "engines": {}}


def _save_unified_cache(path: str, cache: dict) -> None:
    """Sauvegarde le cache unifié sur le disque (écrase l'existant)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _is_cacheable(technique: str) -> bool:
    """Retourne True si le moteur doit être mis en cache disque."""
    return technique in CACHEABLE_TECHNIQUES


def _build_cache_entry(ocr_data: dict) -> dict:
    """Construit une entrée de cache compatible avec un stockage par paragraphe."""
    paragraphs = [
        item.get("raw_text", "").strip()
        for item in (ocr_data.get("paragraphs") or [])
        if item.get("raw_text", "").strip()
    ]
    return {
        "data": ocr_data,
        "by_paragraph": [
            {"paragraph_index": idx, "raw_text": text}
            for idx, text in enumerate(paragraphs)
        ],
        "paragraph_count": len(paragraphs),
    }


def _read_cache_entry(cache_entry: dict) -> tuple[dict, list[str]]:
    """Lit une entrée de cache (nouveau format ou ancien) et retourne (ocr_data, paragraphes)."""
    # Nouveau format
    if isinstance(cache_entry, dict) and "data" in cache_entry:
        ocr_data = cache_entry.get("data") or {}
        by_paragraph = cache_entry.get("by_paragraph") or []
        paragraphs = [
            item.get("raw_text", "").strip()
            for item in by_paragraph
            if item.get("raw_text", "").strip()
        ]
        if not paragraphs:
            paragraphs = [
                item.get("raw_text", "").strip()
                for item in (ocr_data.get("paragraphs") or [])
                if item.get("raw_text", "").strip()
            ]
        return ocr_data, paragraphs

    # Ancien format (ocr_data direct)
    ocr_data = cache_entry if isinstance(cache_entry, dict) else {}
    paragraphs = [
        item.get("raw_text", "").strip()
        for item in (ocr_data.get("paragraphs") or [])
        if item.get("raw_text", "").strip()
    ]
    return ocr_data, paragraphs

# CORRECTION LLM
def apply_llm_correction(
    raw_text: str,
    marked_text: str,
    doubtful_words: list[dict[str, Any]],
    llm_provider: str,
    llm_model: str,
):
    """Applique la correction sémantique LLM avec fallback gracieux."""
    api_key = get_api_key_for_provider(llm_provider)
    api_env_var = get_api_env_var_for_provider(llm_provider)
    if not api_key:
        st.warning(f"⚠️ {api_env_var} manquante - correction LLM désactivée.")
        return None

    try:
        with st.spinner(f"🔄 Correction sémantique via {llm_provider}..."):
            return semantic_correct_text(
                raw_text=raw_text,
                marked_text=marked_text,
                doubtful_words=doubtful_words,
                provider=llm_provider,
                api_key=api_key,
                model=llm_model,
            )
    except Exception as exc:
        st.warning(f"⚠️ Correction LLM via {llm_provider} temporairement indisponible: {exc}")
        return None

# PIPELINE PRINCIPAL

def _run_single_technique(
    technique: str,
    ocr_file_bytes: bytes,
    ocr_file_name: str,
    ocr_file_type: str,
    confidence_threshold: float,
    show_detection_window: bool,
    llm_provider: str,
    llm_model: str,
) -> tuple[dict, str, dict | None]:
    """Exécute un moteur OCR + correction LLM. Retourne (ocr_data, final_text, llm_output)."""
    file_for_ocr = io.BytesIO(ocr_file_bytes)
    file_for_ocr.name = ocr_file_name
    file_for_ocr.type = ocr_file_type

    ocr_data = OCR_TECHNIQUES[technique]["function"](
        file_for_ocr,
        show_detection_window=show_detection_window,
        confidence_threshold=confidence_threshold,
    )

    extracted_text = clean_text(ocr_data["text"]["raw"])
    marked_text = clean_text(ocr_data["text"]["marked"])
    doubtful_words = ocr_data["analysis"]["doubtful_words"]

    final_text = extracted_text
    llm_output = apply_llm_correction(
        extracted_text, marked_text, doubtful_words, llm_provider, llm_model
    )
    if llm_output:
        final_text = llm_output["corrected_text"]

    return ocr_data, final_text, llm_output


def _run_technique_on_crops(
    technique: str,
    crop_bytes_list: list[bytes],
    file_name: str,
    confidence_threshold: float,
    llm_provider: str,
    llm_model: str,
) -> tuple[dict, str, dict | None]:
    """Exécute un moteur OCR sur chaque acte cropé et fusionne les résultats.

    Chaque crop est traité indépendamment ; les textes extraits sont assemblés
    en paragraphes compatibles avec la pipeline d'extraction BDD.
    """
    crop_texts: list[str] = []
    crop_paragraphs: list[dict] = []
    all_doubtful: list[dict] = []
    all_confidences: list[float] = []
    merged_pages: list[dict] = []

    for i, crop_bytes in enumerate(crop_bytes_list):
        file_for_ocr = io.BytesIO(crop_bytes)
        file_for_ocr.name = f"acte_{i + 1:02d}.png"
        file_for_ocr.type = "image/png"
        try:
            ocr_data = OCR_TECHNIQUES[technique]["function"](
                file_for_ocr,
                show_detection_window=False,
                confidence_threshold=confidence_threshold,
            )
            raw_text = clean_text(ocr_data["text"]["raw"])
            if raw_text.strip():
                crop_texts.append(raw_text)
                crop_paragraphs.append({"raw_text": raw_text})
                all_doubtful.extend(
                    (ocr_data.get("analysis") or {}).get("doubtful_words") or []
                )

            for page in ocr_data.get("pages", []):
                lines = page.get("lines", [])
                for line in lines:
                    for word in line.get("words", []):
                        confidence = word.get("confidence")
                        if confidence is not None:
                            all_confidences.append(float(confidence))
                merged_pages.append(page)
        except Exception as exc:
            st.warning(f"⚠️ Acte {i + 1} ignoré ({technique}): {exc}")

    combined_raw = "\n\n".join(crop_texts)

    final_text = combined_raw
    llm_output = apply_llm_correction(
        raw_text=combined_raw,
        marked_text=combined_raw,
        doubtful_words=all_doubtful,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    if llm_output:
        final_text = llm_output["corrected_text"]

    word_count = len(combined_raw.split())
    doubtful_word_count = len(all_doubtful)
    low_confidence_ratio = doubtful_word_count / word_count if word_count else 0.0
    average_confidence = (
        sum(all_confidences) / len(all_confidences) if all_confidences else None
    )

    merged_ocr_data = {
        "engine": {
            "name": technique,
            "family": technique,
            "supports_confidence": average_confidence is not None,
        },
        "document": {
            "source_name": file_name,
            "mime_type": "image/png",
            "page_count": 1,
        },
        "text": {
            "raw": combined_raw,
            "marked": combined_raw,
            "normalized": clean_text(combined_raw),
        },
        "analysis": {
            "stats": {
                "word_count": word_count,
                "doubtful_word_count": doubtful_word_count,
                "average_confidence": average_confidence,
                "low_confidence_ratio": low_confidence_ratio,
                "confidence_threshold": confidence_threshold,
            },
            "doubtful_words": all_doubtful,
        },
        "paragraphs": crop_paragraphs,
        "pages": merged_pages,
        "_lines_geom_data": [],
        "raw_text": combined_raw,
        "marked_text": combined_raw,
        "doubtful_words": all_doubtful,
    }

    return merged_ocr_data, final_text, llm_output


def run_ocr_pipeline(
    selected_techniques: list[str],
    confidence_threshold: float,
    show_detection_window: bool,
    llm_provider: str,
    llm_model: str,
) -> None:
    """Exécute les moteurs OCR sélectionnés et stocke les résultats en session.

    Cache disque unifié : un seul fichier JSON par image regroupe les résultats
    de tous les moteurs cachés (PaddleOCR, EasyOCR...).
    Le fichier est chargé une fois en début de pipeline, mis à jour au fur et à
    mesure des moteurs exécutés, puis écrit sur le disque une seule fois à la fin.
    """
    st.session_state["is_processing"] = True
    reset_ocr_state()
    results = {}
    final_texts = {}
    llm_outputs = {}

    ocr_file_bytes = st.session_state["_file_bytes"]
    ocr_file_name  = st.session_state["_file_name"]
    ocr_file_type  = st.session_state["_file_type"]
    file_signature = st.session_state.get("ocr_file_signature", "")
    api_key        = get_api_key_for_provider(llm_provider)

    # ── Chargement du cache unifié une seule fois ──
    cache_path    = _unified_cache_path(file_signature, confidence_threshold)
    unified_cache = _load_unified_cache(cache_path)
    cache_updated = False  # flag pour ne sauvegarder que si nécessaire

    # --- Prétraitement : segmentation en actes individuels (images uniquement) ---
    crop_bytes_list: list[bytes] = []
    if not ocr_file_type.startswith("application/pdf"):
        with st.spinner("🔲 Segmentation de l'image en actes..."):
            boxes, preview_image = get_segment_preview(ocr_file_bytes)
            crop_bytes_list = segment_image_bytes(ocr_file_bytes)
        st.session_state["ocr_segment_boxes"] = boxes
        st.session_state["ocr_segment_preview"] = preview_image
        st.session_state["ocr_segment_count"] = len(boxes)
        if crop_bytes_list:
            st.info(
                f"✂️ {len(crop_bytes_list)} acte(s) détecté(s) — "
                "chaque acte sera traité individuellement par l'OCR."
            )
        else:
            st.warning(
                "⚠️ Segmentation non concluante — l'image sera traitée en entier."
            )
    else:
        st.session_state["ocr_segment_boxes"] = []
        st.session_state["ocr_segment_preview"] = None
        st.session_state["ocr_segment_count"] = 0

    try:
        # --- Étape 1 : exécuter tous les moteurs sélectionnés ---
        for technique in selected_techniques:

            # ── Lecture depuis le cache unifié (moteurs cachéables uniquement) ──
            if _is_cacheable(technique) and technique in unified_cache["engines"]:
                cached_entry = unified_cache["engines"][technique]
                cached_ocr_data, cached_paragraphs = _read_cache_entry(cached_entry)

                # Pour DocTR + segmentation crops: ignorer un cache sans paragraphes exploitables
                # afin d'éviter "0 segment(s) OCR" avec une segmentation image pourtant détectée.
                if technique == STRUCTURED_OCR_KEY and crop_bytes_list:
                    if not cached_paragraphs:
                        st.info(
                            "♻️ Cache DocTR sans paragraphes détecté: recalcul OCR sur les actes segmentés."
                        )
                    else:
                        st.info(f"⚡ {technique} : résultat chargé depuis le cache.")
                        ocr_data = cached_ocr_data
                        results[technique] = ocr_data

                        extracted_text = clean_text(ocr_data["text"]["raw"])
                        marked_text    = clean_text(ocr_data["text"]["marked"])
                        doubtful_words = ocr_data["analysis"]["doubtful_words"]
                        final_text     = extracted_text

                        llm_output = apply_llm_correction(
                            extracted_text, marked_text, doubtful_words,
                            llm_provider, llm_model,
                        )
                        if llm_output:
                            llm_outputs[technique] = llm_output
                            final_text = llm_output["corrected_text"]

                        final_texts[technique] = final_text
                        continue  # ← cache trouvé pour ce moteur
                else:
                    st.info(f"⚡ {technique} : résultat chargé depuis le cache.")
                    ocr_data = cached_ocr_data
                    results[technique] = ocr_data

                    extracted_text = clean_text(ocr_data["text"]["raw"])
                    marked_text    = clean_text(ocr_data["text"]["marked"])
                    doubtful_words = ocr_data["analysis"]["doubtful_words"]
                    final_text     = extracted_text

                    llm_output = apply_llm_correction(
                        extracted_text, marked_text, doubtful_words,
                        llm_provider, llm_model,
                    )
                    if llm_output:
                        llm_outputs[technique] = llm_output
                        final_text = llm_output["corrected_text"]

                    final_texts[technique] = final_text
                    continue  # ← cache trouvé pour ce moteur

            # ── OCR complet ──
            with st.spinner(f"Traitement avec {technique}..."):
                try:
                    if crop_bytes_list:
                        ocr_data, final_text, llm_output = _run_technique_on_crops(
                            technique,
                            crop_bytes_list,
                            ocr_file_name,
                            confidence_threshold,
                            llm_provider,
                            llm_model,
                        )
                    else:
                        ocr_data, final_text, llm_output = _run_single_technique(
                            technique,
                            ocr_file_bytes,
                            ocr_file_name,
                            ocr_file_type,
                            confidence_threshold,
                            show_detection_window,
                            llm_provider,
                            llm_model,
                        )
                    results[technique] = ocr_data
                    final_texts[technique] = final_text

                    # ── Mise à jour du cache unifié en mémoire ──
                    if _is_cacheable(technique):
                        unified_cache.setdefault("meta", {}).update({
                            "file_signature":     file_signature,
                            "confidence_threshold": confidence_threshold,
                        })
                        unified_cache.setdefault("engines", {})[technique] = _build_cache_entry(ocr_data)
                        cache_updated = True
                        st.success(f"💾 {technique} : résultat ajouté au cache unifié.")

                    if llm_output:
                        llm_outputs[technique] = llm_output
                except Exception as exc:
                    st.error(f"❌ Erreur {technique}: {exc}")

        # --- Étape 2 : extraction structurée (amont OCR, applicable à tous les moteurs) ---
        structured_extraction = None
        structured_source_engine = None
        if api_key:
            # Si aucun résultat exploitable, tenter DocTR en fallback
            if not results:
                with st.spinner(f"Exécution de {STRUCTURED_OCR_KEY} (fallback) pour l'extraction BDD..."):
                    try:
                        # Toujours utiliser les crops s'ils existent, même en fallback
                        if crop_bytes_list:
                            doctr_ocr_data, doctr_final_text, doctr_llm_output = _run_technique_on_crops(
                                STRUCTURED_OCR_KEY,
                                crop_bytes_list,
                                ocr_file_name,
                                confidence_threshold,
                                llm_provider,
                                llm_model,
                            )
                        else:
                            doctr_ocr_data, doctr_final_text, doctr_llm_output = _run_single_technique(
                                STRUCTURED_OCR_KEY,
                                ocr_file_bytes,
                                ocr_file_name,
                                ocr_file_type,
                                confidence_threshold,
                                show_detection_window,
                                llm_provider,
                                llm_model,
                            )
                        results[STRUCTURED_OCR_KEY] = doctr_ocr_data
                        final_texts[STRUCTURED_OCR_KEY] = doctr_final_text
                        if doctr_llm_output:
                            llm_outputs[STRUCTURED_OCR_KEY] = doctr_llm_output
                    except Exception as exc:
                        st.warning(f"⚠️ DocTR indisponible pour l'extraction: {exc}")

            # Toujours ajouter/mettre à jour DocTR pour garantir la cohérence structurée
            if STRUCTURED_OCR_KEY not in results:
                with st.spinner(f"Exécution de {STRUCTURED_OCR_KEY} pour garantir l'extraction BDD..."):
                    try:
                        # Toujours utiliser les crops s'ils existent pour la cohérence
                        if crop_bytes_list:
                            doctr_ocr_data, doctr_final_text, doctr_llm_output = _run_technique_on_crops(
                                STRUCTURED_OCR_KEY,
                                crop_bytes_list,
                                ocr_file_name,
                                confidence_threshold,
                                llm_provider,
                                llm_model,
                            )
                        else:
                            doctr_ocr_data, doctr_final_text, doctr_llm_output = _run_single_technique(
                                STRUCTURED_OCR_KEY,
                                ocr_file_bytes,
                                ocr_file_name,
                                ocr_file_type,
                                confidence_threshold,
                                show_detection_window,
                                llm_provider,
                                llm_model,
                            )
                        results[STRUCTURED_OCR_KEY] = doctr_ocr_data
                        final_texts[STRUCTURED_OCR_KEY] = doctr_final_text
                        if doctr_llm_output:
                            llm_outputs[STRUCTURED_OCR_KEY] = doctr_llm_output
                    except Exception as exc:
                        st.warning(f"⚠️ DocTR indisponible pour l'extraction: {exc}")

            structured_source_engine = _choose_structured_source_engine(results, selected_techniques)
            structured_text = final_texts.get(structured_source_engine) if structured_source_engine else None
            if structured_text:
                ocr_paragraphs = [
                    item.get("raw_text", "").strip()
                    for item in (results.get(structured_source_engine, {}).get("paragraphs") or [])
                    if item.get("raw_text", "").strip()
                ]

                use_ocr_split = len(ocr_paragraphs) > 0
                split_label = "segmentation OCR amont" if use_ocr_split else "segmentation LLM (fallback)"
                st.info(
                    f"🔎 Source structurée: {structured_source_engine} · "
                    f"{len(ocr_paragraphs)} segment(s) OCR · mode: {split_label}"
                )

                with st.spinner(f"🗄️ Extraction des actes de mariage ({structured_source_engine} → BDD)..."):
                    try:
                        structured_extraction = extract_all_marriages_data(
                            text=structured_text,
                            provider=llm_provider,
                            api_key=api_key,
                            model=llm_model,
                            paragraphs=ocr_paragraphs if use_ocr_split else None,
                        )
                        source_name = results[structured_source_engine].get("document", {}).get(
                            "source_name", ocr_file_name
                        )
                        nb = len(structured_extraction)
                        save_all_extractions(
                            source_name=source_name,
                            extractions=structured_extraction,
                        )
                        st.success(f"✅ {nb} acte(s) de mariage enregistré(s) dans la base de données.")
                    except Exception as exc:
                        st.warning(f"⚠️ Extraction / enregistrement BDD échoué: {exc}")
        else:
            api_env_var = get_api_env_var_for_provider(llm_provider)
            st.warning(
                f"⚠️ Extraction structurée non exécutée: variable {api_env_var} absente, "
                "aucune ligne enregistrée dans la BDD."
            )
    finally:
        # ── Écriture du cache unifié sur le disque (une seule fois) ──
        if cache_updated:
            _save_unified_cache(cache_path, unified_cache)
        
        st.session_state["is_processing"] = False

    st.session_state["ocr_results"] = results
    st.session_state["ocr_final_texts"] = final_texts
    st.session_state["ocr_llm_outputs"] = llm_outputs
    st.session_state["ocr_structured_extraction"] = structured_extraction
    st.session_state["ocr_structured_source_engine"] = structured_source_engine
