import matplotlib
import streamlit as st
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from app_core.common.ocr_utils import build_standard_ocr_result
import io


def show_doctr_detection_window(result, title=None):
    """
    Ouvre une fenêtre de visualisation hors Streamlit via les utilitaires DocTR.
    """
    try:
        if title:
            print(f"[INFO] Ouverture de la fenêtre de détection: {title}")
        import matplotlib.pyplot as plt

        current_backend = matplotlib.get_backend().lower()
        if "agg" in current_backend:
            switched = False
            for backend_name in ("TkAgg", "QtAgg"):
                try:
                    plt.switch_backend(backend_name)
                    switched = True
                    print(f"[INFO] Backend matplotlib basculé vers: {backend_name}")
                    break
                except Exception:
                    continue

            if not switched:
                print("[WARN] Aucun backend interactif matplotlib disponible (TkAgg/QtAgg).")

        result.show()
    except Exception as e:
        print(f"[WARN] Impossible d'ouvrir la fenêtre de détection DocTR: {e}")


def _extract_pages_word_confidences(result):
    """
    Retourne :
      - pages_data      : list[list[list[dict]]]  (pages > lignes > mots) — format plat inchangé
      - lines_geom_data : list[list[dict]]        (pages > lignes avec géométrie + mots)
        Chaque ligne : {y_min, y_max, x_min, x_max, words: [...]}
    """
    pages_data = []
    lines_geom_data = []

    for page in result.pages:
        page_lines_flat = []
        page_lines_geom = []

        for block in page.blocks:
            for line in block.lines:
                line_words = []
                for word in line.words:
                    confidence = getattr(word, "confidence", None)
                    line_words.append({
                        "text": word.value,
                        "confidence": float(confidence) if confidence is not None else None,
                    })

                if line_words:
                    page_lines_flat.append(line_words)
                    # line.geometry = ((x_min, y_min), (x_max, y_max)) en coordonnées relatives [0, 1]
                    geom = line.geometry
                    page_lines_geom.append({
                        "y_min": geom[0][1],
                        "y_max": geom[1][1],
                        "x_min": geom[0][0],
                        "x_max": geom[1][0],
                        "words": line_words,
                    })

        pages_data.append(page_lines_flat)
        lines_geom_data.append(page_lines_geom)

    return pages_data, lines_geom_data


def extract_ocr_data_from_image(uploaded_file, show_detection_window=False, confidence_threshold=0.6):
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    source_name = getattr(uploaded_file, "name", "uploaded_file")
    mime_type = getattr(uploaded_file, "type", None)

    # Détecter PDF via mime_type ou extension du nom de fichier
    is_pdf = (mime_type == "application/pdf") or (source_name.lower().endswith(".pdf"))

    if is_pdf:
        img_doc = DocumentFile.from_pdf(file_bytes)
    else:
        img_doc = DocumentFile.from_images([file_bytes])

    model = ocr_predictor(pretrained=True)
    result = model(img_doc)

    if show_detection_window:
        show_doctr_detection_window(result, title=source_name)

    pages_data, lines_geom_data = _extract_pages_word_confidences(result)
    return build_standard_ocr_result(
        engine_name="DocTR",
        engine_family="python-doctr",
        source_name=source_name,
        mime_type=mime_type,
        pages_data=pages_data,
        lines_geom_data=lines_geom_data,
        confidence_threshold=confidence_threshold,
    )