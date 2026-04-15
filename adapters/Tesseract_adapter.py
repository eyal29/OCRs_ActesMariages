import ast
import io
import os
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytesseract
from app_core.common.ocr_utils import build_standard_ocr_result
from pdf2image import convert_from_bytes
import pypdfium2 as pdfium



def _load_tesseract_functions_from_script():
    script_path = Path(__file__).resolve().parent.parent / "OCRs" / "Tesseract.py"
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script_path))
    allowed_nodes = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
    ]
    module = ast.Module(body=allowed_nodes, type_ignores=[])
    compiled = compile(module, filename=str(script_path), mode="exec")
    namespace = {}
    exec(compiled, namespace, namespace)
    return namespace

_TESSERACT_NS = _load_tesseract_functions_from_script()

# ── Tesseract config

def set_tesseract_executable_path(executable_path: str | None):
    if executable_path:
        pytesseract.pytesseract.tesseract_cmd = executable_path

def _configure_tesseract_runtime():
    if os.name != "nt":
        return
    env_tesseract_cmd = os.getenv("TESSERACT_CMD", "").strip()
    if env_tesseract_cmd:
        set_tesseract_executable_path(env_tesseract_cmd)
        if not os.getenv("TESSDATA_PREFIX", "").strip():
            candidate = Path(env_tesseract_cmd).parent / "tessdata"
            if candidate.exists() and candidate.is_dir():
                os.environ["TESSDATA_PREFIX"] = str(candidate)

def is_tesseract_available():
    try:
        pytesseract.get_tesseract_version()
        return True, None
    except Exception as exc:
        return False, str(exc)

# ── Helpers internes 

def _build_pages_data_from_text_and_confidences(text, confidences):
    """Convertit (texte brut, dict confiances) -> format pages_data du schema standard."""
    page_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line_words = []
        for word in line.split():
            raw_conf = confidences.get(word)
            confidence = max(0.0, min(1.0, float(raw_conf) / 100.0)) if raw_conf is not None else None
            line_words.append({"text": word, "confidence": confidence})
        if line_words:
            page_lines.append(line_words)
    return [page_lines]

# Decoupe l'image en deux colonnes (gauche/droite) et OCR chaque region.
def _extract_text_left_right_with_language(image, language):
    detect_text_regions = _TESSERACT_NS["detect_text_regions"]
    ocr_with_confidence = _TESSERACT_NS["ocr_with_confidence"]

    _, w = image.shape
    middle = int(w * 0.5)
    left_img = image[:, :middle]
    right_img = image[:, middle:]
    config = f"--oem 3 --psm 6 -l {language}"

    left_texts, right_texts = [], []
    all_confidences = {}

    for (x, y, bw, bh) in detect_text_regions(left_img):
        text, conf_dict = ocr_with_confidence(left_img[y:y+bh, x:x+bw], config)
        left_texts.append(text)
        all_confidences.update(conf_dict)

    for (x, y, bw, bh) in detect_text_regions(right_img):
        text, conf_dict = ocr_with_confidence(right_img[y:y+bh, x:x+bw], config)
        right_texts.append(text)
        all_confidences.update(conf_dict)

    full_text = (
        "===== PAGE GAUCHE =====\n" + "\n".join(left_texts)
        + "\n\n\n"
        + "===== PAGE DROITE =====\n" + "\n".join(right_texts)
    )
    return full_text, all_confidences


def _rasterize_pdf_to_cv2_images(pdf_bytes: bytes, dpi: int = 200) -> list:
    """
    Convertit chaque page d'un PDF en image 
    Essaie pypdfium2 en priorite, puis pdf2image comme fallback.
    """
    # Tentative 1 : pypdfium2 (rapide, sans dependance systeme)
    try:
        pdf = pdfium.PdfDocument(pdf_bytes)
        scale = dpi / 72.0
        images = [
            np.array(pdf[i].render(scale=scale, rotation=0).to_pil().convert("L"))
            for i in range(len(pdf))
        ]
        pdf.close()
        return images
    except ImportError:
        pass
    except Exception as exc:
        print(f"[WARN] pypdfium2 echec: {exc}, tentative pdf2image...")

    # Tentative 2 : pdf2image (necessite poppler)
    try:
        return [np.array(p.convert("L")) for p in convert_from_bytes(pdf_bytes, dpi=dpi)]
    except ImportError:
        raise RuntimeError(
            "Aucune librairie de rasterisation PDF disponible. "
            "Installez 'pypdfium2' (pip install pypdfium2) "
            "ou 'pdf2image' + poppler (pip install pdf2image)."
        )
    except Exception as exc:
        raise RuntimeError(f"Impossible de rasteriser le PDF: {exc}")

def _process_single_cv2_image(image, lang_candidates):
    """Lance l'OCR Tesseract sur une image OpenCV pretraitee. Retourne (texte, confidences)."""
    last_error = None
    for lang in lang_candidates:
        try:
            raw_text, confidences = _extract_text_left_right_with_language(image, lang)
            clean = _TESSERACT_NS["clean_text"](raw_text)
            return clean, confidences
        except Exception as exc:
            last_error = exc

    raise ValueError(
        f"Impossible de charger les langues Tesseract. "
        f"Essayees: {', '.join(lang_candidates)}. Derniere erreur: {last_error}"
    )

def extract_ocr_data_from_image(uploaded_file, show_detection_window=False, confidence_threshold=0.6):
    _configure_tesseract_runtime()

    preferred_lang = os.getenv("TESSERACT_LANG", "fra").strip() or "fra"
    lang_candidates = [preferred_lang]
    for fallback in ["fra+eng", "eng"]:
        if fallback not in lang_candidates:
            lang_candidates.append(fallback)

    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    source_name = getattr(uploaded_file, "name", "uploaded_file")
    mime_type = getattr(uploaded_file, "type", None)

    is_pdf = (mime_type == "application/pdf") or source_name.lower().endswith(".pdf")

    if is_pdf:
        # PDF : rasteriser chaque page puis OCR
        all_pages_data = []
        for page_gray in _rasterize_pdf_to_cv2_images(file_bytes):
            _, thresh = cv2.threshold(page_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            raw_text, confidences = _process_single_cv2_image(thresh, lang_candidates)
            all_pages_data.extend(_build_pages_data_from_text_and_confidences(raw_text, confidences))
    else:
        # Image unique : ecrire dans un fichier temporaire pour OpenCV
        suffix = os.path.splitext(source_name)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name
        try:
            processed = _TESSERACT_NS["preprocess_image_for_ocr"](temp_path)
            raw_text, confidences = _process_single_cv2_image(processed, lang_candidates)
            all_pages_data = _build_pages_data_from_text_and_confidences(raw_text, confidences)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return build_standard_ocr_result(
        engine_name="Tesseract",
        engine_family="pytesseract",
        source_name=source_name,
        mime_type=mime_type,
        pages_data=all_pages_data,
        confidence_threshold=confidence_threshold,
    )