import ast
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
import pypdfium2 as pdfium
from pdf2image import convert_from_bytes
from app_core.common.ocr_utils import build_standard_ocr_result
from paddleocr import PaddleOCR

def _filter_import_node(node, excluded_import_roots: set[str]):
    if isinstance(node, ast.Import):
        kept_aliases = [
            alias for alias in node.names
            if alias.name.split(".")[0] not in excluded_import_roots
        ]
        if not kept_aliases:
            return None
        return ast.Import(names=kept_aliases)

    if isinstance(node, ast.ImportFrom):
        module_root = (node.module or "").split(".")[0]
        if module_root in excluded_import_roots:
            return None
        return node

    return node


def _load_paddleocr_functions_from_script():
    script_path = Path(__file__).resolve().parent.parent / "OCRs" / "PaddleOCR.py"
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script_path))

    allowed_nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            filtered_node = _filter_import_node(node, {"paddleocr"})
            if filtered_node is not None:
                allowed_nodes.append(filtered_node)
        elif isinstance(node, ast.FunctionDef):
            allowed_nodes.append(node)

    module = ast.Module(body=allowed_nodes, type_ignores=[])
    module = ast.fix_missing_locations(module)

    namespace = {}
    exec(compile(module, filename=str(script_path), mode="exec"), namespace, namespace)
    return namespace

_PADDLEOCR_NS = _load_paddleocr_functions_from_script()

def _is_pir_runtime_error(exc: Exception) -> bool:
    message = str(exc)
    return (
        "ConvertPirAttribute2RuntimeAttribute" in message
        or "onednn_instruction.cc" in message
    )

def _build_pir_runtime_error_message(exc: Exception) -> str:
    return (
        "Incompatibilité Paddle détectée (PIR/oneDNN) sur cet environnement. "
        "Solution recommandée: utiliser un environnement Python 3.12 (ou 3.11) puis réinstaller "
        "paddlepaddle/paddleocr. "
        f"Python courant: {sys.version.split()[0]}. Détail original: {exc}"
    )

@lru_cache(maxsize=1)
def _get_paddleocr_engine():
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_enable_pir_api", "0")

    language = os.getenv("PADDLEOCR_LANG", "fr").strip() or "fr"
    return PaddleOCR(
        use_textline_orientation=False,
        lang=language,
    )

def is_paddleocr_available():
    try:
        _get_paddleocr_engine()
        return True, None
    except Exception as exc:
        return False, str(exc)

def _normalize_confidence(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None

def _rasterize_pdf_to_cv2_images(pdf_bytes: bytes, dpi: int = 200) -> list[np.ndarray]:
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
        print(f"[WARN] pypdfium2 échec: {exc}, tentative pdf2image...")

    try:
        return [np.array(page.convert("L")) for page in convert_from_bytes(pdf_bytes, dpi=dpi)]
    except ImportError:
        raise RuntimeError(
            "Aucune librairie de rasterisation PDF disponible. "
            "Installez 'pypdfium2' ou 'pdf2image'."
        )
    except Exception as exc:
        raise RuntimeError(f"Impossible de rasteriser le PDF: {exc}")


def _preprocess_uploaded_image(file_bytes: bytes, source_name: str) -> np.ndarray:
    preprocess_image_for_ocr = _PADDLEOCR_NS["preprocess_image"]
    suffix = os.path.splitext(source_name)[1] or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name
    try:
        return preprocess_image_for_ocr(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def _iter_paddle_predictions(region: np.ndarray, engine):
    # preprocess_image retourne déjà du BGR (3 canaux) — conversion uniquement si GRAY
    region_bgr = region if len(region.shape) == 3 else cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)
    ocr_error = None

    if hasattr(engine, "ocr"):
        try:
            raw_results = engine.ocr(region_bgr, cls=False) or []
            for page_result in raw_results:
                for item in page_result or []:
                    if not isinstance(item, (list, tuple)) or len(item) < 2:
                        continue
                    text_info = item[1]
                    if not isinstance(text_info, (list, tuple)) or len(text_info) < 2:
                        continue
                    text = str(text_info[0]).strip()
                    if text:
                        yield text, _normalize_confidence(text_info[1])
            return
        except Exception as exc:
            if _is_pir_runtime_error(exc):
                raise RuntimeError(_build_pir_runtime_error_message(exc)) from exc
            ocr_error = exc

    if hasattr(engine, "predict"):
        try:
            raw_results = engine.predict(region_bgr) or []
            for result in raw_results:
                if not isinstance(result, dict):
                    continue
                texts = result.get("rec_texts", [])
                scores = result.get("rec_scores", [])
                for text, score in zip(texts, scores):
                    text = str(text).strip()
                    if text:
                        yield text, _normalize_confidence(score)
            return
        except Exception as exc:
            if _is_pir_runtime_error(exc):
                raise RuntimeError(_build_pir_runtime_error_message(exc)) from exc
            if ocr_error is not None:
                raise RuntimeError(f"PaddleOCR indisponible sur cet environnement: {ocr_error}") from exc
            raise

    if ocr_error is not None:
        if _is_pir_runtime_error(ocr_error):
            raise RuntimeError(_build_pir_runtime_error_message(ocr_error)) from ocr_error
        raise RuntimeError(f"PaddleOCR indisponible sur cet environnement: {ocr_error}")


def _run_paddle_on_region(region: np.ndarray, engine) -> list[list[dict]]:
    lines = []
    for text, confidence in _iter_paddle_predictions(region, engine):
        words = [
            {"text": word, "confidence": confidence}
            for word in text.split()
            if word.strip()
        ]
        if words:
            lines.append(words)
    return lines


def _process_single_cv2_image(image: np.ndarray, engine) -> list[list[dict]]:
    _, width = image.shape[:2]
    middle = int(width * 0.5)
    left_img = image[:, :middle]
    right_img = image[:, middle:]
    page_lines = []
    page_lines.extend(_run_paddle_on_region(left_img, engine))
    page_lines.extend(_run_paddle_on_region(right_img, engine))
    return page_lines


def extract_ocr_data_from_image(uploaded_file, show_detection_window=False, confidence_threshold=0.6):
    _ = show_detection_window
    engine = _get_paddleocr_engine()
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    source_name = getattr(uploaded_file, "name", "uploaded_file")
    mime_type = getattr(uploaded_file, "type", None)

    is_pdf = (mime_type == "application/pdf") or source_name.lower().endswith(".pdf")

    if is_pdf:
        pages_data = []
        for page_gray in _rasterize_pdf_to_cv2_images(file_bytes):
            _, thresh = cv2.threshold(page_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            pages_data.append(_process_single_cv2_image(thresh, engine))
    else:
        processed = _preprocess_uploaded_image(file_bytes, source_name)
        pages_data = [_process_single_cv2_image(processed, engine)]

    return build_standard_ocr_result(
        engine_name="PaddleOCR",
        engine_family="paddleocr",
        source_name=source_name,
        mime_type=mime_type,
        pages_data=pages_data,
        confidence_threshold=confidence_threshold,
    )