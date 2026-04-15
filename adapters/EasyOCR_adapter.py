import ast
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium
from pdf2image import convert_from_bytes

from app_core.common.ocr_utils import build_standard_ocr_result


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


def _load_easyocr_functions_from_script():
    script_path = Path(__file__).resolve().parent.parent / "OCRs" / "easyocr.py"
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script_path))

    allowed_nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            filtered_node = _filter_import_node(node, {"easyocr"})
            if filtered_node is not None:
                allowed_nodes.append(filtered_node)
        elif isinstance(node, ast.FunctionDef):
            allowed_nodes.append(node)

    module = ast.Module(body=allowed_nodes, type_ignores=[])
    module = ast.fix_missing_locations(module)

    namespace = {}
    exec(compile(module, filename=str(script_path), mode="exec"), namespace, namespace)
    return namespace


_EASYOCR_NS = _load_easyocr_functions_from_script()


@lru_cache(maxsize=1)
def _get_easyocr_reader():
    import easyocr

    raw_languages = os.getenv("EASYOCR_LANGS", "fr").strip()
    languages = [
        lang.strip()
        for lang in raw_languages.replace(",", "+").split("+")
        if lang.strip()
    ] or ["fr"]

    use_gpu = os.getenv("EASYOCR_GPU", "false").strip().lower() in {"1", "true", "yes", "on"}
    return easyocr.Reader(languages, gpu=use_gpu)


def is_easyocr_available():
    try:
        _get_easyocr_reader()
        return True, None
    except Exception as exc:
        return False, str(exc)


def _normalize_confidence(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _decode_image_bytes_to_grayscale(file_bytes: bytes):
    np_bytes = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_bytes, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Impossible de décoder l'image envoyée pour EasyOCR.")
    return image


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


def _run_easyocr_on_region(image: np.ndarray, reader) -> list[list[dict]]:
    detect_text_regions = _EASYOCR_NS["detect_text_regions"]

    boxes = detect_text_regions(image)
    if not boxes:
        height, width = image.shape[:2]
        boxes = [(0, 0, width, height)]

    lines = []
    for x, y, w, h in boxes:
        roi = image[y:y + h, x:x + w]
        results = reader.readtext(roi, text_threshold=0.4, low_text=0.3, detail=1)
        for result in results:
            if len(result) < 3:
                continue

            _, text, confidence = result
            text = str(text).strip()
            if not text:
                continue

            normalized_confidence = _normalize_confidence(confidence)
            words = [
                {"text": word, "confidence": normalized_confidence}
                for word in text.split()
                if word.strip()
            ]
            if words:
                lines.append(words)

    return lines


def _process_single_cv2_image(image: np.ndarray, reader) -> list[list[dict]]:
    _, width = image.shape[:2]
    middle = int(width * 0.5)

    left_img = image[:, :middle]
    right_img = image[:, middle:]

    page_lines = []
    page_lines.extend(_run_easyocr_on_region(left_img, reader))
    page_lines.extend(_run_easyocr_on_region(right_img, reader))
    return page_lines


def extract_ocr_data_from_image(uploaded_file, show_detection_window=False, confidence_threshold=0.6):
    _ = show_detection_window

    reader = _get_easyocr_reader()

    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    source_name = getattr(uploaded_file, "name", "uploaded_file")
    mime_type = getattr(uploaded_file, "type", None)

    is_pdf = (mime_type == "application/pdf") or source_name.lower().endswith(".pdf")

    if is_pdf:
        pages_data = [
            _process_single_cv2_image(page_gray, reader)
            for page_gray in _rasterize_pdf_to_cv2_images(file_bytes)
        ]
    else:
        image = _decode_image_bytes_to_grayscale(file_bytes)
        pages_data = [_process_single_cv2_image(image, reader)]

    return build_standard_ocr_result(
        engine_name="EasyOCR",
        engine_family="easyocr",
        source_name=source_name,
        mime_type=mime_type,
        pages_data=pages_data,
        confidence_threshold=confidence_threshold,
    )