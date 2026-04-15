"""Prétraitement des images : segmentation en actes individuels.

Utilise le pipeline de decoupage_v2.py pour détecter les boîtes de chaque acte
avant de les passer aux moteurs OCR.
"""

import cv2
import numpy as np

from app_core.pipeline.decoupage import detect_paragraph_boxes, draw_boxes, resize_if_needed


def _decode_image(file_bytes: bytes):
    img_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    return resize_if_needed(image_bgr, max_width=2200)


def get_segment_preview(file_bytes: bytes) -> tuple[list[tuple[int, int, int, int]], bytes | None]:
    """Retourne les boîtes détectées et une image annotée PNG pour l'aperçu."""
    try:
        image_bgr = _decode_image(file_bytes)
        if image_bgr is None:
            return [], None

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        boxes, _, _ = detect_paragraph_boxes(gray)
        if not boxes:
            return [], None

        annotated = draw_boxes(image_bgr, boxes)
        success, encoded = cv2.imencode(".png", annotated)
        if not success:
            return list(boxes), None

        normalized_boxes = [tuple(map(int, box)) for box in boxes]
        return normalized_boxes, encoded.tobytes()
    except Exception:
        return [], None


def segment_image_bytes(file_bytes: bytes) -> list[bytes]:
    """Segmente une image en actes individuels.

    Prend les bytes bruts d'une image (PNG/JPEG/…) et retourne une liste de
    bytes PNG (un par acte détecté), dans l'ordre de lecture
    (colonne gauche → colonne droite, haut → bas).

    Retourne une liste vide si la segmentation échoue, si l'entrée n'est pas
    une image décodable, ou si aucun acte n'est détecté.
    """
    try:
        image_bgr = _decode_image(file_bytes)
        if image_bgr is None:
            return []

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        boxes, _, _ = detect_paragraph_boxes(gray)
        if not boxes:
            return []

        h, w = gray.shape[:2]
        crops: list[bytes] = []
        for (x1, y1, x2, y2) in boxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = gray[y1:y2, x1:x2]
            success, encoded = cv2.imencode(".png", crop)
            if success:
                crops.append(encoded.tobytes())

        return crops

    except Exception:
        return []
