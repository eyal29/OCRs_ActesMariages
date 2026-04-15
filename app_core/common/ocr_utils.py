import re
import unicodedata
from typing import TypedDict


class OCRWord(TypedDict):
    text: str
    confidence: float | None
    is_doubtful: bool


class OCRLine(TypedDict):
    line_index: int
    text: str
    marked_text: str
    words: list[OCRWord]


class OCRPage(TypedDict):
    page_index: int
    raw_text: str
    marked_text: str
    lines: list[OCRLine]


class OCRStats(TypedDict):
    word_count: int
    doubtful_word_count: int
    average_confidence: float | None
    low_confidence_ratio: float
    confidence_threshold: float


class OCRResult(TypedDict):
    engine: dict
    document: dict
    text: dict
    analysis: dict
    pages: list[OCRPage]
    raw_text: str
    marked_text: str
    doubtful_words: list[dict]


def clean_text(text):
    return unicodedata.normalize("NFKD", text).encode("latin-1", "ignore").decode("latin-1")


def get_standard_ocr_output_schema():
    return {
        "engine": {
            "name": "str - nom de la technique OCR",
            "family": "str - famille / librairie utilisée",
            "supports_confidence": "bool - présence de scores de confiance",
        },
        "document": {
            "source_name": "str - nom du fichier traité",
            "mime_type": "str | None - type MIME si disponible",
            "page_count": "int - nombre de pages/images traitées",
        },
        "text": {
            "raw": "str - texte OCR brut",
            "marked": "str - texte avec marquage des mots douteux",
            "normalized": "str - texte normalisé optionnellement",
        },
        "analysis": {
            "doubtful_words": [
                {
                    "word": "str - mot signalé comme douteux",
                    "confidence": "float | None - score du moteur",
                }
            ],
            "stats": {
                "word_count": "int - nombre total de mots",
                "doubtful_word_count": "int - nombre de mots douteux",
                "average_confidence": "float | None - moyenne des scores",
                "low_confidence_ratio": "float - proportion de mots douteux",
                "confidence_threshold": "float - seuil utilisé",
            },
        },
        "pages": [
            {
                "page_index": "int - index de page",
                "raw_text": "str - texte brut de la page",
                "marked_text": "str - texte balisé de la page",
                "lines": [
                    {
                        "line_index": "int - index de ligne",
                        "text": "str - texte brut de la ligne",
                        "marked_text": "str - ligne avec marquage",
                        "words": [
                            {
                                "text": "str - mot OCR",
                                "confidence": "float | None - score du mot",
                                "is_doubtful": "bool - vrai si sous le seuil",
                            }
                        ],
                    }
                ],
            }
        ],
        "raw_text": "str - alias de compatibilité",
        "marked_text": "str - alias de compatibilité",
        "doubtful_words": "list - alias de compatibilité",
    }


def _build_texts_from_pages_data(pages_data, confidence_threshold=0.6):
    raw_pages = []
    marked_pages = []
    doubtful_words = []

    for page_lines in pages_data:
        raw_lines = []
        marked_lines = []

        for line_words in page_lines:
            raw_line_words = []
            marked_line_words = []

            for item in line_words:
                word_text = item["text"]
                word_conf = item["confidence"]

                raw_line_words.append(word_text)

                if word_conf is not None and word_conf < confidence_threshold:
                    marked_line_words.append(f"⟦{word_text}?⟧")
                    doubtful_words.append({"word": word_text, "confidence": word_conf})
                else:
                    marked_line_words.append(word_text)

            raw_lines.append(" ".join(raw_line_words))
            marked_lines.append(" ".join(marked_line_words))

        raw_pages.append("\n".join(raw_lines))
        marked_pages.append("\n".join(marked_lines))

    raw_text = "\n\n".join(raw_pages).strip()
    marked_text = "\n\n".join(marked_pages).strip()
    return raw_text, marked_text, doubtful_words


def _build_pages_schema(pages_data, confidence_threshold=0.6):
    pages = []

    for page_index, page_lines in enumerate(pages_data):
        lines = []
        raw_lines = []
        marked_lines = []

        for line_index, line_words in enumerate(page_lines):
            words = []
            raw_line_words = []
            marked_line_words = []

            for item in line_words:
                word_text = item["text"]
                word_conf = item["confidence"]
                is_doubtful = word_conf is not None and word_conf < confidence_threshold
                marked_word = f"⟦{word_text}?⟧" if is_doubtful else word_text

                words.append({
                    "text": word_text,
                    "confidence": word_conf,
                    "is_doubtful": is_doubtful,
                })
                raw_line_words.append(word_text)
                marked_line_words.append(marked_word)

            raw_line_text = " ".join(raw_line_words)
            marked_line_text = " ".join(marked_line_words)
            raw_lines.append(raw_line_text)
            marked_lines.append(marked_line_text)
            lines.append({
                "line_index": line_index,
                "text": raw_line_text,
                "marked_text": marked_line_text,
                "words": words,
            })

        pages.append({
            "page_index": page_index,
            "raw_text": "\n".join(raw_lines).strip(),
            "marked_text": "\n".join(marked_lines).strip(),
            "lines": lines,
        })

    return pages


def _compute_stats(pages_data, doubtful_words, confidence_threshold=0.6):
    confidences = [
        item["confidence"]
        for page_lines in pages_data
        for line_words in page_lines
        for item in line_words
        if item["confidence"] is not None
    ]
    word_count = sum(
        len(line_words)
        for page_lines in pages_data
        for line_words in page_lines
    )
    doubtful_word_count = len(doubtful_words)
    average_confidence = sum(confidences) / len(confidences) if confidences else None
    low_confidence_ratio = doubtful_word_count / word_count if word_count else 0.0

    return {
        "word_count": word_count,
        "doubtful_word_count": doubtful_word_count,
        "average_confidence": average_confidence,
        "low_confidence_ratio": low_confidence_ratio,
        "confidence_threshold": confidence_threshold,
    }


def detect_paragraphs_by_gap(
    lines_geom_data,
    gap_multiplier: float = 1.5,
    confidence_threshold: float = 0.6,
) -> list[dict]:
    """
    Détecte les paragraphes en mesurant les espaces verticaux entre lignes.

    lines_geom_data : list[list[dict]]  (pages > lignes)
    Chaque ligne : {y_min, y_max, words: [...]}

    Algorithme :
      1. Aplatir toutes les lignes et les trier par y_min (haut → bas).
      2. Calculer la hauteur médiane des lignes.
      3. Nouveau paragraphe quand l’écart entre deux lignes dépasse
         gap_multiplier × hauteur_médiane.

    Retourne : list[{"paragraph_index": int, "raw_text": str, "marked_text": str}]
    """
    # Aplatir toutes les lignes de toutes les pages
    all_lines = [line for page_lines in lines_geom_data for line in page_lines]
    if not all_lines:
        return []

    # Trier par position verticale
    all_lines = sorted(all_lines, key=lambda l: l["y_min"])

    # Hauteur médiane des lignes
    heights = sorted([max(l["y_max"] - l["y_min"], 0.001) for l in all_lines])
    median_height = heights[len(heights) // 2]
    gap_threshold = gap_multiplier * median_height

    # Grouper les lignes en paragraphes
    groups: list[list[dict]] = [[all_lines[0]]]
    for i in range(1, len(all_lines)):
        prev_y_max = all_lines[i - 1]["y_max"]
        curr_y_min = all_lines[i]["y_min"]
        if (curr_y_min - prev_y_max) > gap_threshold:
            groups.append([])
        groups[-1].append(all_lines[i])

    # Construire le texte brut/balisé de chaque paragraphe
    paragraphs = []
    for para_idx, para_lines in enumerate(groups):
        raw_lines, marked_lines = [], []
        for line in para_lines:
            raw_parts, marked_parts = [], []
            for item in line["words"]:
                word_text = item["text"]
                word_conf = item["confidence"]
                raw_parts.append(word_text)
                if word_conf is not None and word_conf < confidence_threshold:
                    marked_parts.append(f"⟦{word_text}?⟧")
                else:
                    marked_parts.append(word_text)
            if raw_parts:
                raw_lines.append(" ".join(raw_parts))
                marked_lines.append(" ".join(marked_parts))
        raw_para = "\n".join(raw_lines).strip()
        marked_para = "\n".join(marked_lines).strip()
        if raw_para:
            paragraphs.append({
                "paragraph_index": para_idx,
                "raw_text": raw_para,
                "marked_text": marked_para,
            })
    return paragraphs


def _is_separator_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    punct_count = sum(1 for ch in stripped if ch in ".•·-_=*~")
    alpha_count = sum(1 for ch in stripped if ch.isalpha())
    punct_ratio = punct_count / max(len(stripped), 1)
    return punct_ratio >= 0.35 and alpha_count <= max(2, len(stripped) // 4)


def _looks_like_marriage_start(text: str) -> bool:
    line = clean_text(text).lower().strip()
    if len(line) < 18:
        return False

    start_ok = re.match(r"^(le|la|l[ei1]|ce|c[eé])\b", line) is not None
    if not start_ok:
        return False

    months = (
        "janvier", "fevrier", "février", "mars", "avril", "mai", "juin",
        "juillet", "aout", "août", "septembre", "octobre", "novembre", "decembre", "décembre",
    )
    has_month = any(month in line for month in months)
    return has_month


def detect_paragraphs_from_pages(pages: list[OCRPage]) -> list[dict]:
    """
    Détection amont générique des paragraphes/actes à partir des lignes OCR,
    utilisable par tous les moteurs OCR (DocTR, Tesseract, EasyOCR, PaddleOCR).
    """
    all_lines: list[tuple[str, str]] = []
    for page in pages:
        for line in page.get("lines", []):
            raw = (line.get("text") or "").strip()
            marked = (line.get("marked_text") or raw).strip()
            if raw:
                all_lines.append((raw, marked))

    if not all_lines:
        return []

    paragraphs: list[dict] = []
    current_raw: list[str] = []
    current_marked: list[str] = []

    def flush_current() -> None:
        if not current_raw:
            return
        raw_text = "\n".join(current_raw).strip()
        marked_text = "\n".join(current_marked).strip()
        if raw_text:
            paragraphs.append(
                {
                    "paragraph_index": len(paragraphs),
                    "raw_text": raw_text,
                    "marked_text": marked_text,
                }
            )

    for raw_line, marked_line in all_lines:
        if _is_separator_line(raw_line):
            if len(current_raw) >= 2:
                flush_current()
                current_raw = []
                current_marked = []
            continue

        if _looks_like_marriage_start(raw_line) and len(current_raw) >= 2:
            flush_current()
            current_raw = []
            current_marked = []

        current_raw.append(raw_line)
        current_marked.append(marked_line)

    flush_current()

    return paragraphs


def build_standard_ocr_result(
    *,
    engine_name,
    engine_family,
    source_name,
    mime_type,
    pages_data,
    confidence_threshold,
    lines_geom_data=None,
    blocks_data=None,       # conservé pour compatibilité ascendante (ignoré si lines_geom_data fourni)
):
    raw_text, marked_text, doubtful_words = _build_texts_from_pages_data(
        pages_data,
        confidence_threshold=confidence_threshold,
    )
    pages = _build_pages_schema(
        pages_data,
        confidence_threshold=confidence_threshold,
    )
    stats = _compute_stats(
        pages_data,
        doubtful_words,
        confidence_threshold=confidence_threshold,
    )

    paragraphs = detect_paragraphs_from_pages(pages)

    return {
        "engine": {
            "name": engine_name,
            "family": engine_family,
            "supports_confidence": True,
        },
        "document": {
            "source_name": source_name,
            "mime_type": mime_type,
            "page_count": len(pages),
        },
        "text": {
            "raw": raw_text,
            "marked": marked_text,
            "normalized": clean_text(raw_text),
        },
        "analysis": {
            "doubtful_words": doubtful_words,
            "stats": stats,
        },
        "paragraphs": paragraphs,
        "pages": pages,
        "_lines_geom_data": lines_geom_data or [],
        "raw_text": raw_text,
        "marked_text": marked_text,
        "doubtful_words": doubtful_words,
    }
