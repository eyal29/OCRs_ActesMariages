import base64
import os
from typing import Any

import requests

from app_core.common.ocr_utils import build_standard_ocr_result


def _get_mistral_api_key() -> str:
    return os.getenv("MISTRAL_API_KEY", "").strip()


def _build_data_uri(file_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _extract_page_text(page: dict[str, Any]) -> str:
    for key in ("markdown", "text", "content", "ocr_text"):
        value = page.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _response_to_pages_data(payload: dict[str, Any]) -> list[list[list[dict[str, Any]]]]:
    pages_data: list[list[list[dict[str, Any]]]] = []

    pages = payload.get("pages")
    if isinstance(pages, list) and pages:
        for page in pages:
            if not isinstance(page, dict):
                continue
            page_text = _extract_page_text(page)
            page_lines: list[list[dict[str, Any]]] = []
            for raw_line in page_text.splitlines():
                words = [word for word in raw_line.strip().split() if word]
                if words:
                    page_lines.append(
                        [{"text": word, "confidence": None} for word in words]
                    )
            pages_data.append(page_lines)

    if not pages_data:
        fallback_text = ""
        for key in ("text", "markdown", "content"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                fallback_text = value.strip()
                break

        page_lines = []
        for raw_line in fallback_text.splitlines():
            words = [word for word in raw_line.strip().split() if word]
            if words:
                page_lines.append(
                    [{"text": word, "confidence": None} for word in words]
                )
        pages_data = [page_lines]

    return pages_data


def _call_mistral_ocr(api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    endpoints = [
        "https://api.mistral.ai/v1/ocr",
        "https://api.mistral.ai/v1/ocr/process",
    ]

    last_error = None
    for endpoint in endpoints:
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
            if response.ok:
                return response.json()
            last_error = f"{response.status_code}: {response.text}"
        except Exception as exc:
            last_error = str(exc)

    raise ValueError(f"Appel Mistral OCR échoué: {last_error}")


def extract_ocr_data_from_image(
    uploaded_file,
    show_detection_window: bool = False,
    confidence_threshold: float = 0.6,
):
    _ = show_detection_window

    api_key = _get_mistral_api_key()
    if not api_key:
        raise ValueError("MISTRAL_API_KEY manquante dans l'environnement (.env).")

    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    source_name = getattr(uploaded_file, "name", "uploaded_file")
    mime_type = getattr(uploaded_file, "type", None) or "application/octet-stream"

    data_uri = _build_data_uri(file_bytes, mime_type)
    if str(mime_type).startswith("image/"):
        document = {"type": "image_url", "image_url": data_uri}
    else:
        document = {"type": "document_url", "document_url": data_uri}

    request_payload = {
        "model": os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest"),
        "document": document,
    }

    response_payload = _call_mistral_ocr(api_key=api_key, payload=request_payload)
    pages_data = _response_to_pages_data(response_payload)

    return build_standard_ocr_result(
        engine_name="MistralOCR",
        engine_family="mistral-ocr",
        source_name=source_name,
        mime_type=mime_type,
        pages_data=pages_data,
        confidence_threshold=confidence_threshold,
    )
