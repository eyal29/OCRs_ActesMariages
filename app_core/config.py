from typing import Any, Callable, TypedDict

from OCRs.DocTR import extract_ocr_data_from_image
from adapters.EasyOCR_adapter import extract_ocr_data_from_image as extract_easyocr_data_from_image
from adapters.MistralOCR_adapter import extract_ocr_data_from_image as extract_mistral_ocr_data_from_image
from adapters.PaddleOCR_adapter import extract_ocr_data_from_image as extract_paddleocr_data_from_image
from adapters.Tesseract_adapter import extract_ocr_data_from_image as extract_tesseract_ocr_data_from_image

class OCRTechniqueConfig(TypedDict):
    enabled: bool
    description: str
    icon: str
    function: Callable[..., dict[str, Any]]

class SidebarConfig(TypedDict):
    page: str
    selected_techniques: list[str]
    confidence_threshold: float
    show_detection_window: bool
    llm_provider: str
    llm_model: str

class InputPayload(TypedDict):
    file_bytes: bytes
    file_name: str
    file_type: str
    file_base_name: str
    signature: str
    is_multi: bool

UploadedFileBytes = tuple[Any, bytes]

PAGE_CONFIG = {
    "page_title": "OCR Manuscrits - Comparatif",
    "page_icon": "📄",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

SESSION_DEFAULTS = {
    "ocr_results": {},
    "ocr_final_texts": {},
    "ocr_llm_outputs": {},
    "ocr_structured_outputs": {},
    "ocr_structured_extraction": None,
    "ocr_structured_source_engine": None,
    "ocr_segment_preview": None,
    "ocr_segment_boxes": [],
    "ocr_segment_count": 0,
    "chat_messages": [],
    "ocr_file_signature": None,
    "is_processing": False,
    "last_duplicate_toast_sig": None,
}

DEFAULT_LLM_PROVIDER = "Mistral"

LLM_MODEL_OPTIONS: dict[str, list[str]] = {
    "Groq": [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
    ],
    "Mistral": [
        "mistral-small-latest",
        "mistral-medium-latest",
        "mistral-large-latest",
    ],
}

LLM_API_ENV_VARS: dict[str, str] = {
    "Groq": "GROQ_API_KEY_CY",
    "Mistral": "MISTRAL_API_KEY",
}


def get_default_model_for_provider(provider: str) -> str:
    models = LLM_MODEL_OPTIONS.get(provider) or LLM_MODEL_OPTIONS[DEFAULT_LLM_PROVIDER]
    return models[0]


def get_api_env_var_for_provider(provider: str) -> str:
    return LLM_API_ENV_VARS.get(provider, LLM_API_ENV_VARS[DEFAULT_LLM_PROVIDER])

CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    .app-hero {
        padding: 1.1rem 1.15rem;
        border-radius: 16px;
        border: 1px solid rgba(120, 120, 120, 0.3);
        background:
            radial-gradient(1200px 420px at 100% -20%, rgba(34, 211, 238, 0.16), transparent 42%),
            radial-gradient(1200px 420px at 0% -30%, rgba(59, 130, 246, 0.16), transparent 38%),
            linear-gradient(135deg, rgba(17, 24, 39, 0.08), rgba(76, 29, 149, 0.10));
        margin-bottom: 0.9rem;
    }
    .app-hero h1 {
        margin: 0;
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: .1px;
    }
    .app-hero p {
        margin: .35rem 0 .75rem 0;
        opacity: .92;
        font-size: .98rem;
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: .45rem;
    }
    .chip {
        border: 1px solid rgba(120, 120, 120, 0.35);
        border-radius: 999px;
        padding: .2rem .62rem;
        font-size: .82rem;
        background: rgba(255,255,255,.04);
    }

    .panel {
        border: 1px solid rgba(120, 120, 120, 0.22);
        border-radius: 14px;
        padding: .8rem .9rem;
        background: rgba(255,255,255,.02);
        margin: .55rem 0;
    }
    .panel h3 {
        margin: 0 0 .45rem 0;
        font-size: 1rem;
        font-weight: 700;
    }

    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: .55rem;
        margin: .5rem 0 .2rem 0;
    }
    .kpi {
        border: 1px solid rgba(120, 120, 120, 0.23);
        border-radius: 12px;
        padding: .55rem .65rem;
        background: rgba(255,255,255,.02);
    }
    .kpi .label {
        font-size: .78rem;
        opacity: .82;
    }
    .kpi .value {
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: .08rem;
    }

    div[data-baseweb="tab-list"] {
        gap: .36rem;
        margin-bottom: .4rem;
    }
    div[data-baseweb="tab"] {
        border: 1px solid rgba(120, 120, 120, 0.27);
        border-radius: 10px;
        padding: .2rem .7rem;
        background: rgba(255,255,255,.01);
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
        padding-top: .44rem;
        padding-bottom: .44rem;
    }

    [data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid rgba(120,120,120,.22);
        overflow: hidden;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: .7rem;
    }
    [data-testid="stSidebar"] .sidebar-section {
        border: 1px solid rgba(120,120,120,.22);
        border-radius: 12px;
        padding: .65rem .72rem;
        background: rgba(255,255,255,.02);
        margin-bottom: .62rem;
    }
    [data-testid="stSidebar"] .sidebar-section h4 {
        margin: 0 0 .35rem 0;
        font-size: .95rem;
    }

    @media (max-width: 1100px) {
        .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
</style>
"""

OCR_TECHNIQUES: dict[str, OCRTechniqueConfig] = {
    "DocTR": {
        "enabled": True,
        "description": "Document Text Recognition - Modèle de deep learning pour manuscrits",
        "icon": "🩺​",
        "function": extract_ocr_data_from_image,
    },
    "Tesseract": {
        "enabled": True,
        "description": "OCR classique (adapté depuis votre script collègue)",
        "icon": "🔎",
        "function": extract_tesseract_ocr_data_from_image,
    },
    "EasyOCR": {
        "enabled": True,
        "description": "EasyOCR adapté au même schéma standard que Tesseract",
        "icon": "🪶",
        "function": extract_easyocr_data_from_image,
    },
    "PaddleOCR": {
        "enabled": True,
        "description": "PaddleOCR adapté au même schéma standard que Tesseract",
        "icon": "🏓",
        "function": extract_paddleocr_data_from_image,
    },
    "MistralOCR": {
        "enabled": True,
        "description": "OCR cloud via Mistral OCR API",
        "icon": "🧠",
        "function": extract_mistral_ocr_data_from_image,
    },
}

def get_enabled_techniques() -> list[str]:
    """Retourne la liste des moteurs OCR activés dans la configuration."""
    return [name for name, config in OCR_TECHNIQUES.items() if config["enabled"]]
