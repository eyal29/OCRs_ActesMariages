import json
import time

import pandas as pd
import requests


def get_standard_llm_output_schema():
    return {
        "corrected_text": "str - texte corrigé par le LLM",
        "remaining_doubtful_words": "list - mots restant incertains après correction",
    }


def get_marriage_extraction_schema():
    return {
        "lieu_acte": "str | null - lieu principal de l'acte de mariage",
        "date_acte": "str | null - date complète ou partielle de l'acte",
        "marie_1": {
            "nom": "str | null",
            "profession": "str | null",
            "adresse": "str | null",
            "pere": {
                "nom": "str | null",
                "profession": "str | null",
                "adresse": "str | null",
            },
            "mere": {
                "nom": "str | null",
                "profession": "str | null",
                "adresse": "str | null",
            },
        },
        "marie_2": {
            "nom": "str | null",
            "profession": "str | null",
            "adresse": "str | null",
            "pere": {
                "nom": "str | null",
                "profession": "str | null",
                "adresse": "str | null",
            },
            "mere": {
                "nom": "str | null",
                "profession": "str | null",
                "adresse": "str | null",
            },
        },
    }


def _get_llm_endpoint(provider: str) -> str:
    endpoints = {
        "Groq": "https://api.groq.com/openai/v1/chat/completions",
        "Mistral": "https://api.mistral.ai/v1/chat/completions",
    }
    if provider not in endpoints:
        raise ValueError(f"Provider LLM non supporté: {provider}")
    return endpoints[provider]


def _extract_message_content(message_content):
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        parts = []
        for item in message_content:
            if isinstance(item, dict):
                text_value = item.get("text")
                if text_value:
                    parts.append(text_value)
        return "".join(parts)
    return ""


def _post_llm_chat_completion(provider: str, api_key: str, payload: dict):
    endpoint = _get_llm_endpoint(provider)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    max_attempts = 3
    response = None
    for attempt in range(1, max_attempts + 1):
        response = requests.post(endpoint, headers=headers, json=payload, timeout=90)
        if response.status_code != 429:
            break

        if attempt == max_attempts:
            break

        retry_after_header = response.headers.get("Retry-After")
        try:
            retry_after = float(retry_after_header) if retry_after_header else None
        except Exception:
            retry_after = None

        wait_seconds = retry_after if retry_after is not None else (2 ** attempt)
        time.sleep(min(wait_seconds, 12))

    if response is None:
        raise ValueError(f"Erreur {provider}: aucune réponse reçue.")

    if not response.ok:
        raise ValueError(f"Erreur {provider} ({response.status_code}): {response.text}")

    return response.json()


def extract_marriage_data(
    text: str,
    api_key: str,
    provider: str = "Groq",
    model: str = "llama-3.3-70b-versatile",
):
    if not api_key:
        return None

    system_prompt = (
        "Tu es un expert en extraction d'informations d'actes de mariage anciens. "
        "Ta mission est d'extraire uniquement les informations explicitement présentes dans le texte. "
        "Si une information n'est pas certaine ou absente, retourne null."
    )

    user_prompt = (
        "Extrait les informations principales de l'acte ci-dessous et réponds en JSON strict.\n"
        "Champs attendus: lieu_acte, date_acte, marie_1, marie_2.\n"
        "Pour chaque marié, extrais: nom, profession, adresse, puis pour pere et mere: nom, profession, adresse.\n"
        "Ne complète jamais avec des suppositions. Utilise null si la donnée n'est pas lisible ou absente.\n\n"
        f"Schéma cible: {json.dumps(get_marriage_extraction_schema(), ensure_ascii=False)}\n\n"
        "Texte à analyser:\n"
        f"{text}"
    )

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }

    data = _post_llm_chat_completion(provider=provider, api_key=api_key, payload=payload)
    content = _extract_message_content(data["choices"][0]["message"]["content"])
    return json.loads(content)


def generate_sql_for_marriages(
    question: str,
    schema: dict,
    api_key: str,
    provider: str = "Groq",
    model: str = "llama-3.3-70b-versatile",
) -> dict:
    """Génère une requête SQL SELECT sécurisée pour interroger les actes de mariage."""
    if not api_key:
        raise ValueError(f"Clé API {provider} manquante.")

    preferred_relation = schema.get("preferred_relation", "v_chatbot_actes")
    allowed_relations = schema.get(
        "allowed_relations",
        ["v_chatbot_actes", "actes", "personnes", "acte_personnes", "actes_mariage"],
    )

    system_prompt = (
        "Tu es un assistant expert SQL SQLite. "
        f"Tu génères uniquement des requêtes SELECT sûres sur les relations autorisées: {', '.join(allowed_relations)}. "
        f"Relation à privilégier pour les questions métier: {preferred_relation}. "
        "Interdiction absolue d'écrire INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, PRAGMA ou plusieurs requêtes."
    )

    user_prompt = (
        "Transforme la question suivante en requête SQL SQLite en privilégiant la vue métier agrégée. "
        "Réponds en JSON strict avec les clés: sql, explanation. "
        "Ajoute LIMIT 200 si la question n'impose pas explicitement un autre volume. "
        f"Schéma: {json.dumps(schema, ensure_ascii=False)}\n\n"
        f"Question: {question}"
    )

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }

    data = _post_llm_chat_completion(provider=provider, api_key=api_key, payload=payload)
    content = _extract_message_content(data["choices"][0]["message"]["content"])
    return json.loads(content)


def answer_marriages_question(
    question: str,
    sql: str,
    rows: list[dict],
    api_key: str,
    provider: str = "Groq",
    model: str = "llama-3.3-70b-versatile",
) -> str:
    """Formule une réponse en français à partir des lignes SQL retournées."""
    if not api_key:
        raise ValueError(f"Clé API {provider} manquante.")

    preview_rows = rows[:30]
    system_prompt = (
        "Tu es un assistant d'analyse d'archives de mariages. "
        "Réponds uniquement à partir des résultats SQL fournis, en français, de manière concise. "
        "Si les données sont insuffisantes, dis-le clairement."
    )

    user_prompt = (
        f"Question utilisateur: {question}\n"
        f"Requête SQL exécutée: {sql}\n"
        f"Nombre total de lignes retournées: {len(rows)}\n"
        f"Aperçu des résultats: {json.dumps(preview_rows, ensure_ascii=False)}"
    )

    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    data = _post_llm_chat_completion(provider=provider, api_key=api_key, payload=payload)
    return _extract_message_content(data["choices"][0]["message"]["content"]).strip()


def split_text_into_marriages(
    text: str,
    api_key: str,
    provider: str = "Groq",
    model: str = "llama-3.3-70b-versatile",
) -> list[str]:
    """
    Demande au LLM de segmenter le texte en blocs distincts, un par mariage.
    Retourne une liste de chaînes, chacune correspondant à un acte.
    """
    if not api_key:
        return [text]

    system_prompt = (
        "Tu es un expert en lecture d'archives de registres d'état civil anciens. "
        "Une page de registre contient plusieurs actes de mariage séparés par des sauts de ligne ou des séparateurs visuels. "
        "Ta seule mission est de découper le texte fourni en actes individuels."
    )

    user_prompt = (
        "Le texte ci-dessous est issu d'une page de registre de mariages. "
        "Chaque paragraphe ou bloc séparé par une ligne vide correspond à un acte de mariage distinct.\n"
        "Découpe ce texte en blocs individuels et retourne-les en JSON strict sous la forme :\n"
        '{"mariages": ["texte du mariage 1", "texte du mariage 2", ...]}\n'
        "Ne modifie pas le texte, ne corrige pas les erreurs OCR, ne fusionne pas deux actes.\n"
        "Si tu ne peux pas distinguer plusieurs actes, retourne un seul élément dans la liste.\n\n"
        f"Texte à segmenter:\n{text}"
    )

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }

    try:
        data = _post_llm_chat_completion(provider=provider, api_key=api_key, payload=payload)
        content = _extract_message_content(data["choices"][0]["message"]["content"])
        parsed = json.loads(content)
        segments = parsed.get("mariages") or parsed.get("marriages") or parsed.get("actes")
        if isinstance(segments, list) and all(isinstance(s, str) for s in segments):
            return [s.strip() for s in segments if s.strip()]
    except Exception:
        pass

    # Fallback : découpe naïve sur lignes vides
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs if paragraphs else [text]


def extract_all_marriages_data(
    text: str,
    api_key: str,
    provider: str = "Groq",
    model: str = "llama-3.3-70b-versatile",
    paragraphs: list[str] | None = None,
) -> list[dict]:
    """
    Si `paragraphs` est fourni (détection faite en amont par l'OCR),
    extrait directement un acte par paragraphe sans appel LLM de segmentation.
    Sinon, demande au LLM de segmenter le texte puis extrait chaque segment.
    """
    if paragraphs and len(paragraphs) > 0:
        segments = paragraphs
    else:
        segments = split_text_into_marriages(
            text=text, api_key=api_key, provider=provider, model=model
        )

    results = []
    for segment in segments:
        extracted = extract_marriage_data(
            text=segment, api_key=api_key, provider=provider, model=model
        )
        if extracted:
            extracted["_paragraph_text"] = segment
            results.append(extracted)

    return results


def build_marriage_csv_row(extracted_data: dict | None, engine_name: str, source_name: str) -> dict:
    extracted_data = extracted_data or {}
    marie_1 = extracted_data.get("marie_1") or {}
    marie_2 = extracted_data.get("marie_2") or {}
    marie_1_pere = marie_1.get("pere") or {}
    marie_1_mere = marie_1.get("mere") or {}
    marie_2_pere = marie_2.get("pere") or {}
    marie_2_mere = marie_2.get("mere") or {}

    return {
        "source_name": source_name,
        "ocr_engine": engine_name,
        "lieu_acte": extracted_data.get("lieu_acte"),
        "date_acte": extracted_data.get("date_acte"),
        "marie_1_nom": marie_1.get("nom"),
        "marie_1_profession": marie_1.get("profession"),
        "marie_1_adresse": marie_1.get("adresse"),
        "marie_1_pere_nom": marie_1_pere.get("nom"),
        "marie_1_pere_profession": marie_1_pere.get("profession"),
        "marie_1_pere_adresse": marie_1_pere.get("adresse"),
        "marie_1_mere_nom": marie_1_mere.get("nom"),
        "marie_1_mere_profession": marie_1_mere.get("profession"),
        "marie_1_mere_adresse": marie_1_mere.get("adresse"),
        "marie_2_nom": marie_2.get("nom"),
        "marie_2_profession": marie_2.get("profession"),
        "marie_2_adresse": marie_2.get("adresse"),
        "marie_2_pere_nom": marie_2_pere.get("nom"),
        "marie_2_pere_profession": marie_2_pere.get("profession"),
        "marie_2_pere_adresse": marie_2_pere.get("adresse"),
        "marie_2_mere_nom": marie_2_mere.get("nom"),
        "marie_2_mere_profession": marie_2_mere.get("profession"),
        "marie_2_mere_adresse": marie_2_mere.get("adresse"),
    }


def build_marriage_csv_bytes(extracted_data: dict | None, engine_name: str, source_name: str) -> bytes:
    row = build_marriage_csv_row(extracted_data, engine_name=engine_name, source_name=source_name)
    dataframe = pd.DataFrame([row])
    return dataframe.to_csv(index=False).encode("utf-8-sig")


def semantic_correct_text(
    raw_text,
    marked_text,
    doubtful_words,
    api_key,
    provider="Groq",
    model="llama-3.3-70b-versatile",
):
    if not api_key:
        raise ValueError(f"Clé API {provider} manquante.")

    doubtful_preview = ", ".join(
        [f"{item['word']} ({item['confidence']:.2f})" for item in doubtful_words[:60]]
    )

    system_prompt = (
        "Tu es un correcteur OCR expert en correction sémantique de manuscrits. "
        "Ta mission: corriger les erreurs OCR sans changer le sens du texte."
    )

    user_prompt = (
        "Corrige le texte OCR ci-dessous.\n"
        "Contraintes: \n"
        "1) Préserve la structure des paragraphes et retours ligne.\n"
        "2) Corrige orthographe/grammaire/ponctuation de façon sémantique.\n"
        "3) Les mots marqués sous forme ⟦mot?⟧ sont douteux; si tu n'es pas certain, garde une balise douteuse.\n"
        "4) Réponds en JSON strict avec les clés: corrected_text, remaining_doubtful_words.\n\n"
        f"Mots douteux (confiance OCR): {doubtful_preview if doubtful_preview else 'Aucun'}\n\n"
        # "Texte OCR brut:\n"
        # f"{raw_text}\n\n"
        "Texte OCR balisé:\n"
        f"{marked_text}"
    )

    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }

    data = _post_llm_chat_completion(provider=provider, api_key=api_key, payload=payload)
    content = _extract_message_content(data["choices"][0]["message"]["content"])
    parsed = json.loads(content)

    corrected_text = parsed.get("corrected_text", "")
    remaining_doubtful_words = parsed.get("remaining_doubtful_words", [])

    return {
        "corrected_text": corrected_text,
        "remaining_doubtful_words": remaining_doubtful_words,
    }


def extract_marriage_data_with_groq(text: str, api_key: str, model="llama-3.3-70b-versatile"):
    return extract_marriage_data(text=text, api_key=api_key, provider="Groq", model=model)


def semantic_correct_text_with_groq(
    raw_text,
    marked_text,
    doubtful_words,
    api_key,
    model="llama-3.3-70b-versatile",
):
    return semantic_correct_text(
        raw_text=raw_text,
        marked_text=marked_text,
        doubtful_words=doubtful_words,
        api_key=api_key,
        provider="Groq",
        model=model,
    )
