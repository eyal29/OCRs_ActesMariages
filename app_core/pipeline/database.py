import sqlite3
from datetime import datetime
from pathlib import Path
import re
import unicodedata

import pandas as pd

# Toujours dans le dossier racine du projet, peu importe le répertoire de travail
DB_PATH = str(Path(__file__).resolve().parents[2] / "mariages.db")

_COLUMNS = [
    "source_name",
    "processed_at",
    "paragraph_index",
    "lieu_acte",
    "date_acte",
    "marie_1_nom",
    "marie_1_profession",
    "marie_1_adresse",
    "marie_1_pere_nom",
    "marie_1_pere_profession",
    "marie_1_pere_adresse",
    "marie_1_mere_nom",
    "marie_1_mere_profession",
    "marie_1_mere_adresse",
    "marie_2_nom",
    "marie_2_profession",
    "marie_2_adresse",
    "marie_2_pere_nom",
    "marie_2_pere_profession",
    "marie_2_pere_adresse",
    "marie_2_mere_nom",
    "marie_2_mere_profession",
    "marie_2_mere_adresse",
]

_CHATBOT_ALLOWED_RELATIONS = [
    "actes_mariage",
    "v_chatbot_actes",
    "documents",
    "actes",
    "personnes",
    "acte_personnes",
]


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _normalize_text(value: str | None) -> str:
    if not value or not isinstance(value, str):
        return ""
    text = value.strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_date_to_iso(value: str | None) -> str | None:
    if not value or not isinstance(value, str):
        return None
    text = value.strip()

    # yyyy-mm-dd ou yyyy/mm/dd
    match = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", text)
    if match:
        year, month, day = match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

    # dd-mm-yyyy ou dd/mm/yyyy
    match = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$", text)
    if match:
        day, month, year = match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

    return None


def _create_normalized_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS actes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            paragraph_index INTEGER NOT NULL,
            processed_at TEXT,
            lieu_raw TEXT,
            lieu_norm TEXT,
            date_raw TEXT,
            date_iso TEXT,
            legacy_row_id INTEGER,
            UNIQUE(document_id, paragraph_index),
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS personnes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nom_raw TEXT,
            nom_norm TEXT,
            UNIQUE(nom_raw, nom_norm)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS acte_personnes (
            acte_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            personne_id INTEGER,
            profession TEXT,
            adresse TEXT,
            PRIMARY KEY (acte_id, role),
            FOREIGN KEY (acte_id) REFERENCES actes(id),
            FOREIGN KEY (personne_id) REFERENCES personnes(id)
        )
        """
    )

    conn.execute("CREATE INDEX IF NOT EXISTS idx_actes_date_iso ON actes(date_iso)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_actes_lieu_norm ON actes(lieu_norm)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_personnes_nom_norm ON personnes(nom_norm)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_acte_personnes_role ON acte_personnes(role)")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS db_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )


def _create_chatbot_view(conn: sqlite3.Connection) -> None:
    conn.execute("DROP VIEW IF EXISTS v_chatbot_actes")
    conn.execute(
        """
        CREATE VIEW v_chatbot_actes AS
        SELECT
            a.id AS acte_id,
            d.source_name,
            a.processed_at,
            a.paragraph_index,
            a.lieu_raw AS lieu_acte,
            a.lieu_norm AS lieu_acte_norm,
            a.date_raw AS date_acte,
            a.date_iso,
            MAX(CASE WHEN ap.role = 'marie_1' THEN p.nom_raw END) AS marie_1_nom,
            MAX(CASE WHEN ap.role = 'marie_1' THEN ap.profession END) AS marie_1_profession,
            MAX(CASE WHEN ap.role = 'marie_1' THEN ap.adresse END) AS marie_1_adresse,
            MAX(CASE WHEN ap.role = 'marie_1_pere' THEN p.nom_raw END) AS marie_1_pere_nom,
            MAX(CASE WHEN ap.role = 'marie_1_pere' THEN ap.profession END) AS marie_1_pere_profession,
            MAX(CASE WHEN ap.role = 'marie_1_pere' THEN ap.adresse END) AS marie_1_pere_adresse,
            MAX(CASE WHEN ap.role = 'marie_1_mere' THEN p.nom_raw END) AS marie_1_mere_nom,
            MAX(CASE WHEN ap.role = 'marie_1_mere' THEN ap.profession END) AS marie_1_mere_profession,
            MAX(CASE WHEN ap.role = 'marie_1_mere' THEN ap.adresse END) AS marie_1_mere_adresse,
            MAX(CASE WHEN ap.role = 'marie_2' THEN p.nom_raw END) AS marie_2_nom,
            MAX(CASE WHEN ap.role = 'marie_2' THEN ap.profession END) AS marie_2_profession,
            MAX(CASE WHEN ap.role = 'marie_2' THEN ap.adresse END) AS marie_2_adresse,
            MAX(CASE WHEN ap.role = 'marie_2_pere' THEN p.nom_raw END) AS marie_2_pere_nom,
            MAX(CASE WHEN ap.role = 'marie_2_pere' THEN ap.profession END) AS marie_2_pere_profession,
            MAX(CASE WHEN ap.role = 'marie_2_pere' THEN ap.adresse END) AS marie_2_pere_adresse,
            MAX(CASE WHEN ap.role = 'marie_2_mere' THEN p.nom_raw END) AS marie_2_mere_nom,
            MAX(CASE WHEN ap.role = 'marie_2_mere' THEN ap.profession END) AS marie_2_mere_profession,
            MAX(CASE WHEN ap.role = 'marie_2_mere' THEN ap.adresse END) AS marie_2_mere_adresse
        FROM actes a
        JOIN documents d ON d.id = a.document_id
        LEFT JOIN acte_personnes ap ON ap.acte_id = a.id
        LEFT JOIN personnes p ON p.id = ap.personne_id
        GROUP BY a.id, d.source_name, a.processed_at, a.paragraph_index, a.lieu_raw, a.lieu_norm, a.date_raw, a.date_iso
        """
    )


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO db_meta(key, value) VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


def _get_meta(conn: sqlite3.Connection, key: str, default: str = "0") -> str:
    row = conn.execute("SELECT value FROM db_meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def _upsert_document(conn: sqlite3.Connection, source_name: str) -> int:
    created_at = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        """
        INSERT INTO documents(source_name, created_at)
        VALUES(?, ?)
        ON CONFLICT(source_name) DO NOTHING
        """,
        (source_name, created_at),
    )
    row = conn.execute(
        "SELECT id FROM documents WHERE source_name = ?",
        (source_name,),
    ).fetchone()
    return int(row[0])


def _upsert_person(conn: sqlite3.Connection, name: str | None) -> int | None:
    if not name or not isinstance(name, str) or not name.strip():
        return None
    nom_raw = name.strip()
    nom_norm = _normalize_text(nom_raw)
    conn.execute(
        """
        INSERT INTO personnes(nom_raw, nom_norm)
        VALUES(?, ?)
        ON CONFLICT(nom_raw, nom_norm) DO NOTHING
        """,
        (nom_raw, nom_norm),
    )
    row = conn.execute(
        "SELECT id FROM personnes WHERE nom_raw = ? AND nom_norm = ?",
        (nom_raw, nom_norm),
    ).fetchone()
    return int(row[0]) if row else None


def _upsert_normalized_from_flat_row(conn: sqlite3.Connection, row: dict) -> None:
    source_name = (row.get("source_name") or "").strip()
    if not source_name:
        return

    document_id = _upsert_document(conn, source_name)
    paragraph_index = int(row.get("paragraph_index") or 0)
    processed_at = row.get("processed_at")
    lieu_raw = row.get("lieu_acte")
    date_raw = row.get("date_acte")
    lieu_norm = _normalize_text(lieu_raw)
    date_iso = _normalize_date_to_iso(date_raw)

    conn.execute(
        """
        INSERT INTO actes(
            document_id, paragraph_index, processed_at,
            lieu_raw, lieu_norm, date_raw, date_iso, legacy_row_id
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(document_id, paragraph_index) DO UPDATE SET
            processed_at = excluded.processed_at,
            lieu_raw = excluded.lieu_raw,
            lieu_norm = excluded.lieu_norm,
            date_raw = excluded.date_raw,
            date_iso = excluded.date_iso,
            legacy_row_id = excluded.legacy_row_id
        """,
        (
            document_id,
            paragraph_index,
            processed_at,
            lieu_raw,
            lieu_norm,
            date_raw,
            date_iso,
            row.get("id"),
        ),
    )

    acte_row = conn.execute(
        "SELECT id FROM actes WHERE document_id = ? AND paragraph_index = ?",
        (document_id, paragraph_index),
    ).fetchone()
    if not acte_row:
        return
    acte_id = int(acte_row[0])

    role_mapping = {
        "marie_1": (row.get("marie_1_nom"), row.get("marie_1_profession"), row.get("marie_1_adresse")),
        "marie_1_pere": (row.get("marie_1_pere_nom"), row.get("marie_1_pere_profession"), row.get("marie_1_pere_adresse")),
        "marie_1_mere": (row.get("marie_1_mere_nom"), row.get("marie_1_mere_profession"), row.get("marie_1_mere_adresse")),
        "marie_2": (row.get("marie_2_nom"), row.get("marie_2_profession"), row.get("marie_2_adresse")),
        "marie_2_pere": (row.get("marie_2_pere_nom"), row.get("marie_2_pere_profession"), row.get("marie_2_pere_adresse")),
        "marie_2_mere": (row.get("marie_2_mere_nom"), row.get("marie_2_mere_profession"), row.get("marie_2_mere_adresse")),
    }

    for role, (name, profession, adresse) in role_mapping.items():
        if not any([name, profession, adresse]):
            conn.execute(
                "DELETE FROM acte_personnes WHERE acte_id = ? AND role = ?",
                (acte_id, role),
            )
            continue

        personne_id = _upsert_person(conn, name)
        conn.execute(
            """
            INSERT INTO acte_personnes(acte_id, role, personne_id, profession, adresse)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(acte_id, role) DO UPDATE SET
                personne_id = excluded.personne_id,
                profession = excluded.profession,
                adresse = excluded.adresse
            """,
            (acte_id, role, personne_id, profession, adresse),
        )


def _sync_legacy_incremental(conn: sqlite3.Connection) -> None:
    last_synced_id = int(_get_meta(conn, "legacy_last_synced_id", "0") or "0")
    rows = conn.execute(
        "SELECT * FROM actes_mariage WHERE id > ? ORDER BY id ASC",
        (last_synced_id,),
    ).fetchall()

    max_seen_id = last_synced_id
    for row in rows:
        row_dict = dict(row)
        _upsert_normalized_from_flat_row(conn, row_dict)
        max_seen_id = max(max_seen_id, int(row_dict.get("id") or 0))

    if max_seen_id != last_synced_id:
        _set_meta(conn, "legacy_last_synced_id", str(max_seen_id))


def _rebuild_normalized_from_legacy(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM acte_personnes")
    conn.execute("DELETE FROM actes")
    conn.execute("DELETE FROM personnes")
    conn.execute("DELETE FROM documents")
    conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('documents', 'actes', 'personnes')")

    rows = conn.execute("SELECT * FROM actes_mariage ORDER BY id ASC").fetchall()
    max_seen_id = 0
    for row in rows:
        row_dict = dict(row)
        _upsert_normalized_from_flat_row(conn, row_dict)
        max_seen_id = max(max_seen_id, int(row_dict.get("id") or 0))
    _set_meta(conn, "legacy_last_synced_id", str(max_seen_id))


def init_db() -> None:
    """Crée la base de données et la table si elles n'existent pas. Migration douce incluse."""
    with _get_conn() as conn:
        cols_def = ", ".join(
            ["id INTEGER PRIMARY KEY AUTOINCREMENT"]
            + [f"{col} TEXT" for col in _COLUMNS]
        )
        conn.execute(f"CREATE TABLE IF NOT EXISTS actes_mariage ({cols_def})")
        # Migration douce : ajouter les colonnes manquantes si la table existait déjà
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(actes_mariage)")}
        for col in _COLUMNS:
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE actes_mariage ADD COLUMN {col} TEXT")

        _create_normalized_schema(conn)
        _create_chatbot_view(conn)
        _sync_legacy_incremental(conn)
        conn.commit()


def _build_row_values(source_name: str, extracted_data: dict, paragraph_index: int) -> list:
    extracted_data = extracted_data or {}
    marie_1 = extracted_data.get("marie_1") or {}
    marie_2 = extracted_data.get("marie_2") or {}
    marie_1_pere = marie_1.get("pere") or {}
    marie_1_mere = marie_1.get("mere") or {}
    marie_2_pere = marie_2.get("pere") or {}
    marie_2_mere = marie_2.get("mere") or {}

    values = {
        "source_name": source_name,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "paragraph_index": str(paragraph_index),
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
    return [values[col] for col in _COLUMNS]


def _count_filled_fields(extracted_data: dict) -> int:
    """Compte le nombre de champs remplis (non-None et non-vides) dans une extraction."""
    if not extracted_data:
        return 0
    count = 0
    marie_1 = extracted_data.get("marie_1") or {}
    marie_2 = extracted_data.get("marie_2") or {}
    marie_1_pere = marie_1.get("pere") or {}
    marie_1_mere = marie_1.get("mere") or {}
    marie_2_pere = marie_2.get("pere") or {}
    marie_2_mere = marie_2.get("mere") or {}

    fields = [
        extracted_data.get("lieu_acte"),
        extracted_data.get("date_acte"),
        marie_1.get("nom"),
        marie_1.get("profession"),
        marie_1.get("adresse"),
        marie_1_pere.get("nom"),
        marie_1_pere.get("profession"),
        marie_1_pere.get("adresse"),
        marie_1_mere.get("nom"),
        marie_1_mere.get("profession"),
        marie_1_mere.get("adresse"),
        marie_2.get("nom"),
        marie_2.get("profession"),
        marie_2.get("adresse"),
        marie_2_pere.get("nom"),
        marie_2_pere.get("profession"),
        marie_2_pere.get("adresse"),
        marie_2_mere.get("nom"),
        marie_2_mere.get("profession"),
        marie_2_mere.get("adresse"),
    ]
    
    for field in fields:
        if field and isinstance(field, str) and field.strip():
            count += 1
    return count


def _get_existing_record(source_name: str, paragraph_index: int) -> dict | None:
    """Récupère l'enregistrement existant pour (source_name, paragraph_index), si présent."""
    init_db()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM actes_mariage WHERE source_name = ? AND paragraph_index = ?",
            (source_name, str(paragraph_index)),
        ).fetchone()
    return dict(row) if row else None


def save_extraction(source_name: str, extracted_data: dict | None, paragraph_index: int = 0) -> int:
    """
    Insère ou met à jour une ligne dans actes_mariage.
    Si la clé (source_name, paragraph_index) existe déjà et que les nouvelles données
    sont plus complètes, remplace l'ancienne. Sinon, insère une nouvelle ligne.
    Retourne l'id de la ligne insérée/modifiée.
    """
    init_db()
    extracted_data = extracted_data or {}
    existing = _get_existing_record(source_name, paragraph_index)
    
    new_completeness = _count_filled_fields(extracted_data)
    
    # Si une ligne existe, comparer la complétude
    if existing:
        existing_completeness = _count_filled_fields({
            "lieu_acte": existing.get("lieu_acte"),
            "date_acte": existing.get("date_acte"),
            "marie_1": {
                "nom": existing.get("marie_1_nom"),
                "profession": existing.get("marie_1_profession"),
                "adresse": existing.get("marie_1_adresse"),
                "pere": {
                    "nom": existing.get("marie_1_pere_nom"),
                    "profession": existing.get("marie_1_pere_profession"),
                    "adresse": existing.get("marie_1_pere_adresse"),
                },
                "mere": {
                    "nom": existing.get("marie_1_mere_nom"),
                    "profession": existing.get("marie_1_mere_profession"),
                    "adresse": existing.get("marie_1_mere_adresse"),
                },
            },
            "marie_2": {
                "nom": existing.get("marie_2_nom"),
                "profession": existing.get("marie_2_profession"),
                "adresse": existing.get("marie_2_adresse"),
                "pere": {
                    "nom": existing.get("marie_2_pere_nom"),
                    "profession": existing.get("marie_2_pere_profession"),
                    "adresse": existing.get("marie_2_pere_adresse"),
                },
                "mere": {
                    "nom": existing.get("marie_2_mere_nom"),
                    "profession": existing.get("marie_2_mere_profession"),
                    "adresse": existing.get("marie_2_mere_adresse"),
                },
            },
        })
        
        # Si les nouvelles données ne sont pas plus complètes, ne rien faire
        if new_completeness <= existing_completeness:
            return existing["id"]
        
        # Sinon, faire un UPDATE
        placeholders = ", ".join([f"{col} = ?" for col in _COLUMNS])
        row_values = _build_row_values(source_name, extracted_data, paragraph_index)
        row_values.append(existing["id"])
        
        with _get_conn() as conn:
            conn.execute(
                f"UPDATE actes_mariage SET {placeholders} WHERE id = ?",
                row_values,
            )
            refreshed_row = conn.execute(
                "SELECT * FROM actes_mariage WHERE id = ?",
                (existing["id"],),
            ).fetchone()
            if refreshed_row:
                _upsert_normalized_from_flat_row(conn, dict(refreshed_row))
            conn.commit()
        return existing["id"]
    
    # Sinon, insérer une nouvelle ligne
    placeholders = ", ".join(["?"] * len(_COLUMNS))
    col_names = ", ".join(_COLUMNS)
    row_values = _build_row_values(source_name, extracted_data, paragraph_index)

    with _get_conn() as conn:
        cursor = conn.execute(
            f"INSERT INTO actes_mariage ({col_names}) VALUES ({placeholders})",
            row_values,
        )
        inserted_id = cursor.lastrowid
        inserted_row = conn.execute(
            "SELECT * FROM actes_mariage WHERE id = ?",
            (inserted_id,),
        ).fetchone()
        if inserted_row:
            _upsert_normalized_from_flat_row(conn, dict(inserted_row))
        conn.commit()
        return inserted_id


def save_all_extractions(source_name: str, extractions: list[dict]) -> list[int]:
    """
    Insère ou met à jour les lignes pour chaque mariage extrait d'une même image source.
    Pour chaque extraction, si la clé (source_name, paragraph_index) existe déjà et que
    les nouvelles données sont plus complètes, remplace l'ancienne. Sinon, insère une nouvelle.
    Retourne la liste des ids insérés/modifiés.
    """
    init_db()
    inserted_ids = []

    for idx, extracted_data in enumerate(extractions):
        row_id = save_extraction(source_name, extracted_data, paragraph_index=idx)
        inserted_ids.append(row_id)

    return inserted_ids


def load_all_records() -> list[dict]:
    """Retourne toutes les lignes de actes_mariage sous forme de liste de dicts."""
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM actes_mariage ORDER BY id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_db_path() -> str:
    """Retourne le chemin absolu de la base SQLite."""
    return DB_PATH


def get_table_schema() -> dict:
    """Retourne le schéma utile au chatbot (vue agrégée + tables normalisées)."""
    init_db()
    with _get_conn() as conn:
        view_rows = conn.execute("PRAGMA table_info(v_chatbot_actes)").fetchall()
        actes_rows = conn.execute("PRAGMA table_info(actes)").fetchall()
        personnes_rows = conn.execute("PRAGMA table_info(personnes)").fetchall()
        legacy_rows = conn.execute("PRAGMA table_info(actes_mariage)").fetchall()

    def _to_columns(rows):
        return [
            {
                "name": row[1],
                "type": row[2],
                "notnull": bool(row[3]),
                "pk": bool(row[5]),
            }
            for row in rows
        ]

    return {
        "preferred_relation": "v_chatbot_actes",
        "allowed_relations": _CHATBOT_ALLOWED_RELATIONS,
        "relations": {
            "v_chatbot_actes": _to_columns(view_rows),
            "actes": _to_columns(actes_rows),
            "personnes": _to_columns(personnes_rows),
            "actes_mariage": _to_columns(legacy_rows),
        },
    }


def get_database_overview() -> dict:
    """Retourne quelques métriques utiles pour la vue chatbot."""
    init_db()
    with _get_conn() as conn:
        total_rows = conn.execute("SELECT COUNT(*) FROM v_chatbot_actes").fetchone()[0]
        distinct_sources = conn.execute(
            "SELECT COUNT(DISTINCT source_name) FROM documents"
        ).fetchone()[0]
        latest_processed_at = conn.execute(
            "SELECT MAX(processed_at) FROM v_chatbot_actes"
        ).fetchone()[0]
    return {
        "total_rows": int(total_rows or 0),
        "distinct_sources": int(distinct_sources or 0),
        "latest_processed_at": latest_processed_at,
    }


def run_readonly_query(query: str) -> pd.DataFrame:
    """Exécute uniquement une requête SELECT en lecture seule sur la base."""
    init_db()
    sql = (query or "").strip().rstrip(";")
    lowered = sql.lower()

    if not lowered.startswith("select"):
        raise ValueError("Seules les requêtes SELECT sont autorisées.")
    forbidden = [
        " insert ", " update ", " delete ", " drop ", " alter ", " create ",
        " replace ", " truncate ", " attach ", " detach ", " pragma ",
    ]
    padded = f" {lowered} "
    if any(token in padded for token in forbidden):
        raise ValueError("La requête contient une opération non autorisée.")

    if not any(relation in lowered for relation in _CHATBOT_ALLOWED_RELATIONS):
        raise ValueError(
            "La requête doit cibler une relation autorisée: "
            + ", ".join(_CHATBOT_ALLOWED_RELATIONS)
        )

    with _get_conn() as conn:
        dataframe = pd.read_sql_query(sql, conn)
    return dataframe


def clean_duplicates(keep_strategy: str = "most_complete") -> dict:
    """
    Nettoie les doublons dans la base (plusieurs lignes avec même source_name + paragraph_index).
    
    Stratégies:
    - "most_complete": garde l'enregistrement avec le plus de champs remplis
    - "most_recent": garde l'enregistrement le plus récent (processed_at)
    - "first": garde le premier (id le plus bas)
    
    Retourne un dict avec stats de nettoyage.
    """
    init_db()
    
    deleted_count = 0
    kept_records = []
    
    # Trouver tous les groupes de doublons
    with _get_conn() as conn:
        duplicates = conn.execute("""
            SELECT source_name, paragraph_index, COUNT(*) as count
            FROM actes_mariage
            GROUP BY source_name, paragraph_index
            HAVING count > 1
        """).fetchall()
        
        duplicate_count = len(duplicates)
        
        for dup in duplicates:
            source_name, paragraph_index, count = dup[0], str(dup[1]), dup[2]
            
            # Récupérer tous les enregistrements pour ce groupe
            rows = conn.execute("""
                SELECT * FROM actes_mariage
                WHERE source_name = ? AND paragraph_index = ?
                ORDER BY id ASC
            """, (source_name, paragraph_index)).fetchall()
            
            rows_dict = [dict(row) for row in rows]
            
            if not rows_dict:
                continue
            
            # Décider quel enregistrement garder
            if keep_strategy == "most_complete":
                kept = max(rows_dict, key=lambda r: _count_filled_fields(_extract_from_row(r)))
            elif keep_strategy == "most_recent":
                kept = max(rows_dict, key=lambda r: r["processed_at"] or "")
            else:  # "first"
                kept = rows_dict[0]
            
            kept_id = kept["id"]
            kept_records.append(kept_id)
            
            # Supprimer tous les autres
            for row in rows_dict:
                if row["id"] != kept_id:
                    conn.execute("DELETE FROM actes_mariage WHERE id = ?", (row["id"],))
                    deleted_count += 1

        _rebuild_normalized_from_legacy(conn)
        _create_chatbot_view(conn)
        
        conn.commit()
    
    return {
        "duplicate_groups_found": duplicate_count,
        "records_deleted": deleted_count,
        "records_kept": len(kept_records),
        "strategy": keep_strategy,
    }


def _extract_from_row(row: dict) -> dict:
    """Reconstruit une structure d'extraction à partir d'une ligne de la BDD."""
    return {
        "lieu_acte": row.get("lieu_acte"),
        "date_acte": row.get("date_acte"),
        "marie_1": {
            "nom": row.get("marie_1_nom"),
            "profession": row.get("marie_1_profession"),
            "adresse": row.get("marie_1_adresse"),
            "pere": {
                "nom": row.get("marie_1_pere_nom"),
                "profession": row.get("marie_1_pere_profession"),
                "adresse": row.get("marie_1_pere_adresse"),
            },
            "mere": {
                "nom": row.get("marie_1_mere_nom"),
                "profession": row.get("marie_1_mere_profession"),
                "adresse": row.get("marie_1_mere_adresse"),
            },
        },
        "marie_2": {
            "nom": row.get("marie_2_nom"),
            "profession": row.get("marie_2_profession"),
            "adresse": row.get("marie_2_adresse"),
            "pere": {
                "nom": row.get("marie_2_pere_nom"),
                "profession": row.get("marie_2_pere_profession"),
                "adresse": row.get("marie_2_pere_adresse"),
            },
            "mere": {
                "nom": row.get("marie_2_mere_nom"),
                "profession": row.get("marie_2_mere_profession"),
                "adresse": row.get("marie_2_mere_adresse"),
            },
        },
    }
