import pandas as pd
import streamlit as st
from difflib import SequenceMatcher

from app_core.common.ocr_utils import clean_text
from app_core.config import OCR_TECHNIQUES
from app_core.pipeline.llm_client import build_marriage_csv_row
from app_core.ui.metrics import (
    compute_cer,
    compute_lexical_score,
    compute_ocr_quality_metrics,
    compute_wer,
    extract_confidences,
)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _confidence_level_label(avg_conf: float | None) -> str:
    if avg_conf is None:
        return "N/A"
    if avg_conf >= 0.85:
        return "Élevée"
    if avg_conf >= 0.65:
        return "Moyenne"
    return "Faible"


def _safe_text(t) -> str:
    if isinstance(t, str):
        return t
    if isinstance(t, dict):
        return t.get("text") or t.get("raw") or ""
    return ""


def _flatten_structured_fields(data: dict, parent_key: str = "") -> list[dict]:
    rows = []
    for key, value in (data or {}).items():
        if key == "_paragraph_text":
            continue
        field_path = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            rows.extend(_flatten_structured_fields(value, field_path))
            continue
        filled = bool(isinstance(value, str) and value.strip())
        rows.append({
            "Champ": field_path,
            "Valeur": value if value is not None else "",
            "Statut": "Renseigné" if filled else "Manquant",
        })
    return rows


def _compute_field_confidence(extracted_data: dict, paragraph_text: str) -> list[dict]:
    """Estime la confiance de chaque champ basée sur le texte source."""
    field_rows = _flatten_structured_fields(extracted_data)
    lexical_score = compute_lexical_score(paragraph_text)["lexical_pct"] / 100

    for row in field_rows:
        if row["Statut"] == "Manquant":
            row["Confiance %"] = 0
            row["Note"] = "❓"
        else:
            length_boost = min(20, len(row["Valeur"].split()) * 5)
            conf = round(min(100, lexical_score * 100 + length_boost), 1)
            row["Confiance %"] = conf
            row["Note"] = "✅" if conf >= 80 else ("⚠️" if conf >= 60 else "❌")

    return field_rows


def _build_comparison_df(results: dict) -> pd.DataFrame:
    """Construit le DataFrame de comparaison des moteurs (calcul centralisé)."""
    rows = []
    for technique, ocr_data in results.items():
        stats = (ocr_data.get("analysis") or {}).get("stats") or {}
        raw_text = (ocr_data.get("text") or {}).get("raw") or ""

        word_count = int(stats.get("word_count") or 0)
        doubtful = int(stats.get("doubtful_word_count") or 0)
        avg_conf = stats.get("average_confidence")
        doubtful_ratio = (doubtful / word_count * 100) if word_count else 0.0

        quality = compute_ocr_quality_metrics(
            ocr_data,
            word_count,
            doubtful_ratio,
            len(set(raw_text.lower().split())),
            avg_conf,
        )
        rows.append({
            "Moteur": technique,
            "Score lexical %": quality["lexical_score"],
            "% douteux": round(doubtful_ratio, 2),
            "Conf. moteur %": round(avg_conf * 100, 2) if avg_conf else 0,
            "Mots": word_count,
        })

    return pd.DataFrame(rows).sort_values("Score lexical %", ascending=False)


# ─────────────────────────────────────────────
# ANALYSE / SCORES
# ─────────────────────────────────────────────

def _compute_pipeline_score(results: dict, llm_outputs: dict) -> dict:
    """Score qualité global du pipeline (0-100)."""
    if not results:
        return {"score": 0.0, "insights": ["Aucun résultat OCR"]}

    scores = []
    for technique, ocr_data in results.items():
        stats = (ocr_data.get("analysis") or {}).get("stats") or {}
        word_count = int(stats.get("word_count") or 0)
        doubtful = int(stats.get("doubtful_word_count") or 0)
        avg_conf = float(stats.get("average_confidence") or 0)
        doubtful_ratio = (doubtful / word_count * 100) if word_count else 0

        raw_text = clean_text((ocr_data.get("text") or {}).get("raw") or "")
        lexical_pct = compute_lexical_score(raw_text)["lexical_pct"]

        score = lexical_pct * 0.4 + avg_conf * 100 * 0.3 + (100 - doubtful_ratio) * 0.3
        if llm_outputs.get(technique):
            score *= 1.1
        scores.append(score)

    avg, mn, mx = sum(scores) / len(scores), min(scores), max(scores)

    insights = []
    if avg >= 80:
        insights.append("✅ Pipeline de très bonne qualité")
    elif avg >= 65:
        insights.append("✔️ Qualité acceptable — quelques vérifications recommandées")
    else:
        insights.append("⚠️ Qualité faible — vérification manuelle recommandée")

    if mx - mn > 20:
        insights.append(f"📊 Variance haute ({mn:.0f}-{mx:.0f}%) — choix moteur critique")
    else:
        insights.append(f"🎯 Moteurs homogènes ({mn:.0f}-{mx:.0f}%)")

    return {"score": round(min(avg, 100), 1), "min": round(mn, 1), "max": round(mx, 1), "insights": insights}


def _analyze_consensus(results: dict, final_texts: dict) -> dict:
    """Mesure l'accord entre les moteurs sur la longueur du texte produit."""
    if len(results) <= 1:
        return {"consensus": "N/A", "insights": ["Un seul moteur"]}

    texts = [_safe_text(t) for t in final_texts.values()]
    lengths = [len(t.split()) for t in texts if t]
    if not lengths:
        return {"consensus": 0.0, "insights": ["Aucun texte final"]}

    avg = sum(lengths) / len(lengths)
    std = (sum((l - avg) ** 2 for l in lengths) / len(lengths)) ** 0.5
    consensus = max(0, 100 - (std / avg * 100) * 10) if avg > 0 else 0

    insight = (
        "🎯 Bonne cohérence inter-moteurs" if consensus > 85
        else "✔️ Cohérence acceptable" if consensus > 70
        else "⚠️ Moteurs divergent — vérification cruciale"
    )
    return {"consensus": round(consensus, 1), "avg_words": round(avg), "insights": [insight]}


def _get_recommendations(results: dict, pipeline_score: dict, consensus: dict) -> list[str]:
    recs = []
    score = float(pipeline_score.get("score") or 0)
    cons = float(consensus.get("consensus") or 100)

    if score < 60:
        recs.append("🔍 Score faible → **vérifier manuellement les champs critiques** (dates, noms)")
    elif score < 75:
        recs.append("📋 Score moyen → **vérifier les champs extraits** avant insertion BDD")

    if cons < 70:
        recs.append("🤔 Moteurs divergent → **choisir un moteur dominant** ou fusionner manuellement")

    if float(pipeline_score.get("max") or 0) - float(pipeline_score.get("min") or 0) > 25:
        recs.append("⚡ Grande variance → **fusion par vote de champ recommandée**")

    return recs or ["✅ Pipeline stable — confiance élevée autorisée"]


# ─────────────────────────────────────────────
# RENDU : ACTES DÉTECTÉS
# ─────────────────────────────────────────────

def render_structured_extraction(file_base_name: str) -> None:
    structured_extraction = st.session_state.get("ocr_structured_extraction") or []
    source_engine = st.session_state.get("ocr_structured_source_engine") or "DocTR"

    if not structured_extraction:
        st.info("Aucun acte détecté. Lance l'OCR avec une clé API pour activer l'extraction structurée.")
        return

    rows = [build_marriage_csv_row(m, engine_name=source_engine, source_name=file_base_name)
            for m in structured_extraction]
    recap_df = pd.DataFrame(rows)

    st.success(f"✅ {len(structured_extraction)} acte(s) de mariage enregistré(s) dans la base de données.")
    st.dataframe(recap_df, width="stretch", hide_index=True)
    st.download_button(
        label=f"⬇️ Télécharger les {len(structured_extraction)} actes (CSV)",
        data=recap_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{file_base_name}_actes_detectes.csv",
        mime="text/csv",
        key="dl_structured_all",
    )
    st.divider()

    for idx, marriage in enumerate(structured_extraction, start=1):
        paragraph_text = marriage.get("_paragraph_text")
        display_marriage = {k: v for k, v in marriage.items() if k != "_paragraph_text"}
        label = (
            f"Acte #{idx} — {marriage.get('date_acte') or 'date inconnue'} | "
            f"{marriage.get('marie_1', {}).get('nom') or '?'} & "
            f"{marriage.get('marie_2', {}).get('nom') or '?'}"
        )
        with st.expander(label):
            col_json, col_text = st.columns(2, gap="medium")
            with col_json:
                st.markdown("**Données extraites**")
                st.json(display_marriage, expanded=2)
            with col_text:
                if paragraph_text:
                    st.markdown("**Texte source (paragraphe)**")
                    st.text_area("", value=paragraph_text, height=220, disabled=True,
                                 label_visibility="collapsed", key=f"structured_para_{idx}")


# ─────────────────────────────────────────────
# RENDU : RÉSULTATS PAR MOTEUR
# ─────────────────────────────────────────────

def render_engine_result_tab(technique: str, ocr_data: dict) -> None:
    """Vue OCR par moteur : texte brut + métriques clés."""
    st.header("🔬 Détails moteur")
    st.caption("💡 Pour la comparaison inter-moteurs, voir l'onglet **Comparaisons**.")
    st.divider()

    raw_text = clean_text(ocr_data["text"]["raw"])
    stats = ocr_data.get("analysis", {}).get("stats", {})
    doubtful_words = ocr_data["analysis"]["doubtful_words"]

    word_count = int(stats.get("word_count") or 0)
    doubtful_ratio = (len(doubtful_words) / word_count * 100) if word_count else 0.0
    avg_conf = stats.get("average_confidence")
    quality = compute_ocr_quality_metrics(
        ocr_data, word_count, doubtful_ratio,
        len(set(raw_text.lower().split())), avg_conf,
    )

    tab_text, tab_analysis = st.tabs(["📝 Texte extrait", "🧩 Analyse OCR"])

    with tab_text:
        st.text_area("", raw_text, height=360, key=f"ocr_raw_{technique}",
                     disabled=True, label_visibility="collapsed")

    with tab_analysis:
        col1, col2, col3 = st.columns(3)
        col1.metric("Score lexical", f"{quality['lexical_score']}%")
        col2.metric("% mots douteux", f"{doubtful_ratio:.1f}%")
        col3.metric("Conf. moteur", f"{round(avg_conf * 100, 1) if avg_conf else 0}%")

        if doubtful_words:
            with st.expander(f"⚠️ Mots douteux ({len(doubtful_words)})"):
                st.dataframe(doubtful_words[:200], width="stretch")
        else:
            st.success("✅ Aucun mot sous le seuil de confiance.")


# ─────────────────────────────────────────────
# RENDU : COMPARAISONS
# ─────────────────────────────────────────────

def render_ground_truth_tab(results: dict) -> None:
    st.markdown("Collez la **vérité terrain** pour calculer CER et WER.")
    reference_text = st.text_area("Texte de référence", height=180)

    if not reference_text.strip():
        st.info("Ajoutez un texte de référence.")
        return
    if len(reference_text.strip()) < 30:
        st.warning("Texte très court — CER/WER peuvent être instables.")

    rows = []
    for technique, ocr_data in results.items():
        raw_text = ocr_data["text"]["raw"]
        cer = compute_cer(reference_text, raw_text)
        wer = compute_wer(reference_text, raw_text)
        rows.append({
            "Moteur": technique,
            "CER %": round(cer * 100, 2),
            "WER %": round(wer * 100, 2),
            "Précision char %": round((1 - cer) * 100, 2),
            "Précision mots %": round((1 - wer) * 100, 2),
        })

    df = pd.DataFrame(rows).sort_values("CER %")
    st.dataframe(df, width="stretch")
    if not df.empty:
        best = df.iloc[0]
        st.success(f"Meilleur moteur : **{best['Moteur']}** (CER {best['CER %']}%)")
    st.bar_chart(df.set_index("Moteur")[["CER %", "WER %"]])


def render_visual_comparison(results: dict) -> None:
    """Comparaison multi-moteurs : tableau + graphique unique + CER/WER."""
    df = _build_comparison_df(results)

    # Badge spellchecker
    first_q = compute_ocr_quality_metrics(next(iter(results.values())), 1, 0.0, 1, None)
    if first_q.get("spellchecker_used"):
        st.caption("✅ Score lexical calculé avec **pyspellchecker** (dictionnaire FR complet)")
    else:
        st.caption("⚠️ Dictionnaire embarqué — installez `pyspellchecker` pour plus de précision")

    tab_metrics, tab_cer = st.tabs(["📋 Métriques & Graphiques", "🎯 CER / WER"])

    with tab_metrics:
        # Tableau
        st.dataframe(df, width="stretch")
        if not df.empty:
            best = df.iloc[0]
            st.success(f"Meilleur moteur : **{best['Moteur']}** (Score lexical {best['Score lexical %']}%)")
        st.caption("Score lexical = qualité réelle du texte")

        # Graphique comparatif unique (qualité + quantité côte à côte)
        col_q, col_n = st.columns(2)
        with col_q:
            st.subheader("Qualité")
            st.bar_chart(df.set_index("Moteur")[["Score lexical %", "Conf. moteur %", "% douteux"]])
        with col_n:
            st.subheader("Volume (mots)")
            st.bar_chart(df.set_index("Moteur")[["Mots"]])

    with tab_cer:
        render_ground_truth_tab(results)


# ─────────────────────────────────────────────
# RENDU : EXPLICABILITÉ
# ─────────────────────────────────────────────

def render_explainability_tab(results: dict) -> None:
    final_texts = st.session_state.get("ocr_final_texts") or {}
    llm_outputs = st.session_state.get("ocr_llm_outputs") or {}
    structured_source_engine = st.session_state.get("ocr_structured_source_engine")
    structured_extraction = st.session_state.get("ocr_structured_extraction") or []
    segment_count = int(st.session_state.get("ocr_segment_count") or 0)
    segment_boxes = st.session_state.get("ocr_segment_boxes") or []
    segment_preview = st.session_state.get("ocr_segment_preview")

    st.caption("Audit détaillé et recommandations pour l'interprétation des résultats.")

    tab_analysis, tab_decisions, tab_llm, tab_fields = st.tabs([
        "🎯 Analyse intelligente",
        "🧭 Décisions pipeline",
        "🔁 Avant / Après LLM",
        "🧾 Audit des champs",
    ])

    # ── TAB 1 : ANALYSE INTELLIGENTE ──
    with tab_analysis:
        pipeline_score = _compute_pipeline_score(results, llm_outputs)
        consensus = _analyze_consensus(results, final_texts)
        recommendations = _get_recommendations(results, pipeline_score, consensus)

        col_score, col_cons = st.columns(2, gap="large")
        with col_score:
            st.metric(
                "Score qualité du pipeline",
                f"{pipeline_score['score']:.1f}/100",
                delta=f"Min: {pipeline_score.get('min', 0):.1f} | Max: {pipeline_score.get('max', 0):.1f}",
                delta_color="off",
            )
            for insight in pipeline_score.get("insights", []):
                st.caption(insight)

        with col_cons:
            st.metric(
                "Consensus inter-moteurs",
                f"{consensus.get('consensus', 'N/A')}%",
                delta=f"Moyenne: {consensus.get('avg_words', 0):.0f} mots",
                delta_color="off",
            )
            for insight in consensus.get("insights", []):
                st.caption(insight)

        st.divider()
        st.subheader("💡 Recommandations")
        for rec in recommendations:
            st.markdown(f"- {rec}")

    # ── TAB 2 : DÉCISIONS PIPELINE ──
    with tab_decisions:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Moteurs exécutés", len(results))
        c2.metric("Source extraction BDD", structured_source_engine or "N/A")
        c3.metric("Segments détectés", segment_count)
        c4.metric("Actes extraits", len(structured_extraction))

        decision_rows = []
        for technique, ocr_data in results.items():
            stats = (ocr_data.get("analysis") or {}).get("stats") or {}
            raw_text = clean_text((ocr_data.get("text") or {}).get("raw") or "")
            final_text = clean_text(_safe_text(final_texts.get(technique)) or raw_text)
            decision_rows.append({
                "Moteur": technique,
                "Mots": int(stats.get("word_count") or 0),
                "% douteux": round(
                    int(stats.get("doubtful_word_count") or 0) /
                    max(int(stats.get("word_count") or 1), 1) * 100, 2
                ),
                "Conf. moyenne": round(float(stats.get("average_confidence") or 0) * 100, 2)
                    if stats.get("average_confidence") is not None else None,
                "Niveau confiance": _confidence_level_label(stats.get("average_confidence")),
                "Correction LLM": "Oui" if llm_outputs.get(technique) else "Non",
                "Δ caractères": len(final_text) - len(raw_text),
                "Segments OCR": len(ocr_data.get("paragraphs") or []),
            })

        if decision_rows:
            st.dataframe(pd.DataFrame(decision_rows), width="stretch", hide_index=True)
            st.caption("Privilégier : peu de mots douteux, conf. élevée, Δ caractères modéré.")
        else:
            st.info("Aucune décision OCR disponible.")

        if segment_preview is not None:
            with st.expander(f"Aperçu segmentation ({len(segment_boxes)} boîtes)"):
                st.image(segment_preview, caption="Segmentation utilisée par le pipeline")

    # ── TAB 3 : AVANT / APRÈS LLM ──
    with tab_llm:
        engine = st.selectbox("Moteur à inspecter", options=list(results.keys()),
                              key="explainability_engine_select")
        ocr_data = results[engine]
        raw_text = clean_text((ocr_data.get("text") or {}).get("raw") or "")
        final_text = clean_text(final_texts.get(engine) or raw_text)
        llm_output = llm_outputs.get(engine)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Texte OCR brut**")
            st.text_area("", value=raw_text, height=260, disabled=True,
                         label_visibility="collapsed", key=f"explainability_raw_{engine}")
        with col_b:
            st.markdown("**Texte final (après correction éventuelle)**")
            st.text_area("", value=final_text, height=260, disabled=True,
                         label_visibility="collapsed", key=f"explainability_final_{engine}")

        similarity = SequenceMatcher(None, raw_text, final_text).ratio() if (raw_text or final_text) else 1.0
        st.metric("Similarité brut/final", f"{similarity * 100:.1f}%")

        if llm_output:
            with st.expander("Détails sortie LLM"):
                st.json(llm_output)
        else:
            st.info("Aucune correction LLM pour ce moteur.")

        doubtful_words = (ocr_data.get("analysis") or {}).get("doubtful_words") or []
        if doubtful_words:
            with st.expander(f"Mots douteux détectés ({len(doubtful_words)})"):
                st.dataframe(pd.DataFrame(doubtful_words[:250]), width="stretch", hide_index=True)

    # ── TAB 4 : AUDIT DES CHAMPS ──
    with tab_fields:
        if not structured_extraction:
            st.info("Aucun acte structuré à auditer pour ce document.")
        else:
            selected_label = st.selectbox(
                "Acte à auditer",
                options=[f"Acte #{i + 1}" for i in range(len(structured_extraction))],
                key="explainability_structured_select",
            )
            idx = int(selected_label.replace("Acte #", "")) - 1
            act = structured_extraction[idx]
            paragraph_text = act.get("_paragraph_text")
            field_rows = _compute_field_confidence(act, paragraph_text or "")

            c1, c2, c3 = st.columns(3)
            c1.metric("Acte", selected_label)
            c2.metric("Champs totaux", len(field_rows))
            c3.metric("Champs renseignés", sum(1 for r in field_rows if r["Statut"] == "Renseigné"))

            st.dataframe(pd.DataFrame(field_rows), width="stretch", hide_index=True)
            st.caption("✅ > 80% confiance fiable · ⚠️/❌ à confirmer sur le texte source")

            if paragraph_text:
                with st.expander("Texte source du paragraphe"):
                    st.text_area("", value=paragraph_text, height=220, disabled=True,
                                 label_visibility="collapsed",
                                 key=f"explainability_structured_paragraph_{idx}")


# ─────────────────────────────────────────────
# POINT D'ENTRÉE PRINCIPAL
# ─────────────────────────────────────────────

def render_results(file_base_name: str) -> None:
    results = st.session_state.get("ocr_results", {})
    if not results:
        st.info("Veuillez lancer l'extraction pour voir les résultats.")
        return

    st.header("📚 Centre de résultats")

    tab_structured, tab_ocr, tab_comparison, tab_explainability = st.tabs([
        "🗄️ Actes détectés",
        "📝 Résultats OCR",
        "📊 Comparaisons",
        "🧠 Explicabilité",
    ])

    with tab_structured:
        render_structured_extraction(file_base_name)

    with tab_ocr:
        ocr_tabs = st.tabs([f"{OCR_TECHNIQUES[t]['icon']} {t}" for t in results])
        for ocr_tab, technique in zip(ocr_tabs, results):
            with ocr_tab:
                render_engine_result_tab(technique=technique, ocr_data=results[technique])

    with tab_comparison:
        if len(results) > 1:
            render_visual_comparison(results)
        else:
            st.info("Comparaison multi-moteurs indisponible avec un seul moteur.")
            render_ground_truth_tab(results)

    with tab_explainability:
        render_explainability_tab(results)