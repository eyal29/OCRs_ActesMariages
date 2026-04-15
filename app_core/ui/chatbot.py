import pandas as pd
import streamlit as st

from app_core.pipeline.database import (
    get_database_overview,
    get_db_path,
    get_table_schema,
    run_readonly_query,
    clean_duplicates,
)
from app_core.pipeline.llm_client import answer_marriages_question, generate_sql_for_marriages
from app_core.setup import get_api_key_for_provider


EXAMPLE_QUESTIONS = [
    "Combien d'actes de mariage sont enregistrés dans la base ?",
    "Liste les 10 actes les plus récents avec la date et les noms des mariés.",
    "Quels sont les noms des mariés pour les actes du document Archives_1937 ?",
    "Quelles professions apparaissent le plus souvent pour marie_1 ?",
]


def _render_chat_history() -> None:
    for message in st.session_state.get("chat_messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sql"):
                with st.expander("SQL utilisé"):
                    st.code(message["sql"], language="sql")
            if message.get("dataframe") is not None:
                with st.expander("Résultats tabulaires"):
                    st.dataframe(pd.DataFrame(message["dataframe"]), width="stretch")


def render_chatbot_page(llm_provider: str, llm_model: str) -> None:
    st.title("💬 Chatbot BDD mariages")
    st.caption("Interrogez la base SQLite des actes de mariage en langage naturel.")

    overview = get_database_overview()
    col1, col2, col3 = st.columns(3)
    col1.metric("Actes en base", overview["total_rows"])
    col2.metric("Sources distinctes", overview["distinct_sources"])
    col3.metric("Dernier import", overview["latest_processed_at"] or "N/A")

    with st.expander("Aide / exemples", expanded=False):
        st.write("Exemples de questions :")
        for question in EXAMPLE_QUESTIONS:
            st.markdown(f"- {question}")
        st.caption(f"Base utilisée: {get_db_path()}")

    col_chat, col_clean = st.columns([3, 1])
    
    with col_chat:
        if st.button("🧹 Vider la conversation"):
            st.session_state["chat_messages"] = []
            st.rerun()
    
    with col_clean:
        if st.button("🔄 Nettoyer les doublons"):
            try:
                result = clean_duplicates(keep_strategy="most_complete")
                if result["records_deleted"] > 0:
                    st.success(
                        f"✅ {result['duplicate_groups_found']} groupe(s) de doublons trouvé(s), "
                        f"{result['records_deleted']} enregistrement(s) supprimé(s)."
                    )
                else:
                    st.info("✨ Aucun doublon détecté, la base est propre!")
            except Exception as e:
                st.error(f"❌ Erreur lors du nettoyage: {e}")

    _render_chat_history()

    user_question = st.chat_input("Posez une question sur les actes enregistrés...")
    if not user_question:
        return

    st.session_state.setdefault("chat_messages", []).append(
        {"role": "user", "content": user_question}
    )

    api_key = get_api_key_for_provider(llm_provider)
    if not api_key:
        st.session_state["chat_messages"].append(
            {
                "role": "assistant",
                "content": (
                    f"Clé API manquante pour {llm_provider}. "
                    "Configurez-la dans le .env pour activer le chatbot."
                ),
            }
        )
        st.rerun()

    try:
        schema = get_table_schema()
        sql_payload = generate_sql_for_marriages(
            question=user_question,
            schema=schema,
            api_key=api_key,
            provider=llm_provider,
            model=llm_model,
        )
        sql_query = (sql_payload.get("sql") or "").strip()
        dataframe = run_readonly_query(sql_query)
        rows = dataframe.to_dict(orient="records")
        answer = answer_marriages_question(
            question=user_question,
            sql=sql_query,
            rows=rows,
            api_key=api_key,
            provider=llm_provider,
            model=llm_model,
        )
        st.session_state["chat_messages"].append(
            {
                "role": "assistant",
                "content": answer or "Aucune réponse générée.",
                "sql": sql_query,
                "dataframe": rows[:200],
            }
        )
    except Exception as exc:
        st.session_state["chat_messages"].append(
            {
                "role": "assistant",
                "content": f"Je n'ai pas pu interroger la base: {exc}",
            }
        )

    st.rerun()