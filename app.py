"""
app.py — Streamlit-интерфейс мультимодального RAG по нескольким PDF.

Запуск:  streamlit run app.py
"""

import json
from pathlib import Path

import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Geo RAG · Нефть и газ",
    page_icon="🛢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.src-tag  { background:#2a3f5f; color:#7eb8f7; padding:2px 8px;
            border-radius:10px; font-size:0.76rem; font-weight:600; }
.page-tag { background:#2a3f2a; color:#7ecf7e; padding:2px 8px;
            border-radius:10px; font-size:0.76rem; }
.chunk-box { background:#1e1e2e; border-left:3px solid #6c9ef8;
             padding:8px 12px; border-radius:4px; font-size:0.80rem;
             margin-bottom:5px; color:#cdd6f4; }
.score    { color:#a6e3a1; font-size:0.73rem; font-weight:bold; }
</style>
""", unsafe_allow_html=True)


# ─── Вспомогательные функции (определяем ДО использования) ───────────────────

def short_name(path: str) -> str:
    n = Path(path).stem
    return n[:40] + "…" if len(n) > 40 else n


def get_stats() -> dict:
    f = Path("chroma_db/stats.json")
    return json.loads(f.read_text()) if f.exists() else {}


def find_pdfs_in_dir() -> list[Path]:
    return sorted(Path(".").glob("*.pdf"))


def show_images(images: list[dict]) -> None:
    imgs = images[:4]
    if not imgs:
        return
    cols = st.columns(len(imgs))
    for ci, img_data in enumerate(imgs):
        with cols[ci]:
            try:
                label = f'{short_name(img_data["source_file"])} · стр. {img_data["page"]}'
                st.image(Image.open(img_data["path"]), caption=label, use_container_width=True)
            except Exception:
                st.warning("Изображение недоступно")


def render_chunks(chunks: list[dict]) -> None:
    for c in chunks:
        icon = "🖼" if c["type"] == "image" else "📝"
        st.markdown(
            f'<div class="chunk-box">'
            f'{icon} <span class="src-tag">{short_name(c["source_file"])}</span> '
            f'<span class="page-tag">стр.{c["page"]}</span> '
            f'<span class="score">score={c["score"]}</span><br>'
            f'{c["text"][:280]}{"…" if len(c["text"]) > 280 else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_sources(sources: list[tuple]) -> None:
    if not sources:
        return
    parts = [
        f'<span class="src-tag">{short_name(src)}</span> '
        f'<span class="page-tag">стр. {pg}</span>'
        for src, pg in sources
    ]
    st.markdown("📖 " + " &nbsp; ".join(parts), unsafe_allow_html=True)


# ─── Сайдбар ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🛢 Geo RAG")
    st.caption("Геология и геохимия нефти и газа\nQwen2-VL-7B · bge-base · Ollama")
    st.divider()

    st.subheader("📄 Источники")
    tab_upload, tab_folder = st.tabs(["Загрузить", "Из папки"])

    with tab_upload:
        uploaded_files = st.file_uploader(
            "Выберите PDF",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded_files:
            saved_paths = []
            for f in uploaded_files:
                p = Path(f.name)
                p.write_bytes(f.read())
                saved_paths.append(str(p))
            st.success(f"Загружено: {len(saved_paths)} файлов")

            if st.button("⚡ Индексировать загруженные", type="primary", use_container_width=True):
                with st.spinner("Обработка…"):
                    try:
                        from ingest import ingest_multiple
                        ingest_multiple(saved_paths)
                        st.session_state.pop("rag", None)
                        st.success("Готово!")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

    with tab_folder:
        pdfs = find_pdfs_in_dir()
        if pdfs:
            st.write(f"Найдено {len(pdfs)} PDF:")
            for p in pdfs:
                st.caption(f"• {p.name}")
            if st.button("⚡ Индексировать все", type="primary", use_container_width=True):
                with st.spinner(f"Обработка {len(pdfs)} файлов…"):
                    try:
                        from ingest import ingest_multiple
                        ingest_multiple([str(p) for p in pdfs])
                        st.session_state.pop("rag", None)
                        st.success("Готово!")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
        else:
            st.info("PDF не найдены в текущей папке.")

    stats = get_stats()
    if stats:
        st.divider()
        st.subheader("📊 Индекс")
        c1, c2, c3 = st.columns(3)
        c1.metric("Всего", stats.get("total", 0))
        c2.metric("Текст", stats.get("text", 0))
        c3.metric("Картинки", stats.get("images", 0))

        per_src = stats.get("per_source", {})
        if per_src:
            st.caption("По источникам:")
            for src, cnt in per_src.items():
                st.markdown(
                    f'<span class="src-tag">{short_name(src)}</span> '
                    f'{cnt.get("text", 0)} текст · {cnt.get("images", 0)} изобр.',
                    unsafe_allow_html=True,
                )

    st.divider()
    with st.expander("⚙️ Настройки", expanded=False):
        show_chunks = st.checkbox("Показывать источники", value=True)
        st.session_state["show_chunks"] = show_chunks

        per_src = stats.get("per_source", {}) if stats else {}
        if per_src:
            src_options = ["Все источники"] + list(per_src.keys())
            chosen = st.selectbox("Искать только в:", src_options)
            st.session_state["filter_src"] = None if chosen == "Все источники" else chosen
        else:
            st.session_state["filter_src"] = None


# ─── Основная область ────────────────────────────────────────────────────────

st.header("Вопрос–ответ по геологическим книгам")

if "messages" not in st.session_state:
    st.session_state.messages = []

# История диалога
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])
        if msg.get("images"):
            show_images(msg["images"])
        if st.session_state.get("show_chunks") and msg.get("chunks"):
            with st.expander("Источники", expanded=False):
                render_chunks(msg["chunks"])

# Примеры вопросов
examples = [
    "Что такое геохимия нефти и газа?",
    "Опиши типы нефтяных ловушек",
    "Какие карты и схемы есть в книгах?",
    "Сравни данные из двух книг",
    "Объясни процесс миграции углеводородов",
]
with st.expander("💡 Примеры вопросов", expanded=False):
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state["prefill"] = ex
            st.rerun()

# Ввод
prefill = st.session_state.pop("prefill", "")
query = st.chat_input("Задайте вопрос по книгам…") or prefill

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.info("Поиск в книгах…")

        try:
            if "rag" not in st.session_state:
                placeholder.info("Инициализация RAG…")
                from rag import GeoRAG
                st.session_state.rag = GeoRAG()

            result = st.session_state.rag.answer(
                query,
                filter_source=st.session_state.get("filter_src"),
            )

            placeholder.empty()
            st.write(result["answer"])

            if result["sources"]:
                render_sources(result["sources"])

            if result["images"]:
                st.write("**Релевантные изображения:**")
                show_images(result["images"])

            if st.session_state.get("show_chunks"):
                with st.expander("Источники (чанки)", expanded=False):
                    render_chunks(result["chunks"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "images": result["images"],
                "sources": result["sources"],
                "chunks": result["chunks"],
            })

        except Exception as e:
            placeholder.empty()
            err = str(e)
            if "does not exist" in err or "No such collection" in err:
                st.warning("Сначала проиндексируйте PDF через боковую панель.")
            elif "Connection refused" in err or "ollama" in err.lower():
                st.error("Ollama не запущен. Выполните в терминале: `ollama serve`")
            else:
                st.error(f"Ошибка: {e}")
