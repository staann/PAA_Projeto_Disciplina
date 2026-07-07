"""Interface Streamlit para busca semântica de filmes.

Permite escolher o embedding, a técnica de busca e o top-k, digitar um
prompt em inglês e ver o filme mais similar (com resposta opcional via LLM)
além dos k resultados ranqueados.

Execução:
    source .venv/bin/activate && streamlit run cinema_benchmark/app.py
"""

import base64
import os
import sys

import streamlit as st

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from frontend import servico_busca as servico

_LOGO = os.path.join(_SRC, "assets", "unb-logo.png")

_UNB_AZUL = "#003087"
_UNB_VERDE = "#009640"

st.set_page_config(
    page_title="Assistente de Filmes UnB",
    page_icon=_LOGO if os.path.exists(_LOGO) else "🎬",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def _preparar_word2vec():
    servico.carregar_word2vec()
    servico.carregar_matriz_word2vec()
    servico.carregar_stopwords()
    return True


@st.cache_resource(show_spinner=False)
def _preparar_sbert():
    servico.carregar_sbert()
    return True


@st.cache_resource(show_spinner=False)
def _preparar_hnsw():
    servico.carregar_indice_hnsw()
    return True


def _executar_busca(pergunta, embedding, tecnica, k):
    return servico.buscar(pergunta, embedding, tecnica, k)


def _carregar_recursos(embedding, tecnica):
    if embedding == servico.EMBEDDING_WORD2VEC:
        with st.spinner("Carregando Word2Vec e embeddings..."):
            _preparar_word2vec()
    else:
        with st.spinner("Carregando modelo SentenceTransformer..."):
            _preparar_sbert()
        if tecnica == servico.TECNICA_HNSW:
            with st.spinner("Carregando índice HNSW..."):
                _preparar_hnsw()


st.markdown(
    f"""
    <style>
    /* Esconde o menu/rodapé padrão para um visual limpo */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    .block-container {{
        max-width: 760px;
        padding-top: 3rem;
    }}

    /* Cabeçalho central da tela inicial */
    .unb-hero {{
        text-align: center;
        margin-bottom: 1.5rem;
    }}
    .unb-hero h1 {{
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0.5rem 0 0.25rem 0;
    }}
    .unb-hero p {{
        color: #6b7280;
        margin: 0;
    }}

    /* Botões (chips e primário) com o azul da UnB */
    .stButton > button {{
        border-radius: 9999px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        color: #374151;
        font-size: 0.85rem;
        padding: 0.35rem 0.9rem;
        transition: all 0.15s ease;
    }}
    .stButton > button:hover {{
        border-color: {_UNB_AZUL};
        color: {_UNB_AZUL};
    }}

    /* Balão do chat_input */
    [data-testid="stChatInput"] textarea {{
        border-radius: 12px;
    }}

    /* Barra de acento verde/azul no filme mais similar */
    .unb-top {{
        border-left: 4px solid {_UNB_VERDE};
        padding-left: 0.9rem;
        margin: 0.25rem 0 0.75rem 0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []
if "prompt_pendente" not in st.session_state:
    st.session_state.prompt_pendente = None


_SUGESTOES = [
    "a movie about space exploration and aliens",
    "a heist gone terribly wrong",
    "a coming-of-age drama in the 80s",
    "an epic medieval fantasy adventure",
]


def _render_resultado(saida, usar_llm=False, resposta_llm=None):
    resultados = saida.get("resultados", [])
    if not resultados:
        st.info("Nenhum resultado encontrado.")
        return

    top = resultados[0]

    col1, col2 = st.columns(2)
    col1.metric("Tempo de inferência", f"{saida['tempo_ms']:.1f} ms")
    col2.metric("Documentos no corpus", f"{saida['n_documentos']:,}")

    st.markdown("**🥇 Filme mais similar**")
    st.markdown(
        f"<div class='unb-top'><b>{top['titulo']}</b><br>"
        f"<span style='color:#6b7280;font-size:0.85rem'>"
        f"{saida['tipo_score']}: {top['score']:.4f}</span></div>",
        unsafe_allow_html=True,
    )
    st.write(top["sinopse"])

    if usar_llm and resposta_llm is not None:
        st.markdown("**🤖 Resposta do assistente**")
        if resposta_llm.startswith("[Erro"):
            st.warning(resposta_llm)
        else:
            st.info(resposta_llm)

    st.markdown(f"**📋 Top {len(resultados)} resultados**")
    for filme in resultados:
        with st.expander(
            f"{filme['posicao']}º — {filme['titulo']} "
            f"(score: {filme['score']:.4f})"
        ):
            st.write(filme["sinopse"])

conversa_vazia = len(st.session_state.mensagens) == 0

if conversa_vazia:
    if os.path.exists(_LOGO):
        with open(_LOGO, "rb") as _f:
            _logo_b64 = base64.b64encode(_f.read()).decode("utf-8")
        st.markdown(
            "<div style='display:flex;justify-content:center;'>"
            f"<img src='data:image/png;base64,{_logo_b64}' width='72'/>"
            "</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<div class='unb-hero'>"
        "<h1>Assistente de Filmes</h1>"
        "<p>Descreva o filme que procura — a busca semântica encontra o mais parecido.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


if hasattr(st, "popover"):
    _config_ctx = st.popover("⚙️ Configurações", use_container_width=False)
else:  # pragma: no cover - versões antigas do Streamlit
    _config_ctx = st.expander("⚙️ Configurações")

with _config_ctx:
    embedding = st.selectbox(
        "Embedding",
        [servico.EMBEDDING_SENTENCE, servico.EMBEDDING_WORD2VEC],
        help="Modelo usado para transformar textos em vetores. "
        "SentenceTransformer (SBERT) é o recomendado para busca semântica; "
        "Word2Vec é um baseline mais fraco (média de palavras).",
    )

    tecnicas_disponiveis = [
        t
        for t in (servico.TECNICA_HNSW, servico.TECNICA_COSSENO)
        if servico.combinacao_valida(embedding, t)
    ]

    tecnica = st.selectbox(
        "Técnica de busca",
        tecnicas_disponiveis,
        help="HNSW = índice aproximado (faiss), rápido; Cosseno = força bruta.",
    )

    if embedding == servico.EMBEDDING_WORD2VEC:
        st.caption("Word2Vec só possui busca por Cosseno (sem índice HNSW).")
        st.warning(
            "Word2Vec representa cada filme pela **média** dos vetores de todas "
            "as palavras da sinopse. Para consultas curtas isso dilui o significado "
            "e tende a produzir resultados fracos (similaridades ~0.1). "
            "Use apenas como baseline de comparação; prefira o SentenceTransformer."
        )

    k = st.slider("Top-k resultados", min_value=1, max_value=20, value=5)

    st.divider()

    usar_llm = st.toggle(
        "Resposta em texto (LLM via Ollama)",
        value=False,
        help="Gera uma resposta em linguagem natural sobre o filme mais similar.",
    )
    modelo_llm = st.text_input(
        "Modelo Ollama", value="qwen2.5:1.5b", disabled=not usar_llm
    )

    if embedding == servico.EMBEDDING_WORD2VEC:
        st.info(
            "O Word2Vec (Google News 300, ~1.6GB) é baixado pelo gensim no 1º uso."
        )
    if (
        embedding == servico.EMBEDDING_SENTENCE
        and tecnica == servico.TECNICA_COSSENO
        and not servico.embeddings_sentence_em_cache()
    ):
        st.warning(
            "SBERT + Cosseno: os embeddings dos 42204 filmes serão gerados na 1ª "
            "busca (~1min) e salvos em cache para as próximas."
        )

if conversa_vazia:
    st.caption("Experimente:")
    chip_cols = st.columns(2)
    for i, sugestao in enumerate(_SUGESTOES):
        if chip_cols[i % 2].button(sugestao, key=f"chip_{i}"):
            st.session_state.prompt_pendente = sugestao
            st.rerun()

for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            _render_resultado(
                msg["saida"],
                usar_llm=msg.get("usar_llm", False),
                resposta_llm=msg.get("resposta_llm"),
            )


prompt = st.chat_input("Descreva o filme que procura (em inglês)...")
if not prompt and st.session_state.prompt_pendente:
    prompt = st.session_state.prompt_pendente
    st.session_state.prompt_pendente = None

if prompt:
    prompt = prompt.strip()
    st.session_state.mensagens.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        if not servico.combinacao_valida(embedding, tecnica):
            st.error(f"Combinação inválida: {embedding} + {tecnica}.")
        else:
            _carregar_recursos(embedding, tecnica)

            spinner_msg = "Buscando..."
            if (
                embedding == servico.EMBEDDING_SENTENCE
                and tecnica == servico.TECNICA_COSSENO
                and not servico.embeddings_sentence_em_cache()
            ):
                spinner_msg = "Gerando embeddings do corpus e buscando (~1min)..."

            with st.spinner(spinner_msg):
                saida = _executar_busca(prompt, embedding, tecnica, k)

            resposta_llm = None
            if usar_llm and saida["resultados"]:
                with st.spinner("Gerando resposta com o LLM (Ollama)..."):
                    resposta_llm = servico.gerar_resposta_llm(
                        prompt, saida["resultados"], modelo=modelo_llm
                    )

            _render_resultado(saida, usar_llm=usar_llm, resposta_llm=resposta_llm)

            st.session_state.mensagens.append(
                {
                    "role": "assistant",
                    "saida": saida,
                    "usar_llm": usar_llm,
                    "resposta_llm": resposta_llm,
                }
            )
