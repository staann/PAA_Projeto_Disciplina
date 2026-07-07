"""Camada de serviço para o frontend Streamlit.

Encapsula (sem alterar) as funções de busca já existentes no projeto,
oferecendo loaders com cache e uma função unificada de busca que aceita
a escolha de embedding, técnica e top-k.
"""

import functools
import os
import sys
import time

import numpy as np

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from aplicar_modelos import processar_linha_word2vec
from llmformater import formatar_resposta_llm
from modelos_embeddings import carregar_modelo_word2vec, importar_stopwords
from packages.embedder_sentence import (
    carregar_modelo_sbert,
    gerar_embedding_query,
    gerar_embeddings_sentence,
)
from packages.io_dados import carregar_embeddings, carregar_filmes
from sentence_cosseno.busca_cosseno import buscar_cosseno
from sentence_hnsw.busca_hnsw import realizar_busca_hnsw
from motor_busca import buscar_cosseno_word2vec

# --- Caminhos dos dados ---------------------------------------------------

_DATA = os.path.abspath(os.path.join(_SRC, "..", "data", "tratada"))
CAMINHO_CSV = os.path.join(_DATA, "filmes_processados.csv")
CAMINHO_W2V = os.path.join(_DATA, "embeddings_word2vec.npy")
CAMINHO_SENTENCE = os.path.join(_DATA, "embeddings_sentence.npy")
CAMINHO_HNSW = os.path.join(_DATA, "hnsw_index.bin")

MODELO_SBERT = "all-MiniLM-L6-v2"

# --- Opções expostas na interface ----------------------------------------

EMBEDDING_WORD2VEC = "Word2Vec"
EMBEDDING_SENTENCE = "SentenceTransformer"

TECNICA_COSSENO = "Cosseno"
TECNICA_HNSW = "HNSW"

# Combinações válidas -> (embedding, tecnica)
COMBINACOES_VALIDAS = {
    (EMBEDDING_WORD2VEC, TECNICA_COSSENO),
    (EMBEDDING_SENTENCE, TECNICA_COSSENO),
    (EMBEDDING_SENTENCE, TECNICA_HNSW),
}


def combinacao_valida(embedding, tecnica):
    return (embedding, tecnica) in COMBINACOES_VALIDAS


# --- Loaders com cache ----------------------------------------------------


@functools.lru_cache(maxsize=1)
def carregar_df():
    return carregar_filmes(CAMINHO_CSV)


@functools.lru_cache(maxsize=1)
def carregar_stopwords():
    return importar_stopwords()


@functools.lru_cache(maxsize=1)
def carregar_word2vec():
    return carregar_modelo_word2vec()


@functools.lru_cache(maxsize=1)
def carregar_matriz_word2vec():
    return carregar_embeddings(CAMINHO_W2V)


@functools.lru_cache(maxsize=1)
def carregar_sbert():
    return carregar_modelo_sbert(MODELO_SBERT)


@functools.lru_cache(maxsize=1)
def carregar_indice_hnsw():
    import faiss

    return faiss.read_index(CAMINHO_HNSW)


@functools.lru_cache(maxsize=1)
def garantir_embeddings_sentence():
    """Carrega os embeddings SBERT completos; gera e cacheia se não existirem."""
    if os.path.exists(CAMINHO_SENTENCE):
        return carregar_embeddings(CAMINHO_SENTENCE)

    df = carregar_df()
    modelo = carregar_sbert()
    textos = df["plot"].astype(str).tolist()
    matriz = gerar_embeddings_sentence(textos, modelo)
    np.save(CAMINHO_SENTENCE, matriz)
    return matriz


def embeddings_sentence_em_cache():
    """Indica se os embeddings SBERT completos já estão salvos em disco."""
    return os.path.exists(CAMINHO_SENTENCE)


# --- Busca unificada ------------------------------------------------------


def _normalizar_resultados(resultados):
    """Garante título, sinopse, score e posição em todos os resultados."""
    normalizados = []
    for posicao, item in enumerate(resultados, start=1):
        normalizados.append(
            {
                "posicao": item.get("posicao", posicao),
                "titulo": item.get("titulo", ""),
                "sinopse": item.get("sinopse", ""),
                "score": float(item.get("score", 0.0)),
            }
        )
    return normalizados


def buscar(pergunta, embedding, tecnica, k=5):
    """Executa a busca escolhida e retorna resultados padronizados.

    Retorna dict com: resultados, tipo_score, tempo_ms, n_documentos,
    embedding, tecnica.
    """
    if not combinacao_valida(embedding, tecnica):
        raise ValueError(
            f"Combinação inválida: {embedding} + {tecnica}."
        )

    df = carregar_df()
    inicio = time.perf_counter()

    if embedding == EMBEDDING_WORD2VEC and tecnica == TECNICA_COSSENO:
        modelo = carregar_word2vec()
        stopwords = carregar_stopwords()
        matriz = carregar_matriz_word2vec()
        vetor = processar_linha_word2vec(pergunta, modelo, stopwords)
        resultados = buscar_cosseno_word2vec(vetor, matriz, df, top_n=k)
        tipo_score = "Similaridade de Cosseno"

    elif embedding == EMBEDDING_SENTENCE and tecnica == TECNICA_COSSENO:
        modelo = carregar_sbert()
        matriz = garantir_embeddings_sentence()
        vetor = gerar_embedding_query(pergunta, modelo)
        resultados = buscar_cosseno(vetor, matriz, df, top_n=k)
        tipo_score = "Similaridade de Cosseno"

    elif embedding == EMBEDDING_SENTENCE and tecnica == TECNICA_HNSW:
        modelo = carregar_sbert()
        indice = carregar_indice_hnsw()
        resultados = realizar_busca_hnsw(pergunta, modelo, indice, df, top_n=k)
        tipo_score = "Distância L2 (menor = mais similar)"

    else:  # pragma: no cover - protegido por combinacao_valida
        raise ValueError(f"Combinação não suportada: {embedding} + {tecnica}.")

    tempo_ms = (time.perf_counter() - inicio) * 1000

    return {
        "resultados": _normalizar_resultados(resultados),
        "tipo_score": tipo_score,
        "tempo_ms": tempo_ms,
        "n_documentos": int(len(df)),
        "embedding": embedding,
        "tecnica": tecnica,
    }


def gerar_resposta_llm(pergunta, resultados, modelo="qwen2.5:1.5b"):
    """Delegação direta para o formatador LLM existente."""
    return formatar_resposta_llm(pergunta=pergunta, filmes=resultados, modelo=modelo)
