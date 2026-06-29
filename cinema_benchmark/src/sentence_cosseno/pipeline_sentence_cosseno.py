import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from packages.embedder_sentence import (
    carregar_modelo_sbert,
    gerar_embedding_query,
    gerar_embeddings_sentence,
)
from packages.io_dados import (
    carregar_embeddings,
    carregar_filmes,
    salvar_embeddings,
    salvar_metricas,
)
from packages.temporizador import medir_tempo
from sentence_cosseno.busca_cosseno import buscar_cosseno

MODELO = "all-MiniLM-L6-v2"
CSV = "cinema_benchmark/data/tratada/filmes_processados.csv"
NPY = "cinema_benchmark/data/tratada/embeddings_sentence.npy"
METRICAS = "cinema_benchmark/data/tratada/metricas_sentence_cosseno.json"

def construir_indice(caminho_csv, caminho_npy, nome_modelo=MODELO, modelo=None):
    df = carregar_filmes(caminho_csv)

    tempo_modelo = 0.0
    if modelo is None:
        modelo, tempo_modelo = medir_tempo(carregar_modelo_sbert, nome_modelo)

    textos = df["plot"].astype(str).tolist()
    matriz, tempo_encode = medir_tempo(gerar_embeddings_sentence, textos, modelo)
    salvar_embeddings(matriz, caminho_npy)

    return {
        "modelo_embedding": nome_modelo,
        "dimensao": int(matriz.shape[1]),
        "n_documentos": int(matriz.shape[0]),
        "tempos_segundos": {
            "carregar_modelo": tempo_modelo,
            "encode_corpus": tempo_encode,
        },
    }

def executar_busca(pergunta, caminho_csv, caminho_npy, top_n=10, nome_modelo=MODELO, modelo=None):
    df = carregar_filmes(caminho_csv)
    matriz = carregar_embeddings(caminho_npy)

    tempo_modelo = 0.0
    if modelo is None:
        modelo, tempo_modelo = medir_tempo(carregar_modelo_sbert, nome_modelo)

    vetor_query, tempo_query = medir_tempo(gerar_embedding_query, pergunta, modelo)
    resultados, tempo_busca = medir_tempo(buscar_cosseno, vetor_query, matriz, df, top_n)

    return {
        "pipeline": "sentence_cosseno",
        "modelo_embedding": nome_modelo,
        "dimensao": int(matriz.shape[1]),
        "metodo_busca": "cosseno_forca_bruta",
        "n_documentos": int(matriz.shape[0]),
        "pergunta": pergunta,
        "top_n": top_n,
        "resultados": resultados,
        "tempos_segundos": {
            "carregar_modelo": tempo_modelo,
            "encode_query": tempo_query,
            "busca": tempo_busca,
            "inferencia_total": tempo_query + tempo_busca,
        },
    }

def executar_pipeline_completo(
    pergunta,
    caminho_csv=CSV,
    caminho_npy=NPY,
    caminho_metricas=METRICAS,
    top_n=10,
    nome_modelo=MODELO,
    modelo=None,
):
    if modelo is None:
        modelo = carregar_modelo_sbert(nome_modelo)

    info_indice = construir_indice(caminho_csv, caminho_npy, nome_modelo, modelo=modelo)
    saida = executar_busca(pergunta, caminho_csv, caminho_npy, top_n, nome_modelo, modelo=modelo)
    saida["tempos_segundos"]["encode_corpus"] = info_indice["tempos_segundos"]["encode_corpus"]

    salvar_metricas(saida, caminho_metricas)
    return saida


if __name__ == "__main__":
    pergunta = sys.argv[1] if len(sys.argv) > 1 else "a movie about adventure and exploration"
    resultado = executar_pipeline_completo(pergunta)

    print(f"\nPergunta: {resultado['pergunta']}")
    print(f"Documentos: {resultado['n_documentos']} | Dimensao: {resultado['dimensao']}")
    for filme in resultado["resultados"]:
        print(f"\n{filme['posicao']}o. {filme['titulo']} (score {filme['score']:.4f})")
        print(f"   {filme['sinopse'][:150]}...")
    print(f"\nTempos (s): {resultado['tempos_segundos']}")
