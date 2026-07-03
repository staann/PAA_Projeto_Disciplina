import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

_SRC = os.path.abspath(os.path.dirname(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from aplicar_modelos import processar_linha_word2vec
from llmformater import formatar_resposta_llm
from modelos_embeddings import carregar_modelo_word2vec, importar_stopwords
from sentence_hnsw.busca_hnsw import realizar_busca_hnsw


def buscar_cosseno_word2vec(vetor_pergunta, matriz_embeddings, df_filmes, top_n=3):
    query_respeitado = vetor_pergunta.reshape(1, -1)
    scores = cosine_similarity(query_respeitado, matriz_embeddings)[0]
    indices_vencedores = np.argsort(scores)[::-1][:top_n]

    resultados = []
    for idx in indices_vencedores:
        linha_filme = df_filmes.iloc[idx]
        resultados.append(
            {
                "titulo": linha_filme["title"],
                "sinopse": linha_filme["plot"],
                "score": float(scores[idx]),
            }
        )
    return resultados


def main():
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError as exc:
        raise ImportError(
            "As dependências `sentence-transformers` e `faiss-cpu` são necessárias para executar o motor de busca."
        ) from exc

    print("Iniciando aplicação de StopWords...")
    stopwords = importar_stopwords()

    print("Carregando Modelo Word2Vec...")
    word2vec = carregar_modelo_word2vec()

    print("Carregando Modelo Sentence Transformers (HNSW)...")
    modelo_sentence = SentenceTransformer("all-MiniLM-L6-v2")

    print("Iniciando carregamentos de filmes...")
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_data = os.path.join(diretorio_atual, "..", "data", "tratada")

    caminho_data_final = os.path.join(caminho_data, "filmes_processados.csv")
    df = pd.read_csv(caminho_data_final)

    print("Carregando Matriz de Embeddings (Word2Vec)...")
    matriz_data_final = os.path.join(caminho_data, "embeddings_word2vec.npy")
    matriz_w2v = np.load(matriz_data_final)

    print("Carregando Grafo (HNSW)...")
    caminho_indice = os.path.join(caminho_data, "hnsw_index.bin")
    indice_hnsw_carregado = faiss.read_index(caminho_indice)

    print("\nTodos os dados carregados com sucesso!")

    while True:
        print("\n" + "=" * 50)
        pergunta_usuario = input("Digite sua pergunta relacionada aos filmes: ")

        print("\nEscolha o motor de busca:")
        print("1 - Word2Vec + Similaridade de Cosseno")
        print("2 - Sentence Embeddings + HNSW")
        escolha = input("Opção (1 ou 2): ")

        inicio_busca = time.perf_counter()

        if escolha == "1":
            print("\nExecutando busca por Cosseno...")
            vetor_da_pergunta = processar_linha_word2vec(pergunta_usuario, word2vec, stopwords)
            filmes_encontrados = buscar_cosseno_word2vec(vetor_da_pergunta, matriz_w2v, df, top_n=10)
            tipo_score = "Similaridade de Cosseno"
        else:
            print("\nExecutando busca por HNSW...")
            filmes_encontrados = realizar_busca_hnsw(
                pergunta_usuario,
                modelo_sentence,
                indice_hnsw_carregado,
                df,
                top_n=10,
            )
            tipo_score = "Distância L2"

        fim_busca = time.perf_counter()
        tempo_ms = (fim_busca - inicio_busca) * 1000

        print(f"\nTempo de inferência: {tempo_ms:.2f} ms")
        print("\nFILMES ENCONTRADOS")
        for i, filme in enumerate(filmes_encontrados):
            print(f"\n{i + 1}º Lugar: {filme['titulo']}")
            print(f"Grau ({tipo_score}): {filme['score']:.4f}")
            print(f"Trecho da Sinopse: {filme['sinopse'][:150]}...")

        print("\n-------------------------")
        print("\n[Mecanismo RAG] Enviando dados para a LLM local...")

        resposta_final = formatar_resposta_llm(
            pergunta=pergunta_usuario,
            filmes=filmes_encontrados,
            modelo="qwen2.5:1.5b",
        )
        print("\n=============================================")
        print("===       RESPOSTA DO ASSISTENTE LLM      ===")
        print("=============================================")
        print(resposta_final)
        print("=============================================")


if __name__ == "__main__":
    main()
