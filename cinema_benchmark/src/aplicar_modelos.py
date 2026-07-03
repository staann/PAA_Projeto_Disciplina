import os
import re

import numpy as np


def _tokenizar_texto(texto):
    if isinstance(texto, str):
        return re.findall(r"\b\w+\b", texto.lower())
    return [str(palavra).lower() for palavra in texto]


def processar_linha_word2vec(texto, modelo_w2v, stopwords):
    lista_pesos = []
    stopwords_normalizadas = {palavra.lower() for palavra in stopwords}

    for palavra in _tokenizar_texto(texto):
        if palavra not in stopwords_normalizadas and palavra in modelo_w2v:
            lista_pesos.append(modelo_w2v[palavra])

    return np.mean(lista_pesos, axis=0) if lista_pesos else np.zeros(modelo_w2v.vector_size)


def processar_em_lote_word2vec(coluna_sinopses, modelo_w2v, stopwords):
    vetores_totais = []
    total_filmes = len(coluna_sinopses)

    print(f"iniciando o processamento de {total_filmes} filmes")

    for i, texto in enumerate(coluna_sinopses):
        vetor_filme = processar_linha_word2vec(texto, modelo_w2v, stopwords)
        vetores_totais.append(vetor_filme)

        if (i + 1) % 1000 == 0:
            print(f"progresso: {i + 1}/{total_filmes} filmes processados")

    print("criando matriz final")

    return np.vstack(vetores_totais) if vetores_totais else np.empty((0, modelo_w2v.vector_size))


def salvar_matriz_word2vec(matriz, caminho_arquivo):
    caminho_salvamento = os.path.join(caminho_arquivo, "embeddings_word2vec.npy")

    print(f"salvando a matriz de embeddings em: {caminho_salvamento}")
    np.save(caminho_salvamento, matriz)
    print("matriz binaria salva")


if __name__ == "__main__":
    import pandas as pd

    from modelos_embeddings import carregar_modelo_word2vec, importar_stopwords

    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_data = os.path.join(diretorio_atual, "..", "data", "tratada")
    caminho_data_final = os.path.join(caminho_data, "filmes_processados.csv")

    df = pd.read_csv(caminho_data_final)
    stopwords = importar_stopwords()
    word2vec = carregar_modelo_word2vec()

    matriz_word2vec = processar_em_lote_word2vec(df["plot"], word2vec, stopwords)
    salvar_matriz_word2vec(matriz_word2vec, caminho_data)