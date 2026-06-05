from modelos_embeddings import *
import pandas as pd
import os

caminho_data = "cinema_benchmark/data/tratada"
caminho_data_final = os.path.join(caminho_data, "filmes_processados.csv")

df = pd.read_csv(caminho_data_final)
df_plot = df['plot']
#print(df['title'].head(5))

stopwords = importar_stopwords()
word2vec = carregar_modelo_word2vec()

import numpy as np


def processar_linha_word2vec(texto,modelo_w2v,stopwords):

    lista_pesos = []

    for palavra in texto:
        if palavra not in stopwords:
            if palavra in modelo_w2v:
                lista_pesos.append(modelo_w2v[palavra])

    return np.mean(lista_pesos, axis=0) if lista_pesos else np.zeros(modelo_w2v.vector_size)

def processar_em_lote_word2vec(coluna_sinopses, modelo_w2v,stopwords):

    vetores_totais = []
    total_filmes = len(coluna_sinopses)

    print(f"iniciando o processamento de {total_filmes} filmes")

    for i, texto in enumerate(coluna_sinopses):
    
        vetor_filme = processar_linha_word2vec(texto, modelo_w2v,stopwords)

        vetores_totais.append(vetor_filme)

        if (i + 1) % 1000 == 0:
            print(f"prrogresso: {i + 1}/{total_filmes} filmes processados")

    print("criando matriz final")

    # transf em matriz

    matriz_final = np.vstack(vetores_totais)

    return matriz_final


def salvar_matriz_word2vec(matriz, caminho_arquivo):

    # 1. Defina o caminho completo dentro da sua pasta de dados tratados
    caminho_salvamento = os.path.join(caminho_arquivo, "embeddings_word2vec.npy")

    print(f"salvando a matriz de embeddings em: {caminho_salvamento}")

    # 2. Usa a função nativa do NumPy para gravar o arquivo binário
    np.save(caminho_salvamento, matriz)

    print("matriz binaria salva")


matriz_word2vec = processar_em_lote_word2vec(df_plot, word2vec,stopwords)
salvar_matriz_word2vec(matriz_word2vec, caminho_data)