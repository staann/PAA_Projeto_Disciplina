import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
from aplicar_modelos import processar_linha_word2vec
from modelos_embeddings import importar_stopwords,carregar_modelo_word2vec


print("Iniciando aplicação de StopWords")
stopwords = importar_stopwords()

print("Carregando Modelo Word2Vec")
word2vec = carregar_modelo_word2vec()

def buscar_cosseno_word2vec(vetor_pergunta, matriz_embeddings, df_filmes, top_n=3):

    query_respeitado = vetor_pergunta.reshape(1, -1)

    
    # [0] no final serve para extrair a lista simples de scores da matriz retornada

    scores = cosine_similarity(query_respeitado, matriz_embeddings)[0]

    indices_vencedores = np.argsort(scores)[::-1][:top_n]

    resultados = []

    # indices numéricos e busca no Pandas
    for idx in indices_vencedores:
        # Captura a linha usando o .iloc
        linha_filme = df_filmes.iloc[idx]

        # monta um dicionário
        dados_filme = {
            "titulo": linha_filme["title"],
            "sinopse": linha_filme["plot"],
            "score_cosseno": scores[idx],
        }
        resultados.append(dados_filme)

    return resultados


print("Iniciando carregamentos de filmes")
caminho_data = "cinema_benchmark/data/tratada"
caminho_data_final = os.path.join(caminho_data, "filmes_processados.csv")
df = pd.read_csv(caminho_data_final)

print("Iniciando carregamento da matriz de embeddings")

matriz_data_final = os.path.join(caminho_data, "embeddings_word2vec.npy")

matriz_w2v = np.load(matriz_data_final)
print("Matriz carregada com sucesso!")

pergunta_usuario = input("\nDigite sua pergunta relacionada aos filmes: ")


vetor_da_pergunta = processar_linha_word2vec(
    pergunta_usuario, word2vec, stopwords
)


filmes_encontrados = buscar_cosseno_word2vec(
    vetor_da_pergunta, matriz_w2v, df, top_n=3
)

print("\nFILMES ENCONTRADOS ")
for i, filme in enumerate(filmes_encontrados):
    print(f"\n{i+1}º Lugar: {filme['titulo']}")
    print(f"Grau de Similaridade: {filme['score_cosseno']:.4f}")
    print(f"Trecho da Sinopse: {filme['sinopse'][:150]}...")