import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def extrair_id(linha, colunas, indice):
    if "id" in colunas:
        try:
            return int(linha["id"])
        except (TypeError, ValueError):
            return str(linha["id"])
    return int(indice)


def buscar_cosseno(vetor_query, matriz_embeddings, df_filmes, top_n=10):
    query = np.asarray(vetor_query, dtype=np.float32).reshape(1, -1)
    scores = cosine_similarity(query, matriz_embeddings)[0]
    indices = np.argsort(scores)[::-1][:top_n]

    colunas = df_filmes.columns
    resultados = []
    for posicao, indice in enumerate(indices, start=1):
        linha = df_filmes.iloc[int(indice)]
        resultados.append(
            {
                "posicao": posicao,
                "id": extrair_id(linha, colunas, indice),
                "titulo": str(linha["title"]),
                "sinopse": str(linha["plot"]),
                "score": float(scores[indice]),
            }
        )
    return resultados
