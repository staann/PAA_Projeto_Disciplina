import numpy as np


def carregar_modelo_sbert(nome_modelo="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(nome_modelo)


def gerar_embeddings_sentence(textos, modelo):
    matriz = modelo.encode(list(textos), show_progress_bar=False)
    return np.asarray(matriz, dtype=np.float32)


def gerar_embedding_query(texto, modelo):
    vetor = modelo.encode(texto, show_progress_bar=False)
    return np.asarray(vetor, dtype=np.float32).reshape(-1)
