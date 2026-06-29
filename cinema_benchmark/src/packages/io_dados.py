import json
import numpy as np
import pandas as pd

def carregar_filmes(caminho_csv):
    return pd.read_csv(caminho_csv)

def salvar_embeddings(matriz, caminho_npy):
    np.save(caminho_npy, matriz)

def carregar_embeddings(caminho_npy):
    return np.load(caminho_npy)

def salvar_metricas(dados, caminho_json):
    with open(caminho_json, "w", encoding="utf-8") as arquivo:
        json.dump(dados, arquivo, ensure_ascii=False, indent=2)
