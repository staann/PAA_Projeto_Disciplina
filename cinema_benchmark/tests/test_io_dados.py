import json

import numpy as np
import pandas as pd

from packages.io_dados import (
    carregar_embeddings,
    carregar_filmes,
    salvar_embeddings,
    salvar_metricas,
)

def test_round_trip_embeddings(tmp_path):
    matriz = np.random.rand(3, 4).astype(np.float32)
    caminho = str(tmp_path / "emb.npy")

    salvar_embeddings(matriz, caminho)
    carregada = carregar_embeddings(caminho)

    assert np.allclose(matriz, carregada)

def test_salvar_metricas(tmp_path):
    dados = {"pipeline": "sentence_cosseno", "tempos_segundos": {"busca": 0.1}}
    caminho = str(tmp_path / "m.json")

    salvar_metricas(dados, caminho)

    with open(caminho, encoding="utf-8") as arquivo:
        lido = json.load(arquivo)
    assert lido == dados

def test_carregar_filmes(tmp_path):
    caminho = str(tmp_path / "f.csv")
    pd.DataFrame({"id": [1, 2], "plot": ["a", "b"], "title": ["t1", "t2"]}).to_csv(caminho, index=False)

    df = carregar_filmes(caminho)

    assert len(df) == 2
    assert "plot" in df.columns
    assert "title" in df.columns
