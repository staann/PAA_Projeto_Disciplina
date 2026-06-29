import types

import numpy as np
import pandas as pd

from sentence_cosseno.pipeline_sentence_cosseno import construir_indice, executar_busca

def _fake_modelo():
    return types.SimpleNamespace(
        encode=lambda textos, **kw: (
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            if isinstance(textos, str)
            else np.eye(len(textos), 4, dtype=np.float32)
        )
    )

def _csv(tmp_path):
    caminho = str(tmp_path / "filmes.csv")
    pd.DataFrame({"id": [1, 2, 3], "plot": ["pa", "pb", "pc"], "title": ["A", "B", "C"]}).to_csv(
        caminho, index=False
    )
    return caminho

def test_construir_indice_tem_encode_corpus(tmp_path):
    caminho_csv = _csv(tmp_path)
    caminho_npy = str(tmp_path / "emb.npy")

    info = construir_indice(caminho_csv, caminho_npy, modelo=_fake_modelo())

    assert info["n_documentos"] == 3
    assert info["dimensao"] == 4
    assert "encode_corpus" in info["tempos_segundos"]

def test_executar_busca_schema(tmp_path):
    caminho_csv = _csv(tmp_path)
    caminho_npy = str(tmp_path / "emb.npy")
    modelo = _fake_modelo()
    construir_indice(caminho_csv, caminho_npy, modelo=modelo)

    saida = executar_busca("oi", caminho_csv, caminho_npy, top_n=2, modelo=modelo)

    assert saida["pipeline"] == "sentence_cosseno"
    assert saida["metodo_busca"] == "cosseno_forca_bruta"
    assert saida["top_n"] == 2
    assert saida["n_documentos"] == 3
    assert len(saida["resultados"]) == 2
    assert saida["resultados"][0]["titulo"] == "A"

    tempos = saida["tempos_segundos"]
    for chave in ("carregar_modelo", "encode_query", "busca", "inferencia_total"):
        assert chave in tempos

    assert set(saida["resultados"][0].keys()) == {"posicao", "id", "titulo", "sinopse", "score"}
