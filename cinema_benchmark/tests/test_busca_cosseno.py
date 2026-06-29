import numpy as np
import pandas as pd

from sentence_cosseno.busca_cosseno import buscar_cosseno

def test_ordena_por_score_e_respeita_top_n():
    matriz = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    df = pd.DataFrame({"id": [10, 11, 12], "title": ["A", "B", "C"], "plot": ["pa", "pb", "pc"]})
    query = np.array([1.0, 0.0])

    resultados = buscar_cosseno(query, matriz, df, top_n=2)

    assert len(resultados) == 2
    assert resultados[0]["titulo"] == "A"
    assert resultados[0]["posicao"] == 1
    assert resultados[1]["posicao"] == 2
    assert resultados[0]["score"] >= resultados[1]["score"]

def test_chaves_padronizadas():
    matriz = np.array([[1.0, 0.0]])
    df = pd.DataFrame({"id": [1], "title": ["A"], "plot": ["pa"]})

    resultados = buscar_cosseno(np.array([1.0, 0.0]), matriz, df, top_n=1)

    assert set(resultados[0].keys()) == {"posicao", "id", "titulo", "sinopse", "score"}
    assert isinstance(resultados[0]["score"], float)
    assert isinstance(resultados[0]["id"], int)
