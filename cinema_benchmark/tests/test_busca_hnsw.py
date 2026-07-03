import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import MagicMock


diretorio_teste = os.path.dirname(os.path.abspath(__file__))
caminho_src = os.path.abspath(os.path.join(diretorio_teste, '..', 'src'))
sys.path.insert(0, caminho_src)


from sentence_hnsw.busca_hnsw import realizar_busca_hnsw

def test_realizar_busca_hnsw_retorna_resultados_corretos():
    
    dados = {
        "title": ["Filme A", "Filme B", "Filme C"],
        "plot": ["Sinopse A", "Sinopse B", "Sinopse C"]
    }
    df_fake = pd.DataFrame(dados)


    mock_modelo = MagicMock()
    mock_modelo.encode.return_value = np.array([[0.1, 0.2]])

    mock_indice = MagicMock()
    mock_distancias = np.array([[0.5, 1.2]])
    mock_indices = np.array([[1, 0]])
    mock_indice.search.return_value = (mock_distancias, mock_indices)

   
    resultados = realizar_busca_hnsw(
        pergunta="Uma pergunta teste", 
        modelo=mock_modelo, 
        indice_hnsw=mock_indice, 
        df_filmes=df_fake, 
        top_n=2
    )

    
    assert len(resultados) == 2
    
    assert resultados[0]["titulo"] == "Filme B"
    assert resultados[0]["sinopse"] == "Sinopse B"
    assert resultados[0]["score"] == 0.5

    assert resultados[1]["titulo"] == "Filme A"
    assert resultados[1]["sinopse"] == "Sinopse A"
    assert resultados[1]["score"] == 1.2

    mock_modelo.encode.assert_called_once_with(["Uma pergunta teste"])
    assert mock_indice.search.called