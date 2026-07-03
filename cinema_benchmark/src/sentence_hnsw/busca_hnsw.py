def realizar_busca_hnsw(pergunta, modelo, indice_hnsw, df_filmes, top_n=3):
    """
    Recebe a pergunta em texto, os modelos carregados e o K desejado.
    Retorna uma lista de dicionários padronizada.
    """
    vetor_pergunta = modelo.encode([pergunta]).astype('float32')
    distancias, indices = indice_hnsw.search(vetor_pergunta, top_n)
    
    resultados = []
    for posicao, (idx, dist) in enumerate(zip(indices[0], distancias[0]), start=1):
        linha_filme = df_filmes.iloc[idx]
        identificador = int(linha_filme["id"]) if "id" in df_filmes.columns else int(idx)
        dados_filme = {
            "posicao": posicao,
            "id": identificador,
            "titulo": linha_filme["title"],
            "sinopse": linha_filme["plot"],
            "score": float(dist), # Menor = mais similar
        }
        resultados.append(dados_filme)
        
    return resultados