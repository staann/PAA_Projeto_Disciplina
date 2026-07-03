import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

def construir_e_salvar_indice_hnsw():
    print("Carregando o modelo SentenceTransformer...")
    modelo = SentenceTransformer('all-MiniLM-L6-v2')

    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    # Volta duas pastas: sentence_hnsw -> src -> raiz
    caminho_csv = os.path.join(diretorio_atual, '..', '..', 'data', 'tratada', 'filmes_processados.csv')
    df = pd.read_csv(caminho_csv)

    sinopses = df['plot'].fillna("").tolist()

    print(f"Gerando embeddings para {len(sinopses)} filmes...")
    embeddings = modelo.encode(sinopses, show_progress_bar=True).astype('float32')

    print("Construindo o grafo HNSW...")
    dimensao = embeddings.shape[1]
    M = 32
    indice_hnsw = faiss.IndexHNSWFlat(dimensao, M)

    print("Adicionando os vértices ao índice...")
    indice_hnsw.add(embeddings)

    caminho_indice = os.path.join(diretorio_atual, '..', '..', 'data', 'tratada', 'hnsw_index.bin')
    print(f"Salvando o índice em disco: {caminho_indice}")
    faiss.write_index(indice_hnsw, caminho_indice)
    print("Grafo HNSW salvo com sucesso!")

if __name__ == "__main__":
    construir_e_salvar_indice_hnsw()