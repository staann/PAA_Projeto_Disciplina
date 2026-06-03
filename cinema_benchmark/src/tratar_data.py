import os
import pandas as pd

def carregar_dados(caminho_dados = "cinema_benchmark/data/cmu-movie-summary-corpus"):

    caminho_metadata = os.path.join(caminho_dados, "movie.metadata.tsv")
    caminho_plot = os.path.join(caminho_dados, "plot_summaries.txt")

    if not os.path.exists(caminho_metadata) or not os.path.exists(caminho_plot):
        raise FileNotFoundError(f"arquivos não encontrados em {caminho_dados}. Execute o script de download primeiro.")
    
    print("começando o carregamento dos metadados...")

    colunas_metadata = ["id", "base_id", "title", "year", "revenue", "runtime", "language", "country", "gender"]

    df_movies = pd.read_csv(caminho_metadata, sep="\t", names = colunas_metadata, header=None)

    df_movies = df_movies[["id", "title", "gender"]]

    print("carregamento concluído dos metadados.")

    print("Carregando sinopses (plots)...")

    colunas_plots = ["id", "plot"]
    df_plots = pd.read_csv(caminho_plot, sep="\t", header=None, names=colunas_plots)

    df_final = pd.merge(df_plots, df_movies, on="id", how="inner")

    df_final = df_final.dropna(subset=["plot", "title"])

    print(f"dataset carregado com sucesso. Total de filmes: {len(df_final)}")

    return df_final

if __name__ == "__main__": 
    try:

        dados = carregar_dados()

        print("\nPrimeiras linhas do resultado:")
        print(dados.head(3))

    except Exception as e:
        print("Erro ao carregar os dados:", e)