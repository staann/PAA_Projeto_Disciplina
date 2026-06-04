import os
from tratar_data import carregar_dados


def executar_pipeline_dados():


    pasta_destino = "cinema_benchmark/data/tratada"
    arquivo_csv = os.path.join(pasta_destino, "filmes_processados.csv")

    os.makedirs(pasta_destino, exist_ok=True)

    print("iniciando o Processamento de dados")

    try:

        dados = carregar_dados()

        print(f"salvando os dados em: {arquivo_csv}")

        dados.to_csv(arquivo_csv, index=False, sep=",", encoding="utf-8")

        print("processo concluido")
        print(f"arquivo gerado com {len(dados)} linhas")

    except FileNotFoundError as e:
        print(f"erro de arquivo/diretorio: {e}")


    except Exception as e:
        print(f"\nerro: {e}")


if __name__ == "__main__":
    executar_pipeline_dados()