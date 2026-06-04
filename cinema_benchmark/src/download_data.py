# set the KAGGLE_API_TOKEN environment variable:
#var = "export KAGGLE_API_TOKEN="
import os
import shutil
import kagglehub

diretorio_destino = "cinema_benchmark/data"


os.makedirs(diretorio_destino, exist_ok=True)

# autenticação do Kaggle (certifique-se de que a variável de ambiente está configurada)
api = os.getenv("KAGGLE_API_TOKEN")

try:
    print("iniciando o download do dataset via kagglehub...")

    path_cache = kagglehub.dataset_download("srikarmell/cmu-movie-summary-corpus")
    print("dDownload concluido no cache:", path_cache)

    #mover os arquivos do cache para a sua pasta do projeto
    arquivos = os.listdir(path_cache)
    for arquivo in arquivos:
        origem = os.path.join(path_cache, arquivo)
        destino = os.path.join(diretorio_destino, arquivo)
        
        # se arquivo já existir no destino logo removemos para atualizar
        if os.path.exists(destino):
            if os.path.isdir(destino):
                shutil.rmtree(destino)
            else:
                os.remove(destino)
                
        shutil.move(origem, destino)
        
    print(f"sucesso, arquivos em : {diretorio_destino}")

except FileNotFoundError:
    print("dataset não encontrado no Kaggle.")

except Exception as e:
    print("error ocorreu:", e)