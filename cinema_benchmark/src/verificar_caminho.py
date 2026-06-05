import os
import gensim.downloader as api

# O Gensim armazena o caminho base na variável de ambiente ou em seu módulo interno
# Você pode rodar isso para ver o caminho absoluto na sua máquina:
diretorio_base = os.path.expanduser("~/gensim-data")
caminho_modelo = os.path.join(
    diretorio_base, "word2vec-google-news-300"
)

print("A pasta onde o modelo está guardado é:", caminho_modelo)