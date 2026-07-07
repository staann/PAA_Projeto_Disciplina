"""Gera e persiste os embeddings SBERT do corpus completo.

Cria `data/tratada/embeddings_sentence.npy` (uma linha por filme, alinhado ao
CSV), evitando que o frontend regenere os ~42204 embeddings a cada sessão na
busca SentenceTransformer + Cosseno.

Uso:
    source .venv/bin/activate && python cinema_benchmark/src/gerar_embeddings_sentence.py
"""

import os
import sys

import numpy as np

_SRC = os.path.abspath(os.path.dirname(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from packages.embedder_sentence import (
    carregar_modelo_sbert,
    gerar_embeddings_sentence,
)
from packages.io_dados import carregar_filmes

_DATA = os.path.abspath(os.path.join(_SRC, "..", "data", "tratada"))
CAMINHO_CSV = os.path.join(_DATA, "filmes_processados.csv")
CAMINHO_SENTENCE = os.path.join(_DATA, "embeddings_sentence.npy")
MODELO_SBERT = "all-MiniLM-L6-v2"


def main(forcar=False):
    if os.path.exists(CAMINHO_SENTENCE) and not forcar:
        matriz = np.load(CAMINHO_SENTENCE, mmap_mode="r")
        print(f"Embeddings já existem em {CAMINHO_SENTENCE} {matriz.shape}.")
        print("Use forcar=True para regenerar.")
        return

    print("Carregando CSV de filmes...")
    df = carregar_filmes(CAMINHO_CSV)
    textos = df["plot"].astype(str).tolist()
    print(f"{len(textos)} sinopses carregadas.")

    print(f"Carregando modelo SBERT ({MODELO_SBERT})...")
    modelo = carregar_modelo_sbert(MODELO_SBERT)

    print("Gerando embeddings (pode levar ~1min)...")
    matriz = gerar_embeddings_sentence(textos, modelo)

    np.save(CAMINHO_SENTENCE, matriz)
    print(f"Salvo em {CAMINHO_SENTENCE} {matriz.shape} dtype={matriz.dtype}.")


if __name__ == "__main__":
    main(forcar="--forcar" in sys.argv)
