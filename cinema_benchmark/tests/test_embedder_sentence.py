import types

import numpy as np

from packages.embedder_sentence import (
    gerar_embedding_query,
    gerar_embeddings_sentence,
)

def _fake_modelo():
    return types.SimpleNamespace(
        encode=lambda textos, **kw: (
            np.ones(4, dtype=np.float32)
            if isinstance(textos, str)
            else np.ones((len(textos), 4), dtype=np.float32)
        )
    )

def test_gerar_embeddings_sentence_shape():
    matriz = gerar_embeddings_sentence(["a", "b", "c"], _fake_modelo())
    assert matriz.shape == (3, 4)

def test_gerar_embedding_query_1d():
    vetor = gerar_embedding_query("oi", _fake_modelo())
    assert vetor.ndim == 1
    assert vetor.shape == (4,)
