import nltk
#import gensim
'''
nltk.download('stopwords')

from nltk.corpus import stopwords

stop_words = list(set(stopwords.words('english')))



frase_teste = 'she is very handsome and more funnier'
nova_frase = ''

for palavra in frase_teste.split():
    if palavra not in stop_words:

        nova_frase += palavra + ' '

print(nova_frase)
'''

'''
from gensim.models import KeyedVectors

# Carrega o modelo binário baixado
modelo = KeyedVectors.load_word2vec_format(
    "cinema_benchmark/modelos/modeloW2V.bin", binary=True
)
'''

import gensim.downloader as api


def carregar_modelo_word2vec():
    """Baixa (se não estiver no PC) e carrega o Word2Vec do Google News de 300 dimensões."""
    print("Carregando o modelo Word2Vec (Google News 300)...")
    print(
        "Nota: Se for a primeira vez, o Gensim fará o download de ~1.6GB automaticamente."
    )

    # O nome oficial do modelo no repositório do Gensim é exatamente este:
    modelo = api.load("word2vec-google-news-300")

    print("Modelo carregado com sucesso!")
    return modelo

word2vec = carregar_modelo_word2vec()