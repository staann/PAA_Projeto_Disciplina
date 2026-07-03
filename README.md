# Cinema Benchmark

Sistema de busca semântica para filmes usando o **CMU Movie Summary Corpus**.
O projeto atende ao trabalho de disciplina de *Projeto e Análise de Algoritmos* e compara abordagens de busca para responder perguntas em linguagem natural.

## Objetivo

Dada uma pergunta sobre filmes, o sistema busca sinopses parecidas e monta uma resposta final com apoio de um LLM local via `Ollama`.

## Modelos implementados

- **Word2Vec Average + Cosseno**: representa cada sinopse pela média dos vetores das palavras e faz busca por similaridade de cosseno.
- **Sentence Embeddings + Cosseno**: gera embeddings de sentenças com `SentenceTransformer` e faz busca por cosseno.
- **Sentence Embeddings + HNSW**: usa `SentenceTransformer` + `FAISS HNSW` para acelerar a busca aproximada.
- **Resposta final com LLM local**: formata a resposta usando `Ollama`.

## Estrutura principal

```text
cinema_benchmark/
├── data/
│   ├── cmu-movie-summary-corpus/
│   └── tratada/
├── src/
│   ├── aplicar_modelos.py
│   ├── download_data.py
│   ├── llmformater.py
│   ├── modelos_embeddings.py
│   ├── motor_busca.py
│   ├── salvar_data_tratada.py
│   ├── tratar_data.py
│   ├── sentence_cosseno/
│   └── sentence_hnsw/
└── tests/
```

## Requisitos

- `Python 3.11+` recomendado
- `pip`
- Conexão com a internet para baixar o corpus e os modelos na primeira execução
- `Ollama` instalado localmente, se você quiser gerar a resposta final com LLM

## Instalação

Crie e ative um ambiente virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```powershell
pip install -r requirements.txt
```

Se o `NLTK` pedir dados adicionais, baixe o pacote de stopwords na primeira execução ou rode:

```powershell
python -c "import nltk; nltk.download('stopwords')"
```

## Preparação dos dados

1. Baixar o corpus do CMU Movie Summary Corpus:

```powershell
python cinema_benchmark\src\download_data.py
```

2. Processar e salvar os dados tratados:

```powershell
python cinema_benchmark\src\salvar_data_tratada.py
```

Isso gera `cinema_benchmark/data/tratada/filmes_processados.csv`.

## Como executar cada modelo

### 1) Word2Vec Average + Cosseno

Gerar a matriz de embeddings Word2Vec:

```powershell
python cinema_benchmark\src\aplicar_modelos.py
```

Executar a busca interativa:

```powershell
python cinema_benchmark\src\motor_busca.py
```

No menu, escolha `1`.

### 2) Sentence Embeddings + Cosseno

Executar o pipeline completo com uma pergunta de exemplo:

```powershell
python cinema_benchmark\src\sentence_cosseno\pipeline_sentence_cosseno.py "a movie about adventure and exploration"
```

Esse pipeline constrói o índice, faz a busca e salva métricas.

### 3) Sentence Embeddings + HNSW

Gerar o índice HNSW:

```powershell
python cinema_benchmark\src\sentence_hnsw\pipeline_hnsw.py
```

Depois usar o menu principal:

```powershell
python cinema_benchmark\src\motor_busca.py
```

No menu, escolha `2`.

## LLM local

O arquivo `cinema_benchmark/src/llmformater.py` usa `Ollama` para formatar a resposta.

Exemplo de uso:

```powershell
ollama run qwen2.5:1.5b
```

Se preferir outro modelo local, ajuste o nome no código ou passe outro valor no parâmetro `modelo`.

## Testes

Executar a suíte de testes:

```powershell
python -m pytest -q
```

Se o `pytest` não estiver instalado, reinstale as dependências com `pip install -r requirements.txt`.

## Observações

- Os modelos `SentenceTransformer` e `FAISS` podem demorar na primeira execução por causa do download.
- O `Word2Vec` do Gensim também pode baixar um modelo grande na primeira vez.
- O arquivo `motor_busca.py` integra a busca e a resposta final com LLM local.

## Licença

Uso acadêmico para fins de disciplina.
