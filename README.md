# Cinema Benchmark - Sistema de Busca de Filmes

Projeto de disciplina para busca de filmes em linguagem natural com base em similaridade de cosseno entre embeddings Word2Vec.

## Visão geral

O usuário digita uma pergunta sobre filmes, o sistema processa o texto e retorna os filmes mais similares usando os dados do corpus `CMU Movie Summary Corpus`.

## Tecnologias usadas

- `Python`
- `NumPy`
- `pandas`
- `scikit-learn`
- `NLTK`
- `Kaggle Hub`

## Estrutura do projeto

```text
cinema_benchmark/
├── data/
│   ├── cmu-movie-summary-corpus/
│   └── tratada/
├── modelos/
└── src/
    ├── download_data.py
    ├── tratar_data.py
    ├── salvar_data_tratada.py
    ├── aplicar_modelos.py
    ├── modelos_embeddings.py
    ├── motor_busca.py
    └── verificar_caminho.py
```

## Pré-requisitos

- `Python 3.7` ou superior
- Conexão com a internet para baixar o dataset
- Conta no Kaggle com acesso à API

## Instalação

1. Crie e ative um ambiente virtual, se desejar:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Instale as dependências:

```powershell
pip install -r requirements.txt
```

3. Configure o Kaggle API Token:

   - Baixe o arquivo `kaggle.json` na sua conta Kaggle
   - Coloque o arquivo em `C:\Users\<seu_usuario>\.kaggle\kaggle.json`
   - Ou defina a variável de ambiente `KAGGLE_API_TOKEN`

## Como executar

### 1. Baixar os dados

Execute o script de download dentro da pasta `src`:

```powershell
cd cinema_benchmark\src
python download_data.py
```

Esse passo baixa o corpus para `cinema_benchmark/data/cmu-movie-summary-corpus/`.

### 2. Processar os dados

Depois, rode o pipeline de tratamento:

```powershell
python salvar_data_tratada.py
```

Esse script gera os arquivos processados em `cinema_benchmark/data/tratada/`, incluindo:

- `filmes_processados.csv`
- `embeddings_word2vec.npy`

### 3. Fazer buscas

Execute o motor de busca interativo:

```powershell
python motor_busca.py
```

Digite uma pergunta relacionada a filmes e o sistema exibirá os 3 resultados mais similares com título, score e trecho da sinopse.

## Exemplo de uso

```text
Digite sua pergunta relacionada aos filmes: filmes de aventura e exploração

FILMES ENCONTRADOS

1º Lugar: ...
Grau de Similaridade: 0.8234
Trecho da Sinopse: ...
```

## Observações

- O projeto já está implementado até a etapa de busca por similaridade
- A qualidade dos resultados depende dos dados processados e dos embeddings gerados
- Se algum arquivo de entrada não existir, verifique se os passos anteriores foram executados corretamente

## Scripts principais

- `download_data.py`: baixa o dataset original
- `salvar_data_tratada.py`: executa o processamento e salva os dados tratados
- `motor_busca.py`: realiza a busca interativa por similaridade

## Próximos passos

Possíveis melhorias futuras:

- ajustar o pré-processamento dos textos
- salvar e reutilizar o modelo de embeddings treinado
