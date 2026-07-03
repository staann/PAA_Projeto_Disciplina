

Projeto e análise de algoritmos 1/2026
Projeto de Disciplina:

O projeto de disciplina é fazer um sistema onde o usuário faz uma pergunta em linguagem
natural e o sistema responde dando a resposta formatada.

O banco de dados a ser utilizado é o “CMU Movie Summary Corpus”.
Quando usuário faz uma pergunta sobre filmes, o sistema faz uma busca semântica nesta
base de dados.  A busca semântica deve ser feita com busca por similaridade de cosseno,
“Word2Vec Average”, “Sentence Embeddings”, “HNSW Search” e verificar qual é o
melhor em desempenho e complexidade.

Uma vez encontradas as sinopses por busca semântica, a resposta é formatada com um
LLM local mais simples como SmolLM, TinyLlama, Phi-3 Mini, Mistral ou outra.

O sistema deverá responder perguntas como qual é o filme baseado em uma descrição,
recomendar filmes baseado em uma descrição, entre outros.

Este é um problema muito complexo e o objetivo é fazer o melhor possível com os
recursos disponíveis.

Deve ser feita uma apresentação prática

