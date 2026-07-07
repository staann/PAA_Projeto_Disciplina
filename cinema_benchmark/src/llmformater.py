def montar_prompt(pergunta, filmes):

    contexto_filmes = ""

    for i, filme in enumerate(filmes):
        rotulo = " (BEST MATCH)" if i == 0 else ""
        contexto_filmes += f"\n--- MOVIE {i+1}{rotulo} ---"
        contexto_filmes += f"\nTitle: {filme['titulo']}"
        contexto_filmes += f"\nSynopsis: {filme['sinopse']}"

    titulo_top = filmes[0]["titulo"] if filmes else ""

    prompt = f"""You are a movie recommendation assistant. The list below contains movies already retrieved from the database and RANKED by semantic similarity to what the user is looking for. MOVIE 1 is the top match.

        RETRIEVED MOVIES (ranked, most similar first):
        {contexto_filmes}

        USER IS LOOKING FOR:
        "{pergunta}"

        RULES:
        1. Recommend the best matching movie from the list above. The top match is "{titulo_top}" (MOVIE 1); prefer it unless another listed movie is a clearly stronger fit for the description.
        2. Start your answer by naming the recommended movie, then briefly justify it in 1-2 sentences using ONLY its synopsis.
        3. Base your answer STRICTLY on the movies provided. Do not invent movies or facts.
        4. Do not add conversational filler. Never reply that there is "not enough information": always recommend the most relevant movie from the ranked list.

        ANSWER:"""

    return prompt


def formatar_resposta_llm(pergunta,filmes,modelo='qwen2.5:1.5b'):

    prompt = montar_prompt(pergunta,filmes)

    try:
        import ollama

        resposta = ollama.chat(model=modelo, messages=[{"role": "user", "content": prompt}])

        return resposta["message"]["content"]
    
    except Exception as e:
        return f"[Erro ao nos comunicar com o Ollama]: {e}\nCertifique-se de que o Ollama está rodando em segundo plano e que o modelo '{modelo}' foi baixado."