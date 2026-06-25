import ollama

def montar_prompt(pergunta,filmes):

    contexto_filmes = ""

    for i,filmes in enumerate(filmes):
        contexto_filmes += f"\n--- MOVIE {i+1} ---"
        contexto_filmes += f"\nTitle: {filmes['titulo']}"
        contexto_filmes += f"\nSynopsis: {filmes['sinopse']}"

    prompt = f"""You are a movie expert assistant. Your task is to answer the user's question based ESTRICLY on the movie context provided below.

        MOVIE CONTEXT:
        {contexto_filmes}

        USER QUESTION: 
        "{pergunta}"

        RULES:
        1. Provide a direct, clear, and well-formatted answer to the user's question using the context.
        2. Do not provide any conversational filler or explanations outside of the answer.
        3. If the answer cannot be determined or deduced from the provided movie context, reply EXACTLY with: "Not enough information in the database to determine the answer."

        ANSWER:"""

    return prompt


def formatar_resposta_llm(pergunta,filmes,modelo='qwen2.5:1.5b'):

    prompt = montar_prompt(pergunta,filmes)

    try:
        resposta = ollama.chat(model=modelo, messages=[{"role": "user", "content": prompt}])

        return resposta["message"]["content"]
    
    except Exception as e:
        return f"[Erro ao nos comunicar com o Ollama]: {e}\nCertifique-se de que o Ollama está rodando em segundo plano e que o modelo '{modelo}' foi baixado."