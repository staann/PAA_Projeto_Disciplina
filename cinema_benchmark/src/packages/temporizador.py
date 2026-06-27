import time

def medir_tempo(funcao, *args, **kwargs):
    inicio = time.perf_counter()
    resultado = funcao(*args, **kwargs)
    tempo = time.perf_counter() - inicio
    return resultado, tempo
