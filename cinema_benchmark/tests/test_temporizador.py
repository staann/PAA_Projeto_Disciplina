from packages.temporizador import medir_tempo

def test_retorna_resultado_e_tempo():
    resultado, tempo = medir_tempo(lambda x: x + 1, 41)

    assert resultado == 42
    assert isinstance(tempo, float)
    assert tempo >= 0.0

def test_preserva_kwargs():
    resultado, tempo = medir_tempo(lambda a, b=0: a + b, 1, b=2)

    assert resultado == 3
    assert tempo >= 0.0
