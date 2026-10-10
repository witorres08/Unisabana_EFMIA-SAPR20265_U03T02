"""Prueba del buffer de frames contra una implementación ingenua.

El buffer reconstruye las pilas de 4 frames desde frames sueltos; si esa
reconstrucción está mal, el agente entrena con estados que nunca existieron y
el error es silencioso. Esta prueba compara, transición por transición, contra
una versión que guarda las pilas completas.

    python -m pytest tests/ -q        (o)        python tests/test_replay.py
"""
from __future__ import annotations

import numpy as np

from dqn_carracing.replay import BufferDeFrames


def _referencia(episodios, k=4):
    """Pilas construidas de la forma obvia: repitiendo el primer frame."""
    transiciones = []
    for frames, finales in episodios:
        pila = [frames[0]] * k
        for t, frame in enumerate(frames):
            pila = pila[1:] + [frame]
            estado = np.stack(pila)
            if t + 1 < len(frames):
                siguiente = np.stack(pila[1:] + [frames[t + 1]])
            else:
                siguiente = estado
            transiciones.append((estado, siguiente, finales[t]))
    return transiciones


def test_pilas_coinciden_con_referencia():
    rng = np.random.default_rng(0)
    largos = [5, 7, 3, 6]
    episodios = []
    for n in largos:
        frames = [np.full((84, 84), i, dtype=np.uint8) for i in rng.integers(1, 250, n)]
        finales = [t == n - 1 for t in range(n)]
        episodios.append((frames, finales))

    buffer = BufferDeFrames(capacidad=100, k=4)
    for frames, finales in episodios:
        for t, frame in enumerate(frames):
            buffer.guardar(frame, accion=t % 5, recompensa=float(t), terminal=finales[t], final=finales[t])

    esperado = _referencia(episodios)
    assert len(buffer) == len(esperado)

    for i, (estado, siguiente, _) in enumerate(esperado):
        obtenido = buffer._pila(i)
        assert np.array_equal(obtenido, estado), f"estado {i}"
        obtenido_sig = obtenido if buffer.finales[i] else buffer._pila((i + 1) % buffer.capacidad)
        assert np.array_equal(obtenido_sig, siguiente), f"siguiente {i}"


def test_muestreo_devuelve_formas_correctas():
    rng = np.random.default_rng(1)
    buffer = BufferDeFrames(capacidad=50, k=4)
    for t in range(40):
        buffer.guardar(
            np.full((84, 84), t, dtype=np.uint8),
            accion=t % 5,
            recompensa=1.0,
            terminal=(t % 10 == 9),
            final=(t % 10 == 9),
        )
    e, a, r, s, term = buffer.muestrear(16, rng)
    assert e.shape == (16, 4, 84, 84) and e.dtype == np.uint8
    assert s.shape == e.shape
    assert a.shape == (16,) and r.shape == (16,) and term.shape == (16,)
    assert set(np.unique(term)) <= {0.0, 1.0}


def test_no_mezcla_episodios():
    """El primer paso de un episodio no puede contener frames del anterior."""
    buffer = BufferDeFrames(capacidad=50, k=4)
    for t in range(6):                      # episodio A: valores 10..15
        buffer.guardar(np.full((84, 84), 10 + t, np.uint8), 0, 0.0, t == 5, t == 5)
    for t in range(4):                      # episodio B: valores 200..203
        buffer.guardar(np.full((84, 84), 200 + t, np.uint8), 0, 0.0, False, False)

    primera_de_b = buffer._pila(6)          # índice 6 = primer paso del episodio B
    assert np.all(primera_de_b == 200), "se filtraron frames del episodio anterior"


def test_truncamiento_bootstrapea_con_el_estado_siguiente():
    """Un truncamiento usa s', aunque no pueda ocupar el siguiente índice del buffer."""
    buffer = BufferDeFrames(capacidad=10, k=4)
    buffer.guardar(np.full((84, 84), 10, np.uint8), 0, 0.0, False, False)
    buffer.guardar(
        np.full((84, 84), 20, np.uint8),
        0,
        0.0,
        terminal=False,
        final=True,
        frame_siguiente_final=np.full((84, 84), 30, np.uint8),
    )

    siguiente = buffer._siguiente_pila(1)
    assert np.all(siguiente == np.array([10, 10, 20, 30])[:, None, None])


if __name__ == "__main__":
    test_pilas_coinciden_con_referencia()
    test_muestreo_devuelve_formas_correctas()
    test_no_mezcla_episodios()
    print("Todas las pruebas pasaron.")
