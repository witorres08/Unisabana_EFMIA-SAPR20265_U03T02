"""Experience replay para observaciones apiladas.

Guardar cada transición como (s, s') con s de forma (4, 84, 84) uint8 cuesta
56 KB: un buffer de 100 000 transiciones pediría 5.6 GB. Como los frames se
repiten entre pasos consecutivos, aquí se guarda **un solo frame por paso** y
la pila se reconstruye al muestrear: 7 KB por transición, ocho veces menos.

Convención de índices: la posición `i` guarda el frame más reciente del estado
en el que se tomó `accion[i]`. Entonces

    estado(i)     = frames[i-k+1 .. i]
    estado'(i)    = frames[i-k+2 .. i+1]

sin cruzar el inicio de un episodio (los frames que faltan se repiten) ni la
cabeza de escritura del buffer circular (esas muestras se descartan). En un
truncamiento se guarda además, de forma dispersa, el último frame de ``s'``:
no pertenece al episodio siguiente, pero sí se necesita para bootstrappear.
"""
from __future__ import annotations

import numpy as np

ALTO = ANCHO = 84


class BufferDeFrames:
    """Buffer circular de frames sueltos con reconstrucción de pilas."""

    def __init__(self, capacidad: int, k: int = 4) -> None:
        self.capacidad = capacidad
        self.k = k
        self.frames = np.zeros((capacidad, ALTO, ANCHO), dtype=np.uint8)
        self.acciones = np.zeros(capacidad, dtype=np.int64)
        self.recompensas = np.zeros(capacidad, dtype=np.float32)
        self.terminales = np.zeros(capacidad, dtype=np.float32)  # 1.0 solo si `terminated`
        self.finales = np.zeros(capacidad, dtype=bool)           # último paso del episodio
        # Solo los truncamientos requieren un frame fuera de la secuencia normal.
        # Un dict evita reservar otros 849 MB para una matriz completa.
        self.frames_siguientes_finales: dict[int, np.ndarray] = {}
        self.pos = 0
        self.llenos = 0

    def __len__(self) -> int:
        return self.llenos

    def guardar(
        self,
        frame: np.ndarray,
        accion: int,
        recompensa: float,
        terminal: bool,
        final: bool,
        frame_siguiente_final: np.ndarray | None = None,
    ) -> None:
        """Guarda un paso.

        terminal: el episodio terminó por el entorno (choque / pista completa),
                  así que no hay valor futuro que estimar.
        final:    el episodio se acabó por cualquier razón (incluye truncado),
                  así que el frame siguiente pertenece a otro episodio.
        frame_siguiente_final: último frame de ``s'`` si el episodio fue
                  truncado. Se usa para bootstrappear sin cruzar episodios.
        """
        if final and not terminal and frame_siguiente_final is None:
            raise ValueError("un truncamiento requiere frame_siguiente_final")
        if frame_siguiente_final is not None and (not final or terminal):
            raise ValueError("frame_siguiente_final solo corresponde a un truncamiento")

        i = self.pos
        self.frames_siguientes_finales.pop(i, None)
        self.frames[i] = frame
        self.acciones[i] = accion
        self.recompensas[i] = recompensa
        self.terminales[i] = float(terminal)
        self.finales[i] = final
        if frame_siguiente_final is not None:
            self.frames_siguientes_finales[i] = frame_siguiente_final.copy()
        self.pos = (i + 1) % self.capacidad
        self.llenos = min(self.llenos + 1, self.capacidad)

    # ── reconstrucción de pilas ───────────────────────────────────────

    def _valido(self, i: int) -> bool:
        """¿Se puede formar una transición completa desde el índice i?"""
        if self.finales[i]:
            return True                       # no hace falta el frame siguiente
        return (i + 1) % self.capacidad != self.pos

    def _pila(self, i: int) -> np.ndarray:
        indices = [i]
        for _ in range(self.k - 1):
            anterior = (indices[-1] - 1) % self.capacidad
            corta_episodio = self.finales[anterior]                  # el previo cerró un episodio
            sin_historia = self.llenos < self.capacidad and indices[-1] == 0   # aún no hay dato anterior
            pisado = (                                               # el anterior ya fue sobrescrito
                self.llenos == self.capacidad
                and anterior == (self.pos - 1) % self.capacidad
            )
            if corta_episodio or sin_historia or pisado:
                indices.append(indices[-1])                          # repite el frame disponible
            else:
                indices.append(anterior)
        return self.frames[indices[::-1]]

    def _siguiente_pila(self, i: int) -> np.ndarray:
        """Reconstruye ``s'`` sin perder bootstrap en un truncamiento."""
        estado = self._pila(i)
        if not self.finales[i]:
            return self._pila((i + 1) % self.capacidad)
        if self.terminales[i]:
            return estado  # No importa: el objetivo lo anula con (1 - terminal).

        frame_siguiente = self.frames_siguientes_finales[i]
        return np.concatenate((estado[1:], frame_siguiente[None]), axis=0)

    def muestrear(self, lote: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        """(estados, acciones, recompensas, siguientes, terminales) de un lote aleatorio."""
        estados = np.empty((lote, self.k, ALTO, ANCHO), dtype=np.uint8)
        siguientes = np.empty_like(estados)
        acciones = np.empty(lote, dtype=np.int64)
        recompensas = np.empty(lote, dtype=np.float32)
        terminales = np.empty(lote, dtype=np.float32)

        n = 0
        while n < lote:
            i = int(rng.integers(0, self.llenos))
            if not self._valido(i):
                continue
            estados[n] = self._pila(i)
            siguientes[n] = self._siguiente_pila(i)
            acciones[n] = self.acciones[i]
            recompensas[n] = self.recompensas[i]
            terminales[n] = self.terminales[i]
            n += 1
        return estados, acciones, recompensas, siguientes, terminales

    def memoria_mb(self) -> float:
        return (
            self.frames.nbytes
            + self.acciones.nbytes
            + self.recompensas.nbytes
            + self.terminales.nbytes
            + self.finales.nbytes
            + sum(frame.nbytes for frame in self.frames_siguientes_finales.values())
        ) / 1e6
