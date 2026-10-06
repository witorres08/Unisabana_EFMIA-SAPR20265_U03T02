"""Construcción del entorno y preprocesamiento de observaciones.

CarRacing-v3 entrega imágenes RGB de 96x96x3 a 50 FPS. Tal cual son demasiado
caras para una CNN en CPU y, sobre todo, una sola imagen no contiene la
velocidad del carro. Este módulo resuelve ambas cosas con cuatro pasos:

    RGB 96x96x3  --gris-->  96x96  --recorte+escala-->  84x84
                 --repetición de acción (4)-->  --apilado (4)-->  (4, 84, 84)

Cada wrapper está escrito a mano (no se usan los de `gymnasium.wrappers`) para
que el README pueda explicar exactamente qué hace cada uno.
"""
from __future__ import annotations

import cv2
import gymnasium as gym
import numpy as np

ENV_ID = "CarRacing-v3"
ALTO = ANCHO = 84          # lado de la imagen que ve la red
APILADO = 4                # frames apilados -> canales de entrada
REPETICION = 4             # cada acción se mantiene 4 frames del simulador
FRAMES_ZOOM = 50           # frames iniciales de la animación de zoom

#: Las 5 acciones discretas de CarRacing-v3, en orden.
ACCIONES = ("nada", "derecha", "izquierda", "acelerar", "frenar")


class Preprocesado(gym.ObservationWrapper):
    """RGB 96x96x3 (uint8) -> gris 84x84 (uint8).

    El recorte elimina la franja inferior de 12 píxeles, que es el panel de
    indicadores (velocímetro, ABS, volante) dibujado sobre la imagen. Para la
    red son píxeles que cambian sin aportar información espacial de la pista;
    la velocidad se recupera del apilado de frames.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 255, (ALTO, ANCHO), dtype=np.uint8)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        gris = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)      # 96x96x3 -> 96x96
        gris = gris[:84, :]                                # quita el panel inferior
        return cv2.resize(gris, (ANCHO, ALTO), interpolation=cv2.INTER_AREA)


class RepeticionDeAccion(gym.Wrapper):
    """Mantiene la misma acción durante `repeticion` frames y suma su recompensa.

    A 50 FPS, una acción suelta casi no mueve el carro: la decisión relevante
    ocurre a escala de décimas de segundo. Repetir 4 frames cuadruplica el
    horizonte efectivo del agente y divide por 4 el número de pasadas por la
    red, que es el cuello de botella en CPU.

    También corta el episodio cuando el carro acumula `paciencia` pasos
    seguidos con recompensa negativa: eso significa que está fuera de la pista
    o detenido, y los 1000 frames completos se irían en recoger -0.1 por frame.
    """

    def __init__(self, env: gym.Env, repeticion: int = REPETICION, paciencia: int = 50) -> None:
        super().__init__(env)
        self.repeticion = repeticion
        self.paciencia = paciencia
        self._negativos = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._negativos = 0
        # La cámara arranca con una animación de zoom: esos frames no son
        # estados de decisión, así que se consumen con la acción "nada".
        for _ in range(FRAMES_ZOOM // self.repeticion):
            obs, _, terminado, truncado, info = self._repetir(0)
            if terminado or truncado:
                obs, info = self.env.reset(**kwargs)
        return obs, info

    def _repetir(self, accion: int):
        total = 0.0
        terminado = truncado = False
        obs = None
        for _ in range(self.repeticion):
            obs, recompensa, terminado, truncado, info = self.env.step(accion)
            total += float(recompensa)
            if terminado or truncado:
                break
        return obs, total, terminado, truncado, info

    def step(self, accion: int):
        obs, recompensa, terminado, truncado, info = self._repetir(accion)
        self._negativos = self._negativos + 1 if recompensa < 0 else 0
        if self._negativos >= self.paciencia:
            truncado = True
            info = dict(info, corte_por_inactividad=True)
        return obs, recompensa, terminado, truncado, info


class ApiladoDeFrames(gym.Wrapper):
    """Apila los últimos `k` frames en los canales: (84,84) -> (k,84,84).

    Una imagen fija no dice si el carro va a 10 o a 80 km/h, ni si está
    derrapando. Con 4 frames consecutivos la red puede inferir velocidad y
    rotación, que es justo lo que necesita para frenar antes de una curva.
    """

    def __init__(self, env: gym.Env, k: int = APILADO) -> None:
        super().__init__(env)
        self.k = k
        self._frames = np.zeros((k, ALTO, ANCHO), dtype=np.uint8)
        self.observation_space = gym.spaces.Box(0, 255, (k, ALTO, ANCHO), dtype=np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._frames[:] = obs            # el primer frame se repite k veces
        return self._frames.copy(), info

    def step(self, accion: int):
        obs, recompensa, terminado, truncado, info = self.env.step(accion)
        self._frames[:-1] = self._frames[1:]
        self._frames[-1] = obs
        return self._frames.copy(), recompensa, terminado, truncado, info


def crear(
    *,
    render_mode: str | None = None,
    repeticion: int = REPETICION,
    paciencia: int = 50,
    apilado: int = APILADO,
) -> gym.Env:
    """CarRacing-v3 discreto con todo el preprocesamiento aplicado."""
    env = gym.make(ENV_ID, continuous=False, render_mode=render_mode)
    env = Preprocesado(env)
    env = RepeticionDeAccion(env, repeticion=repeticion, paciencia=paciencia)
    env = ApiladoDeFrames(env, k=apilado)
    return env
