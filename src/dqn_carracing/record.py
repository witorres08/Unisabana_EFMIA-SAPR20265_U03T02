"""Graba un episodio del agente entrenado como GIF y guarda lo que ve la red.

    dqn-carracing record --checkpoint runs/dqn/corrida_001/checkpoints/mejor.pt --semilla 10000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from dqn_carracing import env as entorno
from dqn_carracing.agent import AgenteDQN
from dqn_carracing.paths import CORRIDA_POR_DEFECTO

RESULTS = CORRIDA_POR_DEFECTO.resultados


def grabar(checkpoint: Path, semilla: int = 10_000, salida: Path | None = None) -> float:
    agente = AgenteDQN.cargar(checkpoint)
    env = entorno.crear(render_mode="rgb_array", paciencia=100)
    obs, _ = env.reset(seed=semilla)
    cuadros, total, fin, pasos = [], 0.0, False, 0
    ultima_pila = obs
    while not fin:
        accion = agente.seleccionar_accion(obs, deterministico=True)
        obs, recompensa, terminado, truncado, _ = env.step(accion)
        cuadros.append(env.render())
        total += recompensa
        pasos += 1
        ultima_pila = obs
        fin = terminado or truncado
    env.close()

    RESULTS.mkdir(exist_ok=True)
    salida = salida or RESULTS / "agente.gif"
    # 1 paso del agente = 4 frames a 50 FPS -> 12.5 pasos/s de tiempo real
    imageio.mimsave(salida, [c[::2, ::2] for c in cuadros], duration=1 / 12.5, loop=0)

    # Lo que realmente entra a la red: la pila de 4 frames en gris 84x84.
    tira = np.concatenate(list(ultima_pila), axis=1)
    imageio.imwrite(RESULTS / "observacion_apilada.png", tira)
    print(f"{salida}: recompensa {total:.1f} en {pasos} pasos ({len(cuadros)} cuadros)")
    return total


def agregar_argumentos(p: argparse.ArgumentParser) -> None:
    """Registra las opciones del subcomando ``record`` en un parser."""
    p.add_argument("--checkpoint", type=Path, default=CORRIDA_POR_DEFECTO.checkpoints / "mejor.pt")
    p.add_argument("--semilla", type=int, default=10_000)
    p.add_argument("--salida", type=Path, default=None)


def ejecutar(a: argparse.Namespace) -> None:
    """Graba el episodio solicitado por el CLI."""
    grabar(a.checkpoint, a.semilla, a.salida)


def main() -> None:
    """Punto de entrada compatible con ``python -m dqn_carracing.record``."""
    parser = argparse.ArgumentParser(description="Graba un episodio en GIF")
    agregar_argumentos(parser)
    ejecutar(parser.parse_args())


if __name__ == "__main__":
    main()
