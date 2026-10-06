"""Graba un episodio del agente entrenado como GIF y guarda lo que ve la red.

    python -m dqn_carracing.record --checkpoint checkpoints/mejor.pt --semilla 10000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from dqn_carracing import env as entorno
from dqn_carracing.agent import AgenteDQN

RESULTS = Path("results")


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


def main() -> None:
    p = argparse.ArgumentParser(description="Graba un episodio en GIF")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/mejor.pt"))
    p.add_argument("--semilla", type=int, default=10_000)
    p.add_argument("--salida", type=Path, default=None)
    a = p.parse_args()
    grabar(a.checkpoint, a.semilla, a.salida)


if __name__ == "__main__":
    main()
