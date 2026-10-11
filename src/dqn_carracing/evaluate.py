"""Evaluación de una política, sin aprendizaje ni exploración.

    dqn-carracing evaluate --checkpoint runs/dqn/corrida_001/checkpoints/mejor.pt --episodios 100
    python -m dqn_carracing.evaluate --aleatorio --episodios 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dqn_carracing import env as entorno
from dqn_carracing.paths import CORRIDA_POR_DEFECTO


def evaluar(
    agente,
    episodios: int = 10,
    semilla_base: int = 10_000,
    paciencia: int = 100,
    devolver_detalle: bool = False,
):
    """Corre `episodios` con la política greedy sobre semillas fijas.

    Las mismas semillas para todos los agentes: cada semilla genera una pista
    distinta, así que comparar sobre pistas distintas no diría nada.
    """
    env = entorno.crear(paciencia=paciencia)
    recompensas, longitudes = [], []
    for i in range(episodios):
        obs, _ = env.reset(seed=semilla_base + i)
        total, pasos, fin = 0.0, 0, False
        while not fin:
            accion = (
                int(np.random.default_rng(semilla_base + i + pasos).integers(5))
                if agente is None
                else agente.seleccionar_accion(obs, deterministico=True)
            )
            obs, recompensa, terminado, truncado, _ = env.step(accion)
            total += recompensa
            pasos += 1
            fin = terminado or truncado
        recompensas.append(total)
        longitudes.append(pasos)
    env.close()
    if devolver_detalle:
        return np.array(recompensas), np.array(longitudes)
    return np.array(recompensas)


def agregar_argumentos(p: argparse.ArgumentParser) -> None:
    """Registra las opciones del subcomando ``evaluate`` en un parser."""
    p.add_argument("--checkpoint", type=Path, default=CORRIDA_POR_DEFECTO.checkpoints / "mejor.pt")
    p.add_argument("--episodios", type=int, default=100)
    p.add_argument("--aleatorio", action="store_true", help="política aleatoria (baseline)")
    p.add_argument("--salida", type=Path, default=None)


def ejecutar(a: argparse.Namespace) -> None:
    """Evalúa el checkpoint o la política aleatoria solicitada por el CLI."""
    if a.aleatorio:
        agente, nombre = None, "aleatorio"
    else:
        from dqn_carracing.agent import AgenteDQN

        agente = AgenteDQN.cargar(a.checkpoint)
        nombre = a.checkpoint.stem

    recompensas, longitudes = evaluar(agente, a.episodios, devolver_detalle=True)
    resumen = {
        "agente": nombre,
        "episodios": int(a.episodios),
        "media": float(recompensas.mean()),
        "desviacion": float(recompensas.std()),
        "mediana": float(np.median(recompensas)),
        "peor": float(recompensas.min()),
        "mejor": float(recompensas.max()),
        "pasos_medios": float(longitudes.mean()),
        "recompensas": [float(x) for x in recompensas],
    }
    print(
        f"{nombre}: {resumen['media']:.1f} ± {resumen['desviacion']:.1f} "
        f"(mediana {resumen['mediana']:.1f}, peor {resumen['peor']:.1f}, "
        f"mejor {resumen['mejor']:.1f}) sobre {a.episodios} episodios"
    )
    salida = a.salida or CORRIDA_POR_DEFECTO.resultados / f"evaluacion_{nombre}.json"
    salida.parent.mkdir(exist_ok=True)
    salida.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    """Punto de entrada compatible con ``python -m dqn_carracing.evaluate``."""
    parser = argparse.ArgumentParser(description="Evalúa un agente en CarRacing-v3")
    agregar_argumentos(parser)
    ejecutar(parser.parse_args())


if __name__ == "__main__":
    main()
