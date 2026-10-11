"""CLI unificado para entrenar y analizar el agente DQN."""
from __future__ import annotations

import argparse
from collections.abc import Sequence

from dqn_carracing import evaluate, record, train


def _ejecutar_plot(args: argparse.Namespace) -> None:
    """Importa matplotlib/pandas solo cuando se solicitan gráficas."""
    from dqn_carracing import plots

    plots.ejecutar(args)


def crear_parser() -> argparse.ArgumentParser:
    """Construye el parser raíz y sus cuatro subcomandos."""
    parser = argparse.ArgumentParser(prog="dqn-carracing", description="DQN para CarRacing-v3")
    subcomandos = parser.add_subparsers(dest="comando", required=True)

    entrenamiento = subcomandos.add_parser("train", help="entrena un agente")
    train.agregar_argumentos(entrenamiento)
    entrenamiento.set_defaults(ejecutar=train.ejecutar)

    evaluacion = subcomandos.add_parser("evaluate", help="evalúa un checkpoint o baseline")
    evaluate.agregar_argumentos(evaluacion)
    evaluacion.set_defaults(ejecutar=evaluate.ejecutar)

    grabacion = subcomandos.add_parser("record", help="graba un episodio como GIF")
    record.agregar_argumentos(grabacion)
    grabacion.set_defaults(ejecutar=record.ejecutar)

    graficas = subcomandos.add_parser("plot", help="genera las gráficas de resultados")
    graficas.set_defaults(ejecutar=_ejecutar_plot)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Ejecuta el subcomando solicitado; admite argumentos para facilitar pruebas."""
    args = crear_parser().parse_args(argv)
    args.ejecutar(args)


if __name__ == "__main__":
    main()
