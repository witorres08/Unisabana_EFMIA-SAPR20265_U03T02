"""Hiperparámetros del agente DQN de CarRacing."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    """Valores de configuración del entrenamiento DQN.

    Cada valor se justifica en el README. Mantenerlos aquí separa la
    configuración del comportamiento del agente y facilita crear experimentos
    reproducibles desde el CLI.
    """

    gamma: float = 0.99
    lr: float = 1e-4
    lote: int = 32
    capacidad_buffer: int = 120_000
    pasos_de_calentamiento: int = 5_000
    frecuencia_entrenamiento: int = 1
    frecuencia_target: int = 2_000
    epsilon_inicial: float = 1.0
    epsilon_final: float = 0.05
    pasos_de_decaimiento: int = 120_000
    doble_dqn: bool = True
    # Orden: nada, derecha, izquierda, acelerar, frenar.
    pesos_exploracion: tuple[float, ...] = (0.10, 0.20, 0.20, 0.45, 0.05)
    recorte_recompensa: float = 5.0
    semilla: int = 0
