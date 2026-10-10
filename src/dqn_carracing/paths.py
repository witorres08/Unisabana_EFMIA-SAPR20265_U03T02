"""Rutas de salida de una corrida de entrenamiento DQN."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


CORRIDAS_RAIZ = Path("runs") / "dqn"
NOMBRE_CORRIDA_PRINCIPAL = "corrida_001"


@dataclass(frozen=True)
class RutasCorrida:
    """Directorio autocontenido de una corrida y sus checkpoints."""

    raiz: Path

    @property
    def resultados(self) -> Path:
        """Directorio de métricas, configuración, figuras y grabaciones."""
        return self.raiz

    @property
    def checkpoints(self) -> Path:
        """Directorio de pesos guardados para esta misma corrida."""
        return self.raiz / "checkpoints"

    def preparar(self) -> None:
        """Crea los directorios de salida si aún no existen."""
        self.resultados.mkdir(parents=True, exist_ok=True)
        self.checkpoints.mkdir(parents=True, exist_ok=True)


def seleccionar_rutas(
    reanudar: bool,
    *,
    corridas_raiz: Path = CORRIDAS_RAIZ,
) -> RutasCorrida:
    """Elige una salida sin mezclar una corrida nueva con una anterior."""
    principal = RutasCorrida(corridas_raiz / NOMBRE_CORRIDA_PRINCIPAL)
    if reanudar or not (principal.resultados / "entrenamiento.csv").exists():
        return principal

    numero = 2
    while (corridas_raiz / f"corrida_{numero:03d}").exists():
        numero += 1
    rutas = RutasCorrida(corridas_raiz / f"corrida_{numero:03d}")
    print(
        f"[aviso] ya existe {principal.resultados}/entrenamiento.csv de una corrida anterior.\n"
        f"        Esta corrida escribe en {rutas.raiz}/ para no mezclarlas.\n"
        "        Usa --reanudar si lo que querías era continuar la corrida previa."
    )
    return rutas


CORRIDA_POR_DEFECTO = RutasCorrida(CORRIDAS_RAIZ / NOMBRE_CORRIDA_PRINCIPAL)
