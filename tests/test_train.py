from pathlib import Path

import pytest

from dqn_carracing.config import Config
from dqn_carracing.paths import RutasCorrida
from dqn_carracing.train import entrenar


def test_reanudar_sin_ultimo_checkpoint_falla(tmp_path: Path):
    rutas = RutasCorrida(tmp_path / "corrida_001")

    with pytest.raises(FileNotFoundError, match="Inicia una corrida nueva"):
        entrenar(1, Config(), rutas, reanudar=True)

    assert not rutas.raiz.exists()
