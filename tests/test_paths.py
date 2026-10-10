from pathlib import Path

from dqn_carracing.paths import RutasCorrida, seleccionar_rutas


def test_primera_corrida_usa_las_rutas_raiz(tmp_path: Path):
    raiz = tmp_path / "runs" / "dqn"
    rutas = seleccionar_rutas(False, corridas_raiz=raiz)

    assert rutas == RutasCorrida(raiz / "corrida_001")


def test_corrida_nueva_no_mezcla_resultados_previos(tmp_path: Path):
    corridas = tmp_path / "runs" / "dqn"
    (corridas / "corrida_001").mkdir(parents=True)
    (corridas / "corrida_001" / "entrenamiento.csv").touch()
    (corridas / "corrida_002").mkdir()

    rutas = seleccionar_rutas(False, corridas_raiz=corridas)

    assert rutas == RutasCorrida(corridas / "corrida_003")
