"""Gráficas de los resultados.

    python -m dqn_carracing.plots
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = Path("results")
AZUL, NARANJA, GRIS, VERDE = "#2a78d6", "#eb6834", "#8a8a85", "#1baf7a"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
    }
)


def media_movil(x: np.ndarray, ventana: int) -> np.ndarray:
    if len(x) < ventana:
        return x
    return np.convolve(x, np.ones(ventana) / ventana, mode="valid")


def curva_de_aprendizaje() -> None:
    df = pd.read_csv(RESULTS / "entrenamiento.csv")
    ev = pd.read_csv(RESULTS / "evaluaciones.csv") if (RESULTS / "evaluaciones.csv").exists() else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    ax1.plot(df["pasos"], df["recompensa"], color=AZUL, alpha=0.18, lw=0.7)
    v = 30
    if len(df) >= v:
        ax1.plot(
            df["pasos"].to_numpy()[v - 1 :],
            media_movil(df["recompensa"].to_numpy(), v),
            color=AZUL,
            lw=2,
            label=f"media móvil ({v} episodios)",
        )
    if ev is not None and len(ev):
        ax1.errorbar(
            ev["pasos"],
            ev["media"],
            yerr=ev["desviacion"],
            fmt="o-",
            color=NARANJA,
            lw=1.8,
            ms=4,
            capsize=3,
            label="evaluación greedy (5 ep.)",
        )
    ax1.axhline(0, color=GRIS, lw=1, ls=":")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.0f} k" if v else "0"))
    ax1.set_xlabel("pasos del agente (1 paso = 4 frames del simulador)")
    ax1.set_ylabel("recompensa del episodio")
    ax1.set_title("Curva de aprendizaje")
    ax1.legend(fontsize=8, loc="upper left")

    ax2b = ax2.twinx()
    ax2.plot(df["pasos"], df["epsilon"], color=VERDE, lw=1.8, label="ε")
    ax2.set_ylabel("ε (exploración)", color=VERDE)
    ax2.tick_params(axis="y", labelcolor=VERDE)
    perdidas = df.dropna(subset=["perdida"])
    ax2b.plot(perdidas["pasos"], perdidas["perdida"], color=NARANJA, lw=1, alpha=0.8)
    ax2b.set_ylabel("pérdida de Huber (media por episodio)", color=NARANJA)
    ax2b.tick_params(axis="y", labelcolor=NARANJA)
    ax2b.set_yscale("log")
    ax2b.grid(False)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.0f} k" if v else "0"))
    ax2.set_xlabel("pasos del agente")
    ax2.set_title("Exploración y pérdida")

    fig.suptitle("DQN en CarRacing-v3 (acciones discretas)", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS / "curva_aprendizaje.png", bbox_inches="tight")
    plt.close(fig)


def comparacion_final() -> None:
    archivos = sorted(RESULTS.glob("evaluacion_*.json"))
    if not archivos:
        return
    datos = {a.stem.replace("evaluacion_", ""): json.loads(a.read_text(encoding="utf-8")) for a in archivos}
    orden = [n for n in ("aleatorio", "mejor", "ultimo") if n in datos]
    if not orden:          # solo hay evaluaciones con otros nombres: nada que comparar
        return
    etiquetas = {"aleatorio": "Política aleatoria", "mejor": "DQN (mejor checkpoint)", "ultimo": "DQN (final)"}
    colores = {"aleatorio": GRIS, "mejor": AZUL, "ultimo": NARANJA}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1, 1.2]})
    cajas = [datos[n]["recompensas"] for n in orden]
    bp = ax1.boxplot(
        cajas,
        tick_labels=[f"{etiquetas[n]}\n(n={datos[n]['episodios']})" for n in orden],
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color="black", lw=1.5),
    )
    for caja, n in zip(bp["boxes"], orden):
        caja.set_facecolor(colores[n])
        caja.set_alpha(0.75)
    ax1.axhline(0, color=GRIS, ls=":", lw=1)
    ax1.set_ylabel("recompensa del episodio")
    ax1.set_title("Evaluación final (política greedy)")

    bins = np.arange(-100, 1000, 40)
    for n in orden:
        r = np.array(datos[n]["recompensas"])
        ax2.hist(
            r,
            bins=bins,
            alpha=0.6,
            color=colores[n],
            weights=np.full(len(r), 100 / len(r)),
            label=f"{etiquetas[n]} (μ={r.mean():.0f})",
        )
    ax2.set_xlabel("recompensa del episodio")
    ax2.set_ylabel("% de los episodios")
    ax2.set_title("Distribución")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS / "comparacion.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    curva_de_aprendizaje()
    comparacion_final()
    print(f"Figuras escritas en {RESULTS}/")


if __name__ == "__main__":
    main()
