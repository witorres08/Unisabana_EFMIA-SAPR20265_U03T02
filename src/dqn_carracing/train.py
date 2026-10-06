"""Bucle de entrenamiento.

    python -m dqn_carracing.train --pasos 400000

Registra una fila por episodio en results/entrenamiento.csv, evalúa la política
greedy cada `--eval-cada` pasos y guarda checkpoints (el último y el mejor).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from dqn_carracing import env as entorno
from dqn_carracing.agent import AgenteDQN, Config
from dqn_carracing.evaluate import evaluar

RESULTS = Path("results")
CKPT = Path("checkpoints")


def entrenar(
    pasos_totales: int,
    cfg: Config,
    eval_cada: int = 25_000,
    eval_episodios: int = 5,
    reanudar: bool = False,
) -> None:
    torch.set_num_threads(2)
    RESULTS.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)

    env = entorno.crear()
    agente = (
        AgenteDQN.cargar(CKPT / "ultimo.pt")
        if reanudar and (CKPT / "ultimo.pt").exists()
        else AgenteDQN(n_acciones=env.action_space.n, cfg=cfg)
    )
    print(
        f"Dispositivo: {agente.dispositivo} | parámetros: {agente.q.n_parametros:,} | "
        f"buffer: {agente.buffer.memoria_mb():.0f} MB | acciones: {entorno.ACCIONES}"
    )

    csv_path = RESULTS / "entrenamiento.csv"
    nuevo = not csv_path.exists()
    f_csv = open(csv_path, "a", newline="", encoding="utf-8")
    escritor = csv.writer(f_csv)
    if nuevo:
        escritor.writerow(
            ["episodio", "pasos", "recompensa", "longitud", "epsilon", "perdida", "segundos"]
        )

    eval_path = RESULTS / "evaluaciones.csv"
    nuevo_eval = not eval_path.exists()
    f_eval = open(eval_path, "a", newline="", encoding="utf-8")
    escritor_eval = csv.writer(f_eval)
    if nuevo_eval:
        escritor_eval.writerow(["pasos", "media", "desviacion", "episodios"])

    mejor_eval = -1e9
    episodio = 0
    t0 = time.time()
    proxima_eval = agente.pasos + eval_cada

    while agente.pasos < pasos_totales:
        episodio += 1
        obs, _ = env.reset(seed=int(agente.rng.integers(1_000_000)))
        total, longitud, perdidas = 0.0, 0, []
        fin = False
        primer_paso = True

        while not fin:
            accion = agente.seleccionar_accion(obs)
            siguiente, recompensa, terminado, truncado, _ = env.step(accion)
            fin = terminado or truncado

            # El buffer guarda el frame más reciente del estado actual.
            agente.buffer.guardar(
                frame=obs[-1],
                accion=accion,
                recompensa=float(
                    np.clip(recompensa, -cfg.recorte_recompensa, cfg.recorte_recompensa)
                ),
                terminal=terminado,          # truncado NO corta el bootstrapping
                final=fin,
            )
            agente.pasos += 1
            if agente.pasos % cfg.frecuencia_entrenamiento == 0:
                p = agente.aprender()
                if p is not None:
                    perdidas.append(p)

            obs = siguiente
            total += recompensa
            longitud += 1
            primer_paso = False

        escritor.writerow(
            [
                episodio,
                agente.pasos,
                round(total, 2),
                longitud,
                round(agente.epsilon, 4),
                round(float(np.mean(perdidas)), 5) if perdidas else "",
                round(time.time() - t0, 1),
            ]
        )
        f_csv.flush()

        if episodio % 10 == 0:
            print(
                f"ep {episodio:4d} | pasos {agente.pasos:7,} | recompensa {total:7.1f} | "
                f"largo {longitud:4d} | eps {agente.epsilon:.3f} | "
                f"perdida {np.mean(perdidas) if perdidas else float('nan'):.4f} | "
                f"{agente.pasos / (time.time() - t0):.0f} pasos/s",
                flush=True,
            )

        if agente.pasos >= proxima_eval:
            proxima_eval += eval_cada
            recompensas = evaluar(agente, episodios=eval_episodios, semilla_base=5_000)
            media, de = float(np.mean(recompensas)), float(np.std(recompensas))
            escritor_eval.writerow([agente.pasos, round(media, 2), round(de, 2), eval_episodios])
            f_eval.flush()
            print(f"  [eval] pasos {agente.pasos:,}: {media:.1f} ± {de:.1f}", flush=True)
            if media > mejor_eval:
                mejor_eval = media
                agente.guardar(CKPT / "mejor.pt")
            agente.guardar(CKPT / "ultimo.pt")

    agente.guardar(CKPT / "ultimo.pt")
    env.close()
    f_csv.close()
    f_eval.close()
    (RESULTS / "config.json").write_text(
        json.dumps(
            {
                "pasos_totales": agente.pasos,
                "actualizaciones": agente.actualizaciones,
                "episodios": episodio,
                "minutos": round((time.time() - t0) / 60, 1),
                "parametros": agente.q.n_parametros,
                "cfg": vars(cfg),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Listo: {agente.pasos:,} pasos en {(time.time() - t0) / 60:.1f} min")


def main() -> None:
    p = argparse.ArgumentParser(description="Entrena DQN en CarRacing-v3 discreto")
    p.add_argument("--pasos", type=int, default=400_000)
    p.add_argument("--eval-cada", type=int, default=25_000)
    p.add_argument("--eval-episodios", type=int, default=5)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--semilla", type=int, default=Config.semilla)
    p.add_argument("--sin-doble", action="store_true", help="DQN clásico en vez de Double DQN")
    p.add_argument(
        "--exploracion-uniforme",
        action="store_true",
        help="ε-greedy uniforme en vez de sesgada hacia el acelerador",
    )
    p.add_argument("--reanudar", action="store_true")
    a = p.parse_args()

    cfg = Config(lr=a.lr, semilla=a.semilla, doble_dqn=not a.sin_doble)
    if a.exploracion_uniforme:
        cfg.pesos_exploracion = (1.0,) * 5
    entrenar(a.pasos, cfg, a.eval_cada, a.eval_episodios, a.reanudar)


if __name__ == "__main__":
    main()
