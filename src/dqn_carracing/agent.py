"""El agente DQN: política ε-greedy, paso de Bellman y persistencia."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dqn_carracing.config import Config
from dqn_carracing.model import RedQ
from dqn_carracing.replay import BufferDeFrames


class AgenteDQN:
    """DQN con replay buffer, red target y ε-greedy lineal."""

    def __init__(self, n_acciones: int = 5, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self.n_acciones = n_acciones
        torch.manual_seed(self.cfg.semilla)
        self.rng = np.random.default_rng(self.cfg.semilla)

        self.dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q = RedQ(n_acciones=n_acciones).to(self.dispositivo)
        self.q_target = RedQ(n_acciones=n_acciones).to(self.dispositivo)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.opt = torch.optim.Adam(self.q.parameters(), lr=self.cfg.lr)
        self.perdida_fn = nn.SmoothL1Loss()   # Huber: menos sensible a targets atípicos
        self.buffer = BufferDeFrames(self.cfg.capacidad_buffer)

        self.pasos = 0              # pasos de entorno vistos
        self.actualizaciones = 0    # pasos de gradiente aplicados

    # ── política ──────────────────────────────────────────────────────

    @property
    def epsilon(self) -> float:
        c = self.cfg
        avance = min(1.0, self.pasos / c.pasos_de_decaimiento)
        return c.epsilon_inicial + avance * (c.epsilon_final - c.epsilon_inicial)

    def seleccionar_accion(self, estado: np.ndarray, *, deterministico: bool = False) -> int:
        if not deterministico and self.rng.random() < self.epsilon:
            pesos = np.asarray(self.cfg.pesos_exploracion[: self.n_acciones], dtype=float)
            return int(self.rng.choice(self.n_acciones, p=pesos / pesos.sum()))
        with torch.no_grad():
            x = torch.from_numpy(estado).unsqueeze(0).to(self.dispositivo)  # (1,4,84,84)
            return int(self.q(x).argmax(dim=1).item())

    # ── aprendizaje ───────────────────────────────────────────────────

    def aprender(self) -> float | None:
        """Un paso de gradiente sobre la ecuación de Bellman. None si aún no toca."""
        c = self.cfg
        if len(self.buffer) < c.pasos_de_calentamiento:
            return None

        estados, acciones, recompensas, siguientes, terminales = self.buffer.muestrear(
            c.lote, self.rng
        )
        d = self.dispositivo
        estados_t = torch.from_numpy(estados).to(d)
        siguientes_t = torch.from_numpy(siguientes).to(d)
        acciones_t = torch.from_numpy(acciones).to(d).unsqueeze(1)
        recompensas_t = torch.from_numpy(recompensas).to(d).unsqueeze(1)
        terminales_t = torch.from_numpy(terminales).to(d).unsqueeze(1)

        # Q(s, a) de las acciones efectivamente tomadas.
        q_actual = self.q(estados_t).gather(1, acciones_t)

        with torch.no_grad():
            if c.doble_dqn:
                # La red online ELIGE la mejor acción; la target la EVALÚA.
                mejor = self.q(siguientes_t).argmax(dim=1, keepdim=True)
                q_siguiente = self.q_target(siguientes_t).gather(1, mejor)
            else:
                q_siguiente = self.q_target(siguientes_t).max(dim=1, keepdim=True).values
            objetivo = recompensas_t + c.gamma * q_siguiente * (1.0 - terminales_t)

        perdida = self.perdida_fn(q_actual, objetivo)
        self.opt.zero_grad()
        perdida.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        self.actualizaciones += 1
        if self.actualizaciones % c.frecuencia_target == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return float(perdida.item())

    # ── persistencia ──────────────────────────────────────────────────

    def guardar(self, ruta: Path) -> None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q": self.q.state_dict(),
                "q_target": self.q_target.state_dict(),
                "opt": self.opt.state_dict(),
                "pasos": self.pasos,
                "actualizaciones": self.actualizaciones,
                "cfg": asdict(self.cfg),
            },
            ruta,
        )

    @classmethod
    def cargar(cls, ruta: Path, n_acciones: int = 5) -> AgenteDQN:
        datos = torch.load(ruta, map_location="cpu", weights_only=False)
        agente = cls(n_acciones=n_acciones, cfg=Config(**datos["cfg"]))
        agente.q.load_state_dict(datos["q"])
        agente.q_target.load_state_dict(datos["q_target"])
        agente.opt.load_state_dict(datos["opt"])
        agente.pasos = datos["pasos"]
        agente.actualizaciones = datos["actualizaciones"]
        return agente
