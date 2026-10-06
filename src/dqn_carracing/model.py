"""La red Q: CNN que mapea una pila de 4 frames a 5 valores Q.

Arquitectura (entrada 4x84x84):

    capa              kernel/stride   salida          parámetros
    ---------------------------------------------------------------
    Conv2d(4 -> 16)   8x8 / 4         16 x 20 x 20     4 112
    ReLU
    Conv2d(16 -> 32)  4x4 / 2         32 x  9 x  9     8 224
    ReLU
    Conv2d(32 -> 32)  3x3 / 1         32 x  7 x  7     9 248
    ReLU
    Flatten                           1 568
    Linear(1568 -> 256)                 256          401 664
    ReLU
    Linear(256 -> 5)                      5            1 285
    ---------------------------------------------------------------
    total                                            424 533

Es la arquitectura de Mnih et al. (2015) con la mitad de filtros y 256 en vez
de 512 unidades densas: en CPU, cada paso de gradiente de la red completa
(1.7 M parámetros) costaba más que el propio simulador, y la pista de
CarRacing es visualmente mucho más simple que un juego de Atari.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RedQ(nn.Module):
    """Q(s, ·) para una pila de frames; devuelve un valor por acción."""

    def __init__(self, n_frames: int = 4, n_acciones: int = 5, densa: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_frames, 16, kernel_size=8, stride=4),  # 84 -> 20
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),        # 20 -> 9
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),        # 9 -> 7
            nn.ReLU(),
            nn.Flatten(),
        )
        self.cabeza = nn.Sequential(
            nn.Linear(32 * 7 * 7, densa),
            nn.ReLU(),
            nn.Linear(densa, n_acciones),                      # sin activación: Q no está acotado
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (lote, 4, 84, 84) uint8 o float. Devuelve (lote, n_acciones)."""
        x = x.float() / 255.0          # la normalización vive en la red, no en el buffer
        return self.cabeza(self.conv(x))

    @property
    def n_parametros(self) -> int:
        return sum(p.numel() for p in self.parameters())
