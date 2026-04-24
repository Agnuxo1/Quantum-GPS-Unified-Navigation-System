"""High-level route planning on a 2D cost grid.

Classical, CPU-only. Uses the fast-marching Eikonal solver to produce an
arrival-time field and then walks steepest-descent from target to source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .reference_eikonal import (
    gradient_descent_path,
    solve_eikonal_fmm,
)


@dataclass(frozen=True)
class Route:
    """Result of a path-planning query."""

    path: list[tuple[int, int]]
    arrival_time: np.ndarray
    total_time: float

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.path)


def plan_path(
    speed: np.ndarray,
    source: Tuple[int, int],
    target: Tuple[int, int],
    obstacles: np.ndarray | None = None,
) -> Route:
    """Plan a path from source to target on a 2D speed grid.

    Args:
        speed: 2D array of positive local speeds.
        source: (x, y) start pixel.
        target: (x, y) end pixel.
        obstacles: optional boolean mask, True = blocked.

    Returns:
        A :class:`Route` with the path (list of (x, y) in forward order),
        the full arrival-time field and the total travel time to target.
    """
    T = solve_eikonal_fmm(speed, source, obstacles=obstacles)
    tx, ty = target
    path = gradient_descent_path(T, source, target, obstacles=obstacles)
    return Route(path=path, arrival_time=T, total_time=float(T[ty, tx]))
