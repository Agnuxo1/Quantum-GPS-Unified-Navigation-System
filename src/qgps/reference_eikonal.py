"""
Reference CPU Eikonal solver (pure NumPy, classical fast-marching).

Solves  |grad T| * speed = 1  on a 2D grid with a point source using the
standard Sethian fast-marching-method (FMM) upwind quadratic update.

This module has NO GPU dependency and NO "quantum" content. It is a
deterministic classical algorithm used both for CLI offline planning and as
the analytic-validation oracle for the GPU solver.
"""

from __future__ import annotations

import heapq
from typing import Tuple

import numpy as np

INF = np.float32(1e30)

# Cell tags for FMM
FAR = 0
NARROW = 1
FROZEN = 2


def analytic_point_source(
    grid_shape: Tuple[int, int], source: Tuple[int, int], speed: float = 1.0
) -> np.ndarray:
    """Analytic arrival time from a point source in a homogeneous medium.

    T(x, y) = sqrt((x - x0)**2 + (y - y0)**2) / speed.
    """
    h, w = grid_shape
    sx, sy = source
    ys, xs = np.mgrid[0:h, 0:w]
    return np.sqrt((xs - sx) ** 2 + (ys - sy) ** 2).astype(np.float32) / float(speed)


def _solve_quadratic(a: float, b: float, f: float) -> float:
    """Upwind Eikonal quadratic update.

    Given already-frozen minimal times a (horizontal) and b (vertical) and
    slowness f = 1/speed, return the updated arrival time at this cell.
    """
    if a > b:
        a, b = b, a
    # If the cross-axis neighbour is further than one slowness step, the 1-D
    # update along the minimum axis is optimal.
    if (b - a) >= f:
        return a + f
    # Otherwise solve 0.5*(a+b) + 0.5*sqrt(2 f^2 - (a-b)^2)
    disc = 2.0 * f * f - (a - b) * (a - b)
    if disc < 0.0:
        return a + f
    return 0.5 * (a + b + float(np.sqrt(disc)))


def solve_eikonal_fmm(
    speed: np.ndarray,
    source: Tuple[int, int],
    obstacles: np.ndarray | None = None,
) -> np.ndarray:
    """Classical fast-marching Eikonal solver on a 2D grid.

    Args:
        speed: 2D array of positive local speeds (same units as 1/time-per-cell).
        source: (x, y) source pixel (column, row).
        obstacles: optional bool/0-1 array; cells > 0.5 are impassable.

    Returns:
        T: arrival-time field, same shape as ``speed``.
    """
    speed = np.asarray(speed, dtype=np.float32)
    if speed.ndim != 2:
        raise ValueError("speed must be 2D")
    h, w = speed.shape
    sx, sy = int(source[0]), int(source[1])
    if not (0 <= sx < w and 0 <= sy < h):
        raise ValueError("source outside grid")

    T = np.full((h, w), INF, dtype=np.float32)
    tag = np.zeros((h, w), dtype=np.uint8)

    if obstacles is None:
        blocked = np.zeros((h, w), dtype=bool)
    else:
        blocked = np.asarray(obstacles) > 0.5

    if blocked[sy, sx]:
        raise ValueError("source is on an obstacle")

    T[sy, sx] = 0.0
    tag[sy, sx] = FROZEN

    heap: list[tuple[float, int, int]] = []

    def _try_update(x: int, y: int) -> None:
        if blocked[y, x] or tag[y, x] == FROZEN:
            return
        s = float(speed[y, x])
        if s <= 1e-8:
            return
        f = 1.0 / s
        # Minimal frozen neighbour along each axis
        a = INF
        if x > 0 and tag[y, x - 1] == FROZEN:
            a = min(a, float(T[y, x - 1]))
        if x < w - 1 and tag[y, x + 1] == FROZEN:
            a = min(a, float(T[y, x + 1]))
        b = INF
        if y > 0 and tag[y - 1, x] == FROZEN:
            b = min(b, float(T[y - 1, x]))
        if y < h - 1 and tag[y + 1, x] == FROZEN:
            b = min(b, float(T[y + 1, x]))
        if a >= INF and b >= INF:
            return
        if a >= INF:
            new_t = b + f  # 1D update along the only known axis
        elif b >= INF:
            new_t = a + f
        else:
            new_t = _solve_quadratic(a, b, f)
        if new_t < float(T[y, x]):
            T[y, x] = new_t
            tag[y, x] = NARROW
            heapq.heappush(heap, (new_t, x, y))

    # Seed from source
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nx, ny = sx + dx, sy + dy
        if 0 <= nx < w and 0 <= ny < h:
            _try_update(nx, ny)

    while heap:
        t, x, y = heapq.heappop(heap)
        if tag[y, x] == FROZEN:
            continue
        # (no stale-entry rejection: the FROZEN check above is sufficient and
        # float32 T storage would otherwise falsely reject live entries.)
        tag[y, x] = FROZEN
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                _try_update(nx, ny)

    return T


def gradient_descent_path(
    T: np.ndarray,
    source: Tuple[int, int],
    target: Tuple[int, int],
    obstacles: np.ndarray | None = None,
    max_steps: int | None = None,
) -> list[tuple[int, int]]:
    """Walk from target to source along steepest descent on T."""
    h, w = T.shape
    sx, sy = source
    tx, ty = target
    if obstacles is None:
        blocked = np.zeros((h, w), dtype=bool)
    else:
        blocked = np.asarray(obstacles) > 0.5
    if max_steps is None:
        max_steps = h * w
    path: list[tuple[int, int]] = [(tx, ty)]
    cx, cy = tx, ty
    visited: set[tuple[int, int]] = set()
    for _ in range(max_steps):
        if (cx, cy) == (sx, sy):
            break
        visited.add((cx, cy))
        best = (cx, cy)
        best_t = float(T[cy, cx])
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if blocked[ny, nx]:
                continue
            if (nx, ny) in visited:
                continue
            if float(T[ny, nx]) < best_t:
                best_t = float(T[ny, nx])
                best = (nx, ny)
        if best == (cx, cy):
            break
        cx, cy = best
        path.append((cx, cy))
    path.reverse()
    return path
