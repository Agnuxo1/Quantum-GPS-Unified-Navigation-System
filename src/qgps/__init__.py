"""
qgps - Quantum-Inspired GPS Navigator
======================================

Directional-diffusion GPU navigator using a 4-channel (N/E/S/W) amplitude
representation on a 2D raster.

IMPORTANT: This package is 100% CLASSICAL. The ``vec4`` values and the
normalization step are quantum-INSPIRED metaphors borrowed from wave
propagation; there are no qubits, no superposition, and no entanglement.
No quantum computing library (Qiskit, Cirq, PennyLane) is used or required.
"""

__version__ = "1.0.0"
__author__ = "Francisco Angulo de Lafuente"
__license__ = "Apache-2.0"

from .reference_eikonal import (
    analytic_point_source,
    gradient_descent_path,
    solve_eikonal_fmm,
)
from .navigator import Route, plan_path
from .tile_manager import TileManager

__all__ = [
    "__version__",
    "solve_eikonal_fmm",
    "analytic_point_source",
    "gradient_descent_path",
    "plan_path",
    "Route",
    "TileManager",
]
