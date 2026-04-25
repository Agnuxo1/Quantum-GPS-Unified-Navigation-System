# Quantum-Inspired GPS Navigator

**Tagline:** Directional-diffusion GPU navigator; 4-channel (N/E/S/W) amplitude metaphor; 100% classical. No qubits, no superposition, no entanglement.

> **Disclaimer — does NOT use quantum computing.** This project is a *classical* pathfinder on a 2D raster. The name "quantum-inspired" refers only to the **4-channel amplitude metaphor** (N/E/S/W directional components with a wave-propagation flavour) used by the directional-diffusion update. There are **no qubits**, no superposition, no entanglement, and no dependency on any quantum-computing library (Qiskit, Cirq, PennyLane, etc.). The core math is the classical Eikonal equation `|grad T| * v = 1`, solved with either a GPU shader pass or a reference CPU fast-marching method.

[![PyPI](https://img.shields.io/pypi/v/quantum-gps-navigator.svg)](https://pypi.org/project/quantum-gps-navigator/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

## Install

```bash
pip install quantum-gps-navigator             # core (NumPy + Pillow)
pip install "quantum-gps-navigator[gpu]"      # + moderngl, glfw
pip install "quantum-gps-navigator[osm]"      # + osmnx
pip install "quantum-gps-navigator[dev]"      # + pytest, build, twine
```

## CLI

```bash
qgps info                                     # show version + backend availability
qgps demo --seed 0                            # 64x64 synthetic demo, ASCII route
qgps plan --speed grid.npy --sx 1 --sy 1 \
           --tx 100 --ty 100 --out route.npy  # offline planner on a .npy speed grid
qgps --help
```

## Python API

```python
import numpy as np
from qgps import plan_path

speed = np.ones((128, 128), dtype=np.float32)
route = plan_path(speed, source=(4, 4), target=(120, 120))
print(len(route.path), route.total_time)
```

## How it works

1. **Eikonal arrival-time field.** Given a local-speed raster `v(x, y)` and a source cell, we solve `|grad T| * v = 1` for the scalar field `T(x, y)` = minimum travel time from the source.
2. **Solvers:**
    * `src/qgps/reference_eikonal.py` - classical Sethian fast-marching on a binary heap (pure NumPy, CPU, no GPU required).
    * `src/qgps/gpu_eikonal_solver.py` *(optional `[gpu]` extra)* - a directional-diffusion GPU pass using a 4-channel (N/E/S/W) amplitude raster. The channel metaphor is wave-inspired; the update is a plain upwind operator.
3. **Path extraction.** Steepest descent on `T` from target to source, then reversed.

```
    source o----->----->----->----->-----o target
             Eikonal T(x,y) field solved in O(N log N) (FMM)
             or O(N) GPU sweeps (directional diffusion)
```

## Graceful fallback

If `moderngl` / `glfw` / `osmnx` are unavailable at import time, the library falls back to the pure-NumPy reference solver and logs a clear message. `qgps info` prints exactly which optional backends are available.

## Tests

```bash
pip install -e .[dev]
pytest -v
```

The test suite runs **entirely on CPU**. Coverage includes the fast-marching solver against the analytic point-source solution on a 128x128 grid (relative error < 5%), the high-level planner (monotonicity, detours, path length), the HTTP tile cache (stdlib `http.server` mock), and the CLI.

## Layout

```
src/qgps/
    __init__.py
    reference_eikonal.py   # CPU fast-marching (validation oracle)
    navigator.py           # plan_path, Route
    tile_manager.py        # HTTP tile cache (stdlib urllib)
    cli.py                 # qgps plan / demo / info
tests/
    test_eikonal_correctness.py
    test_navigator.py
    test_tile_manager.py
    test_cli.py
```

## License

Apache-2.0 (c) 2026 Francisco Angulo de Lafuente.

## Citation

```
Angulo de Lafuente, F. (2026). Quantum-Inspired GPS Navigator (v1.0.0).
https://github.com/Agnuxo1/Quantum-GPS-Unified-Navigation-System
```

---

## Related projects

Part of the [@Agnuxo1](https://github.com/Agnuxo1) v1.0.0 open-source catalog (April 2026).

**AgentBoot constellation** — agents and research loops
- [AgentBoot](https://github.com/Agnuxo1/AgentBoot) — Conversational AI agent for bare-metal hardware detection and OS install.
- [autoresearch-nano](https://github.com/Agnuxo1/autoresearch) — nanoGPT-based autonomous ML research loop.
- [The Living Agent](https://github.com/Agnuxo1/The-Living-Agent) — 16x16 Chess-Grid autonomous research agent.
- [benchclaw-integrations](https://github.com/Agnuxo1/benchclaw-integrations) — Agent-framework adapters for the BenchClaw API.

**CHIMERA / neuromorphic constellation** — GPU-native scientific computing
- [NeuroCHIMERA](https://github.com/Agnuxo1/NeuroCHIMERA__GPU-Native_Neuromorphic_Consciousness) — GPU-native neuromorphic framework on OpenGL compute shaders.
- [Holographic-Reservoir](https://github.com/Agnuxo1/Holographic-Reservoir) — Reservoir computing with simulated ASIC backend.
- [ASIC-RAG-CHIMERA](https://github.com/Agnuxo1/ASIC-RAG-CHIMERA) — GPU simulation of a SHA-256 hash engine wired into a RAG pipeline.
- [QESN-MABe](https://github.com/Agnuxo1/QESN_MABe_V2_REPO) — Quantum-inspired Echo State Network on a 2D lattice (classical).
- [ARC2-CHIMERA](https://github.com/Agnuxo1/ARC2_CHIMERA) — Research PoC: OpenGL primitives for symbolic reasoning.
