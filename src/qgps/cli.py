"""qgps command-line interface.

Subcommands:
    plan   - plan a path on a stored .npy speed grid (offline)
    demo   - run a synthetic 64x64 demo and print an ASCII path
    info   - report availability of optional GPU/OSM backends
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .navigator import plan_path


def _render_ascii(shape, path, source, target) -> str:
    h, w = shape
    grid = [["."] * w for _ in range(h)]
    for x, y in path:
        if 0 <= x < w and 0 <= y < h:
            grid[y][x] = "*"
    sx, sy = source
    tx, ty = target
    grid[sy][sx] = "S"
    grid[ty][tx] = "T"
    return "\n".join("".join(row) for row in grid)


def _cmd_demo(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    h = w = 64
    speed = np.ones((h, w), dtype=np.float32)
    obstacles = np.zeros((h, w), dtype=bool)
    # A few random rectangular obstacles
    for _ in range(6):
        x0 = int(rng.integers(5, w - 15))
        y0 = int(rng.integers(5, h - 15))
        obstacles[y0 : y0 + 8, x0 : x0 + 3] = True
    source = (4, 4)
    target = (w - 5, h - 5)
    obstacles[source[1], source[0]] = False
    obstacles[target[1], target[0]] = False

    route = plan_path(speed, source, target, obstacles=obstacles.astype(np.float32))
    print(f"qgps demo {w}x{h}  source={source}  target={target}")
    print(f"path length: {len(route.path)}  total_time: {route.total_time:.3f}")
    print(_render_ascii((h, w), route.path, source, target))
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    speed_path = Path(args.speed)
    if not speed_path.exists():
        print(f"error: speed file not found: {speed_path}", file=sys.stderr)
        return 2
    speed = np.load(speed_path).astype(np.float32)
    source = (args.sx, args.sy)
    target = (args.tx, args.ty)
    route = plan_path(speed, source, target)
    print(f"path_len={len(route.path)} total_time={route.total_time:.4f}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, np.asarray(route.path, dtype=np.int32))
        print(f"saved path -> {out}")
    return 0


def _cmd_info(_: argparse.Namespace) -> int:
    print(f"qgps {__version__}")
    print("tagline: Directional-diffusion GPU navigator; 4-channel (N/E/S/W)")
    print("         amplitude metaphor; 100% classical. No qubits.")
    for name in ("moderngl", "glfw", "osmnx", "numpy", "PIL"):
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "?")
            print(f"  {name:10s} OK    ({ver})")
        except Exception as exc:  # pragma: no cover
            print(f"  {name:10s} MISS  ({exc.__class__.__name__})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qgps",
        description="Quantum-Inspired GPS Navigator (100% classical).",
    )
    p.add_argument("--version", action="version", version=f"qgps {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    pl = sub.add_parser("plan", help="plan a path on a stored speed grid")
    pl.add_argument("--speed", required=True, help=".npy file with 2D speed grid")
    pl.add_argument("--sx", type=int, required=True)
    pl.add_argument("--sy", type=int, required=True)
    pl.add_argument("--tx", type=int, required=True)
    pl.add_argument("--ty", type=int, required=True)
    pl.add_argument("--out", help="optional .npy output path for the route")
    pl.set_defaults(func=_cmd_plan)

    dm = sub.add_parser("demo", help="run a synthetic 64x64 demo")
    dm.add_argument("--seed", type=int, default=0)
    dm.set_defaults(func=_cmd_demo)

    info = sub.add_parser("info", help="show backend availability")
    info.set_defaults(func=_cmd_info)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
