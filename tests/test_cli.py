"""CLI smoke tests."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

from qgps.cli import main


def _run(args: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(args)
    return rc, buf.getvalue()


def test_info():
    rc, out = _run(["info"])
    assert rc == 0
    assert "qgps" in out
    assert "numpy" in out
    # Must never advertise real quantum hardware
    assert "No qubits" in out


def test_demo():
    rc, out = _run(["demo", "--seed", "1"])
    assert rc == 0
    assert "path length:" in out
    assert "S" in out and "T" in out


def test_plan_roundtrip(tmp_path: Path):
    grid = np.ones((16, 16), dtype=np.float32)
    speed_path = tmp_path / "speed.npy"
    np.save(speed_path, grid)
    out_path = tmp_path / "route.npy"
    rc, out = _run(
        [
            "plan",
            "--speed", str(speed_path),
            "--sx", "1", "--sy", "1",
            "--tx", "14", "--ty", "14",
            "--out", str(out_path),
        ]
    )
    assert rc == 0
    assert out_path.exists()
    path = np.load(out_path)
    assert path.shape[1] == 2
    assert tuple(path[0]) == (1, 1)
    assert tuple(path[-1]) == (14, 14)


def test_plan_missing_file_errors(tmp_path: Path):
    rc, _ = _run(
        [
            "plan",
            "--speed", str(tmp_path / "nope.npy"),
            "--sx", "0", "--sy", "0", "--tx", "1", "--ty", "1",
        ]
    )
    assert rc == 2


def test_help_exits_cleanly():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
