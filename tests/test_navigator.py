"""Synthetic A -> B tests for the high-level planner."""

import numpy as np

from qgps.navigator import plan_path


def test_straight_line_free_space():
    speed = np.ones((64, 64), dtype=np.float32)
    route = plan_path(speed, (5, 5), (58, 58))
    assert route.path[0] == (5, 5)
    assert route.path[-1] == (58, 58)
    assert route.total_time > 0.0
    # Analytic lower bound for unit speed
    assert route.total_time >= np.sqrt((58 - 5) ** 2 + (58 - 5) ** 2) * 0.95


def test_path_is_monotonic_in_arrival_time():
    speed = np.ones((64, 64), dtype=np.float32)
    route = plan_path(speed, (5, 5), (58, 58))
    T = route.arrival_time
    ts = [float(T[y, x]) for (x, y) in route.path]
    # forward path: arrival time should be non-decreasing from source to target
    diffs = np.diff(ts)
    assert np.all(diffs >= -1e-4), "path not monotonic in arrival time"


def test_obstacles_force_detour():
    h = w = 64
    speed = np.ones((h, w), dtype=np.float32)
    obstacles = np.zeros((h, w), dtype=np.float32)
    obstacles[20:45, 30:33] = 1.0  # vertical wall with a gap
    route = plan_path(speed, (5, 32), (58, 32), obstacles=obstacles)
    # Detoured path must be longer than the straight-line (blocked) distance
    assert route.total_time > (58 - 5)


def test_route_length_reasonable():
    speed = np.ones((32, 32), dtype=np.float32)
    route = plan_path(speed, (1, 1), (30, 30))
    # path length should be close to Chebyshev distance, not wildly larger
    assert 20 <= len(route.path) <= 80
