"""TileManager tests using stdlib http.server as a local mock."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from qgps.tile_manager import TileManager


class _TileHandler(BaseHTTPRequestHandler):
    hits: dict[str, int] = {}

    def log_message(self, *_args, **_kw):  # silence
        return

    def do_GET(self):  # noqa: N802
        _TileHandler.hits[self.path] = _TileHandler.hits.get(self.path, 0) + 1
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        body = b"PNGDATA:" + self.path.encode()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server():
    _TileHandler.hits = {}
    srv = HTTPServer(("127.0.0.1", 0), _TileHandler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def test_fetches_tile_then_caches(server, tmp_path: Path):
    port = server.server_address[1]
    tm = TileManager(
        url_template=f"http://127.0.0.1:{port}/{{z}}/{{x}}/{{y}}.png",
        cache_dir=tmp_path / "tiles",
    )
    data1 = tm.get_tile(5, 1, 2)
    data2 = tm.get_tile(5, 1, 2)
    assert data1 == data2
    assert data1.startswith(b"PNGDATA:")
    # second call must be served from cache - only one HTTP hit
    assert _TileHandler.hits == {"/5/1/2.png": 1}


def test_clear_cache(server, tmp_path: Path):
    port = server.server_address[1]
    tm = TileManager(
        url_template=f"http://127.0.0.1:{port}/{{z}}/{{x}}/{{y}}.png",
        cache_dir=tmp_path / "tiles",
    )
    tm.get_tile(1, 0, 0)
    tm.get_tile(1, 0, 1)
    n = tm.clear_cache()
    assert n == 2
    assert list((tmp_path / "tiles").glob("*.tile")) == []


def test_cache_survives_new_instance(server, tmp_path: Path):
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/{{z}}/{{x}}/{{y}}.png"
    TileManager(url_template=url, cache_dir=tmp_path / "c").get_tile(2, 2, 2)
    TileManager(url_template=url, cache_dir=tmp_path / "c").get_tile(2, 2, 2)
    assert _TileHandler.hits == {"/2/2/2.png": 1}
