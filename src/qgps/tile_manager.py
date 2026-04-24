"""Minimal OSM-style raster tile fetcher with on-disk cache.

Pure-stdlib HTTP (``urllib``) so it can be unit-tested against a local
``http.server`` without any network. No "quantum" content.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen


@dataclass
class TileManager:
    """Fetch and cache raster tiles from a URL template.

    The URL template uses ``{z}``/``{x}``/``{y}`` placeholders, for example
    ``"http://localhost:8000/{z}/{x}/{y}.png"``.
    """

    url_template: str
    cache_dir: Path
    user_agent: str = "qgps/1.0 (+https://pypi.org/project/quantum-gps-navigator/)"
    timeout: float = 10.0

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, z: int, x: int, y: int) -> Path:
        key = f"{z}/{x}/{y}"
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{z}_{x}_{y}_{h}.tile"

    def get_tile(self, z: int, x: int, y: int) -> bytes:
        """Return raw tile bytes, using the on-disk cache when possible."""
        path = self._cache_path(z, x, y)
        if path.exists():
            return path.read_bytes()
        url = self.url_template.format(z=z, x=x, y=y)
        req = Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                data = resp.read()
        except URLError as exc:  # pragma: no cover - network
            raise RuntimeError(f"tile fetch failed: {url}: {exc}") from exc
        path.write_bytes(data)
        return data

    def clear_cache(self) -> int:
        """Remove all cached tiles and return the count deleted."""
        n = 0
        for p in self.cache_dir.glob("*.tile"):
            p.unlink()
            n += 1
        return n
