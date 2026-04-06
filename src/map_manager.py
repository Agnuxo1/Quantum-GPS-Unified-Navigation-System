#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Map Manager - OpenStreetMap Integration
===============================================

Integrated map loading, caching, and rasterization for quantum GPS navigator.
Features:
- Parallel tile downloading from OpenStreetMap
- Grid rasterization (streets vs buildings)
- Speed field generation (fast roads, slow surfaces)
- Turn-by-turn navigation generation
"""

import os
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from io import BytesIO
import requests

try:
    import osmnx as ox
    import networkx as nx
    from shapely.geometry import Point, LineString
    from geopy.distance import geodesic
    OSM_AVAILABLE = True
except ImportError:
    OSM_AVAILABLE = False


@dataclass
class MapBounds:
    """Geographic bounds of map"""
    north: float
    south: float
    east: float
    west: float
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point (lat, lon)"""
        return ((self.north + self.south) / 2,
                (self.east + self.west) / 2)


@dataclass
class NavigationInstruction:
    """A single turn-by-turn instruction"""
    action: str  # "start", "continue", "turn_left", "turn_right", "arrive"
    distance: float  # meters
    street_name: str
    bearing: float  # degrees from north
    position: Tuple[float, float]  # (lat, lon)
    
    def __str__(self) -> str:
        if self.action == "start":
            return f"Start on {self.street_name}"
        elif self.action == "continue":
            return f"Continue {self.distance:.0f}m on {self.street_name}"
        elif self.action == "turn_left":
            return f"Turn left onto {self.street_name}"
        elif self.action == "turn_right":
            return f"Turn right onto {self.street_name}"
        elif self.action == "arrive":
            return f"Arrive at destination"
        else:
            return f"{self.action} - {self.street_name}"


class OptimizedTileManager:
    """Parallel tile downloading with caching"""
    
    def __init__(self, cache_dir: str = "data/tiles_cache"):
        self.tile_server = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        self.headers = {'User-Agent': 'QuantumGPS-Navigator/1.0'}
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=8)
    
    def lat_lon_to_tile(self, lat: float, lon: float, zoom: int) -> Tuple[int, int]:
        """Convert lat/lon to tile coordinates"""
        lat_rad = np.radians(lat)
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / np.pi) / 2.0 * n)
        return x, y
    
    def tile_to_lat_lon(self, x: int, y: int, zoom: int) -> Tuple[float, float]:
        """Convert tile to lat/lon"""
        n = 2.0 ** zoom
        lon = x / n * 360.0 - 180.0
        lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * y / n)))
        lat = np.degrees(lat_rad)
        return lat, lon
    
    def download_tile(self, x: int, y: int, z: int) -> Optional[Image.Image]:
        """Download single tile with caching"""
        cache_file = os.path.join(self.cache_dir, f"{z}_{x}_{y}.png")
        
        if os.path.exists(cache_file):
            try:
                return Image.open(cache_file)
            except:
                pass
        
        try:
            url = self.tile_server.format(z=z, x=x, y=y)
            resp = requests.get(url, headers=self.headers, timeout=5)
            if resp.status_code == 200:
                img = Image.open(BytesIO(resp.content))
                img.save(cache_file)
                return img
        except Exception as e:
            print(f"[TileManager] Error downloading {x},{y}: {e}")
        
        return None
    
    def load_map_area_parallel(self, center_lat: float, center_lon: float, 
                              zoom: int = 15, radius: int = 2) -> Tuple[Optional[Image.Image], Optional[Dict]]:
        """Load tiles in parallel"""
        cx, cy = self.lat_lon_to_tile(center_lat, center_lon, zoom)
        
        tiles_to_load = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                tiles_to_load.append((cx + dx, cy + dy, zoom))
        
        print(f"   [TileManager] Downloading {len(tiles_to_load)} tiles in parallel...")
        
        # Download in parallel
        futures = [self.executor.submit(self.download_tile, x, y, z)
                  for x, y, z in tiles_to_load]
        
        tiles = {}
        for i, future in enumerate(futures):
            result = future.result()
            if result:
                x, y, z = tiles_to_load[i]
                tiles[(x, y)] = result
        
        if not tiles:
            print("   [TileManager] ERROR: No tiles downloaded")
            return None, None
        
        # Stitch tiles
        min_x = min(x for x, y in tiles.keys())
        max_x = max(x for x, y in tiles.keys())
        min_y = min(y for x, y in tiles.keys())
        max_y = max(y for x, y in tiles.keys())
        
        tile_size = 256
        width = (max_x - min_x + 1) * tile_size
        height = (max_y - min_y + 1) * tile_size
        
        stitched = Image.new('RGB', (width, height))
        
        for (x, y), tile in tiles.items():
            px = (x - min_x) * tile_size
            py = (y - min_y) * tile_size
            stitched.paste(tile, (px, py))
        
        # Calculate bounds
        north, west = self.tile_to_lat_lon(min_x, min_y, zoom)
        south, east = self.tile_to_lat_lon(max_x + 1, max_y + 1, zoom)
        
        bounds = {
            'north': north,
            'south': south,
            'east': east,
            'west': west
        }
        
        print(f"   [TileManager] Map stitched: {width}x{height} pixels")
        return stitched, bounds


class MapManager:
    """Unified map management"""
    
    def __init__(self, cache_dir: str = "data/map_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.tile_manager = OptimizedTileManager()
        self.graph = None
        self.bounds = None
        self.map_image = None
        
    def load_city(self, place_name: str = "Madrid, Spain", radius_meters: int = 1500,
                  zoom: int = 15) -> bool:
        """Load street network for a city"""
        print(f"\n[MapManager] Loading map for {place_name}...")
        
        # Get center coordinates
        try:
            if OSM_AVAILABLE:
                location = ox.geocode(place_name)
                center_lat, center_lon = location
            else:
                # Default to Madrid
                center_lat, center_lon = 40.4168, -3.7038
        except:
            print(f"   [MapManager] Could not geocode {place_name}, using Madrid")
            center_lat, center_lon = 40.4168, -3.7038
        
        print(f"   Center: ({center_lat:.4f}, {center_lon:.4f})")
        
        # Download tiles
        print(f"\n[MapManager] Step 1: Downloading OSM tiles...")
        self.map_image, self.bounds = self.tile_manager.load_map_area_parallel(
            center_lat, center_lon, zoom=zoom, radius=2
        )
        
        if self.map_image is None:
            print("   [MapManager] ERROR: Could not load map tiles")
            return False
        
        # Load street network
        print(f"\n[MapManager] Step 2: Loading street network...")
        if OSM_AVAILABLE:
            try:
                self.graph = ox.graph_from_point(
                    (center_lat, center_lon),
                    dist=radius_meters,
                    network_type='drive'
                )
                print(f"   [MapManager] Network loaded: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
            except Exception as e:
                print(f"   [MapManager] WARNING: Could not load street network: {e}")
                self.graph = None
        
        return True
    
    def latlon_to_pixel(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert lat/lon to pixel coordinates on map"""
        if self.bounds is None:
            return 0, 0
        
        # Mercator projection
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # Bounds projection
        lat_rad_n = np.radians(self.bounds['north'])
        lat_rad_s = np.radians(self.bounds['south'])
        lon_w = self.bounds['west']
        lon_e = self.bounds['east']
        
        x_norm = (lon - lon_w) / (lon_e - lon_w)
        
        y_n = (1.0 - np.log(np.tan(lat_rad_n) + 1.0 / np.cos(lat_rad_n)) / np.pi) / 2.0
        y_s = (1.0 - np.log(np.tan(lat_rad_s) + 1.0 / np.cos(lat_rad_s)) / np.pi) / 2.0
        y_p = (1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / np.pi) / 2.0
        
        y_norm = (y_p - y_n) / (y_s - y_n)
        
        x_pix = int(x_norm * self.map_image.width) if self.map_image else 0
        y_pix = int(y_norm * self.map_image.height) if self.map_image else 0
        
        return x_pix, y_pix
    
    def pixel_to_latlon(self, x: int, y: int) -> Tuple[float, float]:
        """Convert pixel coordinates to lat/lon"""
        if self.bounds is None or self.map_image is None:
            return 0.0, 0.0
        
        x_norm = x / self.map_image.width
        y_norm = y / self.map_image.height
        
        lon_w = self.bounds['west']
        lon_e = self.bounds['east']
        lon = lon_w + x_norm * (lon_e - lon_w)
        
        lat_rad_n = np.radians(self.bounds['north'])
        lat_rad_s = np.radians(self.bounds['south'])
        
        y_n = (1.0 - np.log(np.tan(lat_rad_n) + 1.0 / np.cos(lat_rad_n)) / np.pi) / 2.0
        y_s = (1.0 - np.log(np.tan(lat_rad_s) + 1.0 / np.cos(lat_rad_s)) / np.pi) / 2.0
        
        y_p = y_n + y_norm * (y_s - y_n)
        lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * y_p)))
        lat = np.degrees(lat_rad)
        
        return lat, lon
    
    def create_street_grid(self, grid_size: int = 512) -> np.ndarray:
        """Create binary street grid from network"""
        grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        if self.graph is None:
            print("   [MapManager] WARNING: No graph available, creating empty grid")
            return grid
        
        try:
            # Draw edges on grid
            for u, v, key, data in self.graph.edges(keys=True, data=True):
                u_node = self.graph.nodes[u]
                v_node = self.graph.nodes[v]
                
                # Node positions
                u_lat, u_lon = u_node['y'], u_node['x']
                v_lat, v_lon = v_node['y'], v_node['x']
                
                u_pix = self.latlon_to_pixel(u_lat, u_lon)
                v_pix = self.latlon_to_pixel(v_lat, v_lon)
                
                # Bresenham line on grid
                self._draw_line(grid, u_pix, v_pix, value=1.0)
            
            print(f"   [MapManager] Street grid created: {np.sum(grid > 0)} street pixels")
        except Exception as e:
            print(f"   [MapManager] WARNING: Error drawing streets: {e}")
        
        return grid
    
    def create_speed_field(self, grid_size: int = 512) -> np.ndarray:
        """Create speed field based on street types"""
        speed_field = np.ones((grid_size, grid_size), dtype=np.float32)
        
        if self.graph is None:
            return speed_field
        
        try:
            # Assign speeds based on road type
            for u, v, key, data in self.graph.edges(keys=True, data=True):
                highway = data.get('highway', ['residential'])
                if isinstance(highway, str):
                    highway = [highway]
                highway = highway[0] if highway else 'residential'
                
                # Speed modifiers by road type
                speed_map = {
                    'motorway': 1.8,
                    'trunk': 1.7,
                    'primary': 1.6,
                    'secondary': 1.4,
                    'tertiary': 1.2,
                    'residential': 1.0,
                    'living_street': 0.7,
                    'footway': 0.5,
                    'pedestrian': 0.5,
                }
                speed_mult = speed_map.get(highway, 1.0)
                
                u_lat, u_lon = self.graph.nodes[u]['y'], self.graph.nodes[u]['x']
                v_lat, v_lon = self.graph.nodes[v]['y'], self.graph.nodes[v]['x']
                
                u_pix = self.latlon_to_pixel(u_lat, u_lon)
                v_pix = self.latlon_to_pixel(v_lat, v_lon)
                
                self._draw_line(speed_field, u_pix, v_pix, value=speed_mult)
        except Exception as e:
            print(f"   [MapManager] WARNING: Error creating speed field: {e}")
        
        return speed_field
    
    @staticmethod
    def _draw_line(grid: np.ndarray, p0: Tuple[int, int], p1: Tuple[int, int], 
                   value: float = 1.0, width: int = 2) -> None:
        """Bresenham line drawing on grid"""
        x0, y0 = p0
        x1, y1 = p1
        
        x0 = np.clip(x0, 0, grid.shape[1] - 1)
        y0 = np.clip(y0, 0, grid.shape[0] - 1)
        x1 = np.clip(x1, 0, grid.shape[1] - 1)
        y1 = np.clip(y1, 0, grid.shape[0] - 1)
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        
        err = dx - dy
        x, y = x0, y0
        
        steps = max(dx, dy)
        for _ in range(steps + 1):
            # Draw with width
            for dw in range(-width, width + 1):
                for dh in range(-width, width + 1):
                    nx = np.clip(x + dw, 0, grid.shape[1] - 1)
                    ny = np.clip(y + dh, 0, grid.shape[0] - 1)
                    grid[ny, nx] = max(grid[ny, nx], value)
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
