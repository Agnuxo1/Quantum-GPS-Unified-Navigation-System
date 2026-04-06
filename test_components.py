#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite - Quantum GPS Navigator
==================================

Validation tests for all components
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from map_manager import MapManager, OptimizedTileManager
from gpu_eikonal_solver import GpuEikonalSolver


def test_tile_manager():
    """Test OptimizedTileManager"""
    print("\n[TEST] TileManager...")
    try:
        mgr = OptimizedTileManager()
        
        # Test coordinate conversion
        x, y = mgr.lat_lon_to_tile(40.4168, -3.7038, zoom=15)
        print(f"  ✓ lat_lon_to_tile: ({x}, {y})")
        
        lat, lon = mgr.tile_to_lat_lon(x, y, zoom=15)
        print(f"  ✓ tile_to_lat_lon: ({lat:.4f}, {lon:.4f})")
        
        return True
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return False


def test_map_manager():
    """Test MapManager"""
    print("\n[TEST] MapManager...")
    try:
        mgr = MapManager()
        
        # Test street grid creation
        grid = np.ones((256, 256), dtype=np.float32)
        print(f"  ✓ Test grid created: {grid.shape}")
        
        # Test coordinate conversion
        x, y = mgr.latlon_to_pixel(40.4168, -3.7038)
        print(f"  ✓ latlon_to_pixel: ({x}, {y})")
        
        return True
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return False


def test_gpu_solver():
    """Test GpuEikonalSolver"""
    print("\n[TEST] GpuEikonalSolver...")
    try:
        solver = GpuEikonalSolver(grid_size=256, headless=True)
        print(f"  ✓ Solver initialized (headless mode)")
        
        # Test field setup
        solver.set_obstacle_field(np.zeros((256, 256), dtype=np.float32))
        print(f"  ✓ Obstacle field set")
        
        solver.set_speed_field(np.ones((256, 256), dtype=np.float32))
        print(f"  ✓ Speed field set")
        
        # Test source/target
        solver.set_source_target((32, 128), (224, 128))
        print(f"  ✓ Source/target set")
        
        return True
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("QUANTUM GPS NAVIGATOR - TEST SUITE".center(60))
    print("="*60)
    
    tests = [
        ("TileManager", test_tile_manager),
        ("MapManager", test_map_manager),
        ("GpuSolver", test_gpu_solver),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[TEST] {name} CRASHED: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY".center(60))
    print("="*60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:30s} {status}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return all(r for _, r in results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
