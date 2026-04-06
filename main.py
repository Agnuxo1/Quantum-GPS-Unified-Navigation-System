#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHIMERA v3.0 - Unified Quantum GPS Navigator
============================================

The Next-Generation Navigation System.
Based on "Intelligence as Process" and "Zero-Memory" architecture.

Features:
- **Quantum Positioning**: Wavefunction simulation for GPS-free localization.
- **Optical Pathfinding**: Real-time Eikonal solver on GPU.
- **Neuromorphic Architecture**: 4-state quantum memory in texture loops.
- **Unified System**: Single GPU kernel for physics, logic, and rendering.

Usage:
    python main.py --city "Madrid, Spain" --grid-size 512
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chimera_system import ChimeraSystem

def main():
    """Application entry point"""
    parser = argparse.ArgumentParser(
        description="CHIMERA v3.0 - Quantum GPS Navigator"
    )
    parser.add_argument(
        "--city",
        type=str,
        default="Madrid, Spain",
        help="City/place to navigate (default: Madrid, Spain)"
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=512,
        help="Computation grid size (default: 512)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no window)"
    )
    
    args = parser.parse_args()
    
    print(f"\nInitializing CHIMERA v3.0 System...")
    print(f"Target: {args.city}")
    print(f"Grid:   {args.grid_size}x{args.grid_size}")
    print(f"Mode:   {'Headless' if args.headless else 'Interactive'}")
    
    try:
        system = ChimeraSystem(
            city_name=args.city, 
            grid_size=args.grid_size,
            headless=args.headless
        )
        system.run()
    except KeyboardInterrupt:
        print("\n[CHIMERA] Shutdown requested.")
    except Exception as e:
        print(f"\n[CHIMERA] Critical Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
