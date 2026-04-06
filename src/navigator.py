#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Quantum-GPS Navigator
=============================

Complete navigation system combining:
- GPU Eikonal solver for pathfinding
- Map management with real OpenStreetMap
- Navigation state tracking
- Turn-by-turn instruction generation
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class NavigationState:
    """Current navigation state"""
    position: Tuple[float, float]  # (lat, lon)
    heading: float  # degrees
    velocity: float  # m/s
    grid_position: Tuple[int, int]  # (x, y) in grid coords


class UnifiedNavigator:
    """Unified navigation system"""
    
    def __init__(self, map_manager, gpu_solver):
        """
        Initialize navigator
        
        Args:
            map_manager: MapManager instance
            gpu_solver: GpuEikonalSolver instance
        """
        self.map_manager = map_manager
        self.gpu_solver = gpu_solver
        
        # Navigation state
        self.current_position = (40.4168, -3.7038)  # lat, lon
        self.destination = None
        self.heading = 0.0
        self.velocity = 0.0
        
        # Route
        self.route_waypoints: List[Tuple[float, float]] = []
        self.route_grid_coords: List[Tuple[int, int]] = []
        self.current_waypoint_index = 0
        self.instructions: List[str] = []
        
        # Grid setup
        self.grid_size = gpu_solver.grid_size
        self._setup_grid_fields()
    
    def _setup_grid_fields(self) -> None:
        """Setup obstacle and speed fields from map"""
        print("\n[Navigator] Setting up grid fields...")
        
        # Create street grid
        street_grid = self.map_manager.create_street_grid(self.grid_size)
        
        # Create speed field
        speed_field = self.map_manager.create_speed_field(self.grid_size)
        
        # Invert street grid for obstacles (1 = street, 0 = obstacle)
        obstacles = 1.0 - np.clip(street_grid, 0.0, 1.0)
        
        # Set in GPU solver
        self.gpu_solver.set_obstacle_field(obstacles)
        self.gpu_solver.set_speed_field(speed_field)
        
        print(f"   [Navigator] Grid fields ready: {self.grid_size}x{self.grid_size}")
    
    def set_destination(self, lat: float, lon: float) -> bool:
        """Set destination and compute route"""
        print(f"\n[Navigator] Setting destination: ({lat:.4f}, {lon:.4f})")
        
        self.destination = (lat, lon)
        
        # Convert to grid coordinates
        current_grid = self.map_manager.latlon_to_pixel(
            self.current_position[0], self.current_position[1]
        )
        dest_grid = self.map_manager.latlon_to_pixel(lat, lon)
        
        print(f"   Start grid: {current_grid}, Dest grid: {dest_grid}")
        
        # Set in GPU solver
        self.gpu_solver.set_source_target(current_grid, dest_grid)
        
        # Compute path
        try:
            grid_path = self.gpu_solver.compute_path()
            
            if not grid_path:
                print("   [Navigator] ERROR: No path found")
                return False
            
            # Convert grid path to lat/lon
            self.route_grid_coords = grid_path
            self.route_waypoints = []
            
            for gx, gy in grid_path:
                # Normalize to map image coordinates
                lat, lon = self.map_manager.pixel_to_latlon(gx, gy)
                self.route_waypoints.append((lat, lon))
            
            # Generate instructions
            self._generate_instructions()
            
            print(f"   [Navigator] Route computed: {len(self.route_waypoints)} waypoints")
            print(f"   [Navigator] Path length: {self.gpu_solver.path_length:.1f} grid units")
            print(f"   [Navigator] Path cost: {self.gpu_solver.path_cost:.1f}")
            
            return True
        
        except Exception as e:
            print(f"   [Navigator] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_instructions(self) -> None:
        """Generate turn-by-turn instructions"""
        self.instructions = []
        
        if len(self.route_waypoints) < 2:
            return
        
        for i, (lat, lon) in enumerate(self.route_waypoints):
            if i == 0:
                self.instructions.append("Start navigation")
            elif i == len(self.route_waypoints) - 1:
                self.instructions.append("Arrive at destination")
            else:
                # Calculate bearing change
                if i > 0:
                    prev_lat, prev_lon = self.route_waypoints[i - 1]
                    bearing_prev = np.degrees(np.arctan2(
                        lon - prev_lon,
                        lat - prev_lat
                    ))
                else:
                    bearing_prev = self.heading
                
                if i < len(self.route_waypoints) - 1:
                    next_lat, next_lon = self.route_waypoints[i + 1]
                    bearing_next = np.degrees(np.arctan2(
                        next_lon - lon,
                        next_lat - lat
                    ))
                else:
                    bearing_next = bearing_prev
                
                turn = bearing_next - bearing_prev
                
                if abs(turn) < 15:
                    self.instructions.append("Continue")
                elif turn > 0:
                    self.instructions.append(f"Turn right {abs(turn):.0f}°")
                else:
                    self.instructions.append(f"Turn left {abs(turn):.0f}°")
    
    def update(self, dt: float, target_velocity: float = 5.0, angular_velocity: float = 0.0) -> None:
        """Update navigator state"""
        # Update velocity
        accel = 2.0  # m/s²
        velocity_delta = np.clip(target_velocity - self.velocity, -accel * dt, accel * dt)
        self.velocity = np.clip(self.velocity + velocity_delta, 0.0, 30.0)
        
        # Update heading
        self.heading += angular_velocity * dt
        self.heading %= 360.0
        
        # Update position (simple kinematic model)
        if self.velocity > 0.1:
            heading_rad = np.radians(self.heading)
            
            # Displacement in meters
            dx_m = np.sin(heading_rad) * self.velocity * dt
            dy_m = np.cos(heading_rad) * self.velocity * dt
            
            # Convert to lat/lon delta
            lat_m = 111320.0
            lon_m = 111320.0 * np.cos(np.radians(self.current_position[0]))
            
            dlat = dy_m / lat_m
            dlon = dx_m / lon_m
            
            self.current_position = (
                self.current_position[0] + dlat,
                self.current_position[1] + dlon
            )
    
    def get_current_instruction(self) -> Optional[str]:
        """Get current turn-by-turn instruction"""
        if self.current_waypoint_index < len(self.instructions):
            return self.instructions[self.current_waypoint_index]
        return None
    
    def get_distance_to_destination(self) -> float:
        """Get distance to destination in meters"""
        if not self.destination:
            return 0.0
        
        lat1, lon1 = self.current_position
        lat2, lon2 = self.destination
        
        # Haversine distance
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return 6371000.0 * c  # Earth radius in meters
    
    def get_status(self) -> str:
        """Get navigation status"""
        status = f"Position: ({self.current_position[0]:.4f}, {self.current_position[1]:.4f})\n"
        status += f"Heading: {self.heading:.1f}°\n"
        status += f"Velocity: {self.velocity * 3.6:.1f} km/h\n"
        
        if self.destination:
            dist = self.get_distance_to_destination()
            status += f"Distance to destination: {dist:.1f}m\n"
            
            instr = self.get_current_instruction()
            if instr:
                status += f"Instruction: {instr}\n"
        
        return status
