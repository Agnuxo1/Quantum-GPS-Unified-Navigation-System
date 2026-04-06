#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Landmark-Based Position Correction System
==========================================

Detects turns and uses map intersections/corners as landmarks to correct
position with millimeter precision. This is the key to GPS-free navigation.

Concept:
1. Quantum wavefunction continuously estimates position (dead reckoning)
2. When a turn is detected, search for nearby intersections in the map
3. Match the turn pattern to the intersection geometry
4. Correct position to the exact intersection point
5. Update wavefunction to reflect the correction
"""

import numpy as np
import math
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class TurnEvent:
    """Detected turn event"""
    timestamp: float
    position: Tuple[float, float]  # Estimated position before turn
    heading_before: float  # Heading before turn (degrees)
    heading_after: float  # Heading after turn (degrees)
    turn_angle: float  # Turn angle (degrees, positive = right, negative = left)
    speed: float  # Speed during turn (m/s)


@dataclass
class Intersection:
    """Map intersection/corner"""
    position: Tuple[float, float]  # (x, y) in grid coordinates
    incoming_streets: List[float]  # List of incoming street angles (degrees)
    outgoing_streets: List[float]  # List of outgoing street angles (degrees)
    confidence: float  # How well this matches the turn pattern


class LandmarkCorrectionSystem:
    """
    Detects turns and corrects position using map intersections as landmarks.
    
    This is the "quantum measurement" moment: when we detect a turn,
    we collapse the wavefunction to the most likely intersection.
    """
    
    def __init__(self, grid_size: int, obstacle_field: np.ndarray):
        """
        Initialize landmark correction system
        
        Args:
            grid_size: Size of the grid
            obstacle_field: Obstacle field (1.0 = building, 0.0 = street)
        """
        self.grid_size = grid_size
        self.obstacle_field = obstacle_field
        
        # Turn detection parameters
        self.min_turn_angle = 15.0  # Minimum angle to consider a turn (degrees)
        self.turn_detection_window = 0.5  # Time window for turn detection (seconds)
        self.max_turn_radius = 3.0  # Maximum radius to search for intersections (grid cells)
        
        # History for turn detection
        self.heading_history: List[Tuple[float, float]] = []  # [(timestamp, heading), ...]
        self.position_history: List[Tuple[float, Tuple[float, float]]] = []  # [(timestamp, (x, y)), ...]
        
        # Detected intersections in the map
        self.intersections: List[Intersection] = []
        self._extract_intersections()
        
    def _extract_intersections(self):
        """Extract all intersections/corners from the map"""
        print("[LANDMARK] Extracting intersections from map...")
        
        # Find all street pixels
        street_mask = self.obstacle_field < 0.5
        
        # Find corners: pixels with 2-4 street neighbors in cardinal directions
        intersections = []
        for y in range(1, self.grid_size - 1):
            for x in range(1, self.grid_size - 1):
                if not street_mask[y, x]:
                    continue
                
                # Count street neighbors
                neighbors = [
                    street_mask[y-1, x],  # North
                    street_mask[y+1, x],  # South
                    street_mask[y, x-1],  # West
                    street_mask[y, x+1],  # East
                ]
                num_streets = sum(neighbors)
                
                # Intersection: 3 or 4 street neighbors (T-junction or crossroad)
                if num_streets >= 3:
                    # Determine incoming/outgoing street directions
                    incoming = []
                    outgoing = []
                    
                    if neighbors[0]:  # North
                        incoming.append(180.0)  # Coming from south
                        outgoing.append(0.0)     # Going north
                    if neighbors[1]:  # South
                        incoming.append(0.0)    # Coming from north
                        outgoing.append(180.0)  # Going south
                    if neighbors[2]:  # West
                        incoming.append(90.0)   # Coming from east
                        outgoing.append(270.0)  # Going west
                    if neighbors[3]:  # East
                        incoming.append(270.0)  # Coming from west
                        outgoing.append(90.0)   # Going east
                    
                    intersections.append(Intersection(
                        position=(x, y),
                        incoming_streets=incoming,
                        outgoing_streets=outgoing,
                        confidence=0.0
                    ))
        
        self.intersections = intersections
        print(f"[LANDMARK] Found {len(intersections)} intersections in map")
    
    def update(self, timestamp: float, position: Tuple[float, float], 
               heading: float, speed: float) -> Optional[TurnEvent]:
        """
        Update position and detect turns
        
        Returns:
            TurnEvent if a turn was detected, None otherwise
        """
        # Add to history
        self.heading_history.append((timestamp, heading))
        self.position_history.append((timestamp, position))
        
        # Keep only recent history
        cutoff_time = timestamp - self.turn_detection_window
        self.heading_history = [(t, h) for t, h in self.heading_history if t >= cutoff_time]
        self.position_history = [(t, p) for t, p in self.position_history if t >= cutoff_time]
        
        # Detect turn if we have enough history
        if len(self.heading_history) < 3:
            return None
        
        # Calculate heading change
        recent_headings = [h for _, h in self.heading_history[-5:]]
        heading_before = recent_headings[0]
        heading_after = recent_headings[-1]
        
        # Normalize angle difference
        turn_angle = heading_after - heading_before
        if turn_angle > 180:
            turn_angle -= 360
        elif turn_angle < -180:
            turn_angle += 360
        
        # Check if this is a significant turn
        if abs(turn_angle) < self.min_turn_angle:
            return None
        
        # Create turn event
        turn_event = TurnEvent(
            timestamp=timestamp,
            position=position,
            heading_before=heading_before,
            heading_after=heading_after,
            turn_angle=turn_angle,
            speed=speed
        )
        
        return turn_event
    
    def correct_position(self, turn_event: TurnEvent) -> Optional[Tuple[float, float]]:
        """
        Correct position using map intersections as landmarks
        
        Args:
            turn_event: The detected turn event
            
        Returns:
            Corrected position if a matching intersection is found, None otherwise
        """
        tx, ty = turn_event.position
        
        # Search for intersections near the estimated position
        candidates = []
        for intersection in self.intersections:
            ix, iy = intersection.position
            
            # Distance check
            dist = math.sqrt((ix - tx)**2 + (iy - ty)**2)
            if dist > self.max_turn_radius:
                continue
            
            # Check if the turn pattern matches this intersection
            # We're looking for an intersection where:
            # - We can come from heading_before direction
            # - We can go to heading_after direction
            
            # Normalize headings to [0, 360)
            h_before = turn_event.heading_before % 360
            h_after = turn_event.heading_after % 360
            
            # Check if this intersection supports this turn
            # Find closest incoming direction
            best_incoming_match = min(
                intersection.incoming_streets,
                key=lambda h: min(abs(h - h_before), abs(h - h_before + 360), abs(h - h_before - 360))
            )
            incoming_error = min(abs(best_incoming_match - h_before), 
                                abs(best_incoming_match - h_before + 360),
                                abs(best_incoming_match - h_before - 360))
            
            # Find closest outgoing direction
            best_outgoing_match = min(
                intersection.outgoing_streets,
                key=lambda h: min(abs(h - h_after), abs(h - h_after + 360), abs(h - h_after - 360))
            )
            outgoing_error = min(abs(best_outgoing_match - h_after),
                                abs(best_outgoing_match - h_after + 360),
                                abs(best_outgoing_match - h_after - 360))
            
            # Calculate confidence: lower error = higher confidence
            angle_error = (incoming_error + outgoing_error) / 2.0
            distance_error = dist / self.max_turn_radius
            
            # Combined confidence (0.0 = perfect match, 1.0 = no match)
            confidence = (angle_error / 45.0) * 0.5 + distance_error * 0.5
            
            if confidence < 0.7:  # Good enough match
                intersection.confidence = 1.0 - confidence
                candidates.append((intersection, confidence))
        
        if not candidates:
            return None
        
        # Select best match (lowest confidence error = highest confidence)
        best_match, best_confidence = min(candidates, key=lambda x: x[1])
        
        # Return corrected position (exact intersection point)
        corrected_pos = best_match.position
        print(f"[LANDMARK] Turn detected: {turn_event.turn_angle:.1f}° at ({tx:.1f}, {ty:.1f})")
        print(f"  → Corrected to intersection at ({corrected_pos[0]:.1f}, {corrected_pos[1]:.1f})")
        print(f"  → Confidence: {best_match.confidence:.2f}, Error reduction: {math.sqrt((tx-corrected_pos[0])**2 + (ty-corrected_pos[1])**2):.2f} cells")
        
        return corrected_pos
    
    def get_correction_radius(self) -> float:
        """Get the radius around corrected position to update wavefunction"""
        return 2.0  # Update wavefunction in 2-cell radius around intersection

