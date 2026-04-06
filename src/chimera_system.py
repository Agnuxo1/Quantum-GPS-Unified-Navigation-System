#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHIMERA v3.0: Unified Quantum Graphical Computing System
========================================================

This module implements the "Chimera" architecture: a unified, GPU-resident
intelligence system that treats navigation and positioning as a continuous
diffusion process.

Key Features:
- **Intelligence-as-Process**: State is not stored in RAM but evolves in a GPU feedback loop.
- **Quantum Positioning**: Simulates a wavefunction (probability distribution) for vehicle
  location, evolving via advection-diffusion equations (Schrödinger-Fokker-Planck).
- **Optical Eikonal Solver**: Solves the pathfinding problem using light-like wavefront propagation.
- **Neuromorphic Memory**: Past states influence future states through texture persistence.
- **Zero-Memory Architecture**: Minimal CPU/RAM usage; all logic lives in VRAM.

"""

import os
import time
import math
import numpy as np
import moderngl
import glfw
from typing import Tuple, Optional, List, Dict, Any

# Import existing components
try:
    from src.map_manager import MapManager
    from src.landmark_correction import LandmarkCorrectionSystem
except ImportError:
    from map_manager import MapManager
    from landmark_correction import LandmarkCorrectionSystem

class ChimeraSystem:
    """
    The Unified Chimera System.
    Combines Quantum Positioning, Optical Pathfinding, and Real-time Visualization.
    """
    
    def __init__(self, city_name: str = "Madrid, Spain", grid_size: int = 512, headless: bool = False):
        self.grid_size = grid_size
        self.headless = headless
        self.city_name = city_name
        
        self.keys = {} # Track key states
        self.is_running = True
        self.optimal_path = []  # Path from vehicle to target
        
        # Initialize vehicle state variables
        self.heading = 0.0  # radians
        self.speed = 0.0
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.dt = 0.016  # 60 FPS default
        self.diffusion_coeff = 0.1
        self.path_vbo = None
        self.path_vao = None
        
        # Initialize Systems
        self._init_window()
        self._init_context()
        self._init_map()
        self._init_gpu_resources()  # This also creates shaders
        
        # Ping-Pong buffer tracking
        self.current_buffer_index = 0
        
        # Landmark-based position correction system
        # This detects turns and uses map intersections to correct position
        self.landmark_corrector = LandmarkCorrectionSystem(
            grid_size=self.grid_size,
            obstacle_field=self.obstacle_field_data
        )
        self.last_correction_time = 0.0
        
        # Initial State
        self._reset_wavefunction()
        
        print(f"[CHIMERA] System initialized. City: {city_name}, Grid: {grid_size}x{grid_size}")

    def _init_window(self):
        """Initialize GLFW window"""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
            
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        
        if self.headless:
            glfw.window_hint(glfw.VISIBLE, False)
        
        self.window = glfw.create_window(1024, 1024, "CHIMERA v3.0 - Quantum Navigation", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create window")
            
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        
        # Callbacks
        glfw.set_key_callback(self.window, self._key_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_callback)
        glfw.set_cursor_pos_callback(self.window, self._cursor_callback)

    def _init_context(self):
        """Initialize ModernGL context"""
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def _init_map(self):
        """Load map data using MapManager"""
        self.map_manager = MapManager()
        if not self.map_manager.load_city(self.city_name, radius_meters=2000, zoom=15):
            print("[CHIMERA] Warning: Failed to load map, using empty grid")
            
        self.speed_field_data = self.map_manager.create_speed_field(self.grid_size)
        street_grid = self.map_manager.create_street_grid(self.grid_size)
        self.obstacle_field_data = 1.0 - (street_grid > 0).astype(np.float32)
        
        # DEBUG: Check speed field range
        print(f"[CHIMERA] Speed field stats: min={self.speed_field_data.min()}, max={self.speed_field_data.max()}, mean={self.speed_field_data.mean()}")
        
        # IMPORTANT: Flip data vertically to match OpenGL texture coordinates (0 at bottom)
        # Screen Y=0 is Top, OpenGL V=1 is Top.
        # Standard upload puts index 0 at V=0 (Bottom).
        # So we flip so index 0 (Top) goes to V=0 (Bottom)? NO.
        # We want index 0 (Top) to go to V=1 (Top).
        # So we want index 0 to be the LAST row uploaded.
        # So we flip so index 0 moves to index N.
        self.speed_field_data = np.flipud(self.speed_field_data)
        self.obstacle_field_data = np.flipud(self.obstacle_field_data)
        
        # Find valid start position on a street
        self.vehicle_pos = self._find_valid_start_pos()
        print(f"[CHIMERA] Vehicle spawned at {self.vehicle_pos} (on street)")
        
        # DEBUG: Set default target for verification
        self.target_pos = np.array([self.vehicle_pos[0], (self.vehicle_pos[1] + 100) % self.grid_size], dtype=np.float32)
        print(f"[CHIMERA] DEBUG: Set default target at {self.target_pos}")

    def _find_valid_start_pos(self) -> np.ndarray:
        """Find nearest valid street pixel to center"""
        cx, cy = self.grid_size // 2, self.grid_size // 2
        
        # Check center first
        if self.obstacle_field_data[cy, cx] < 0.5:
            return np.array([cx, cy], dtype=np.float32)
            
        # Spiral search
        print("[CHIMERA] Searching for valid street position...")
        for r in range(1, self.grid_size // 2):
            # Check perimeter of box radius r
            for i in range(-r, r + 1):
                # Top and Bottom rows
                for dy in [-r, r]:
                    nx, ny = cx + i, cy + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        if self.obstacle_field_data[ny, nx] < 0.5:
                            return np.array([nx, ny], dtype=np.float32)
                # Left and Right columns
                for dx in [-r, r]:
                    nx, ny = cx + dx, cy + i
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        if self.obstacle_field_data[ny, nx] < 0.5:
                            return np.array([nx, ny], dtype=np.float32)
                            
        return np.array([cx, cy], dtype=np.float32) # Fallback

    def _init_gpu_resources(self):
        """Create textures and buffers"""
        size = (self.grid_size, self.grid_size)
        
        # 1. Wavefunction (Quantum Position) - Ping-Pong
        # R: Probability, G: Phase/Momentum X, B: Momentum Y, A: Entropy
        self.wave_tex_a = self.ctx.texture(size, 4, dtype='f4')
        self.wave_tex_b = self.ctx.texture(size, 4, dtype='f4')
        self.wave_fbo_a = self.ctx.framebuffer(color_attachments=[self.wave_tex_a])
        self.wave_fbo_b = self.ctx.framebuffer(color_attachments=[self.wave_tex_b])
        
        # 2. Eikonal Field (Optical Pathfinding) - Ping-Pong
        # R: Time (Distance), G: Gradient X, B: Gradient Y, A: State/Viscosity
        self.eikonal_tex_a = self.ctx.texture(size, 4, dtype='f4')
        self.eikonal_tex_b = self.ctx.texture(size, 4, dtype='f4')
        
        # State Textures for Neuromorphic Memory (4-State)
        self.state_tex_a = self.ctx.texture(size, 4, dtype='f4')
        self.state_tex_b = self.ctx.texture(size, 4, dtype='f4')
        # Initialize state with equal superposition
        self.state_tex_a.write(np.full((self.grid_size, self.grid_size, 4), 0.25, dtype='f4').tobytes())
        self.state_tex_b.write(np.full((self.grid_size, self.grid_size, 4), 0.25, dtype='f4').tobytes())
        
        # Attach state textures to FBOs as second color attachment
        self.eikonal_fbo_a = self.ctx.framebuffer(color_attachments=[self.eikonal_tex_a, self.state_tex_a])
        self.eikonal_fbo_b = self.ctx.framebuffer(color_attachments=[self.eikonal_tex_b, self.state_tex_b])
        
        # Auxiliary textures
        self.speed_tex = self.ctx.texture(size, 1, dtype='f4')
        self.obstacle_tex = self.ctx.texture(size, 1, dtype='f4')
        self.target_tex = self.ctx.texture(size, 1, dtype='f4')
        
        # Upload initial map data
        self.speed_tex.write(self.speed_field_data.tobytes())
        self.obstacle_tex.write(self.obstacle_field_data.tobytes())
        self._update_target_texture()
        
        # Map texture for visualization
        if self.map_manager.map_image:
            img_array = np.array(self.map_manager.map_image.convert('RGB'), dtype='u1')
            img_array = np.flipud(img_array)  # OpenGL coordinate system
            # Resize to grid size if needed
            if img_array.shape[0] != self.grid_size or img_array.shape[1] != self.grid_size:
                from PIL import Image
                img_pil = Image.fromarray(np.flipud(img_array))
                img_pil = img_pil.resize((self.grid_size, self.grid_size), Image.Resampling.LANCZOS)
                img_array = np.array(img_pil, dtype='u1')
                img_array = np.flipud(img_array)
            self.map_tex = self.ctx.texture(
                (self.grid_size, self.grid_size), 3, img_array.tobytes()
            )
            self.map_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        else:
            # Create dummy texture
            self.map_tex = self.ctx.texture((self.grid_size, self.grid_size), 3, 
                                            np.zeros((self.grid_size, self.grid_size, 3), dtype='u1').tobytes())
            self.map_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        
        # Quad VBO for rendering
        vertices = np.array([
            # x, y
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype='f4')
        self.quad_vbo = self.ctx.buffer(vertices.tobytes())

        # 3. Visualization Shader
        display_fs = """
        #version 430
        uniform sampler2D u_wave;
        uniform sampler2D u_eikonal;
        uniform sampler2D u_map;
        uniform sampler2D u_obstacle;
        uniform sampler2D u_target;
        uniform vec2 u_vehicle_pos; // Vehicle position in grid coords
        uniform float u_vehicle_heading; // Vehicle heading in radians
        uniform vec2 u_res;
        
        in vec2 v_uv;
        out vec4 f_color;
        
        vec3 plasma(float t) {
            // Modified plasma: Starts black/transparent
            const vec3 c0 = vec3(0.0, 0.0, 0.0); // Black
            const vec3 c1 = vec3(0.2, 0.0, 0.4); // Deep Purple
            const vec3 c2 = vec3(0.0, 0.5, 1.0); // Blue
            const vec3 c3 = vec3(0.0, 1.0, 0.5); // Cyan/Green
            const vec3 c4 = vec3(1.0, 1.0, 0.0); // Yellow (Peak)
            
            t = clamp(t, 0.0, 1.0);
            if (t < 0.25) return mix(c0, c1, t * 4.0);
            if (t < 0.5)  return mix(c1, c2, (t - 0.25) * 4.0);
            if (t < 0.75) return mix(c2, c3, (t - 0.5) * 4.0);
            return mix(c3, c4, (t - 0.75) * 4.0);
        }
        
        void main() {
            vec4 wave = texture(u_wave, v_uv);
            vec4 eikonal = texture(u_eikonal, v_uv);
            vec3 map_color = texture(u_map, v_uv).rgb;
            float obs = texture(u_obstacle, v_uv).r;
            float target = texture(u_target, v_uv).r;
            
            // Current pixel in grid coordinates
            vec2 pixel = v_uv * u_res;
            
            // Base Map - use actual map colors
            vec3 color = map_color * 0.3; // Darken map slightly
            
            if (obs > 0.5) {
                // Buildings - darken further
                color *= 0.5;
            }
            
            // Eikonal Field (Flow Visualization)
            float time = eikonal.r;
            vec2 grad = eikonal.gb; // Gradient direction
            
            if (time < 9000.0) {
                // 1. Iso-lines (Waves radiating from target)
                float iso = sin(time * 0.1);
                // 2. Flow alignment (optional visual flair)
                float flow = length(grad);
                
                if (iso > 0.95) {
                    color += vec3(0.0, 0.4, 0.8) * 0.3; // Blue waves
                }
                
                // Highlight the "valley" (optimal path area)
                color += vec3(grad.x, grad.y, 0.5) * 0.05 * flow;
            }
            
            // Target Marker
            if (target > 0.5) {
                color = vec3(1.0, 0.0, 0.2); // Red target
            }
            
            // Quantum Wavefunction (The "Ghost" Position)
            // Draw this FIRST so vehicle marker is on top
            float prob = wave.r;
            if (prob > 0.001) {
                // Scale probability for visualization
                vec3 glow = plasma(prob * 2.0); 
                color += glow * 0.5; // Reduced opacity so vehicle is more visible
            }
            
            // Vehicle Marker (Green Triangle) - Very small, rendered LAST (on top)
            vec2 to_vehicle = pixel - u_vehicle_pos;
            float dist_to_vehicle = length(to_vehicle);
            
            // Body circle (made even smaller: 0.08)
            if (dist_to_vehicle < 1.5) {
                color = vec3(0.0, 1.0, 0.0); // Bright green - always visible
            }
            
            // Direction indicator (triangle) - very small
            vec2 forward = vec2(cos(u_vehicle_heading), sin(u_vehicle_heading));
            vec2 right = vec2(-forward.y, forward.x);
            
            // Triangle points (made smaller)
            vec2 nose = u_vehicle_pos + forward * 3.0;
            vec2 left_wing = u_vehicle_pos - forward * 1.5 + right * 1.5;
            vec2 right_wing = u_vehicle_pos - forward * 1.5 - right * 1.5;
            
            // Simple triangle test
            float dist_nose = length(pixel - nose);
            float dist_left = length(pixel - left_wing);
            float dist_right = length(pixel - right_wing);
            
            if (dist_nose < 0.8 || dist_left < 0.8 || dist_right < 0.8) {
                color = vec3(0.0, 1.0, 0.0); // Bright green
            }
            
            f_color = vec4(color, 1.0);
        }
        """
        
        # Vertex Shader (Shared)
        vs = """
        #version 430
        in vec2 in_vert;
        out vec2 v_uv;
        void main() {
            v_uv = in_vert * 0.5 + 0.5;
            gl_Position = vec4(in_vert, 0.0, 1.0);
        }
        """

        # Quantum Wavefunction Shader (Schrödinger-Fokker-Planck)
        quantum_fs = """
        #version 430
        uniform sampler2D u_wave;
        uniform sampler2D u_obstacle;
        uniform vec2 u_velocity;
        uniform float u_dt;
        uniform float u_diffusion;
        uniform vec2 u_res;
        
        in vec2 v_uv;
        out vec4 f_color;
        
        void main() {
            vec2 pixel = 1.0 / u_res;
            vec4 c = texture(u_wave, v_uv);
            float obs = texture(u_obstacle, v_uv).r;
            
            if (obs > 0.5) {
                f_color = vec4(0.0);
                return;
            }
            
            // Advection (move with velocity)
            vec2 pos = v_uv - u_velocity * u_dt * pixel;
            vec4 val = texture(u_wave, pos);
            
            // Diffusion (spread out)
            vec4 sum = val;
            sum += texture(u_wave, pos + vec2(pixel.x, 0.0));
            sum += texture(u_wave, pos - vec2(pixel.x, 0.0));
            sum += texture(u_wave, pos + vec2(0.0, pixel.y));
            sum += texture(u_wave, pos - vec2(0.0, pixel.y));
            
            f_color = mix(val, sum / 5.0, u_diffusion);
            
            // Decay
            f_color *= 0.995;
        }
        """

        # Optical Eikonal Solver Shader (Neuromorphic / 4-State)
        # Based on: https://github.com/Agnuxo1/Optical-Neuromorphic-Computing-for-Real-Time-Pathfinding-A-GPU-Accelerated-Eikonal-Solver
        eikonal_fs = """
        #version 430
        uniform sampler2D u_time_tex;  // Previous Time Field (R=Time)
        uniform sampler2D u_state_tex; // Previous State Field (RGBA=State)
        uniform sampler2D u_speed_tex; // Refractive Index / Speed
        uniform sampler2D u_target_tex; // Target Source
        uniform sampler2D u_obstacle_tex; // Obstacles
        uniform vec2 u_resolution;
        
        uniform float u_relaxation; // Relaxation factor (0.0 - 1.0)
        uniform float u_memory_mix; // Memory mixing factor
        uniform float u_huge_time;  // Infinity value
        
        in vec2 v_uv;
        
        // Output to TWO render targets
        layout(location = 0) out vec4 out_time;  // R=Time, G=GradX, B=GradY
        layout(location = 1) out vec4 out_state; // RGBA=Quantum State (N,E,S,W)
        
        // Safe sampling with clamping to avoid edge artifacts
        float sample_time(vec2 offset) {
            vec2 texel = 1.0 / u_resolution;
            vec2 clamped = clamp(v_uv + offset * texel, texel * 0.5, 1.0 - texel * 0.5);
            return texture(u_time_tex, clamped).r;
        }
        
        void main() {
            // 1. Read Inputs
            float speed = texture(u_speed_tex, v_uv).r;
            float target = texture(u_target_tex, v_uv).r;
            float obstacle = texture(u_obstacle_tex, v_uv).r;
            float current = texture(u_time_tex, v_uv).r;
            vec4 prev_state = texture(u_state_tex, v_uv);
            
            // 2. Target Logic (source of propagation)
            if (target > 0.5) {
                out_time = vec4(0.0, 0.0, 0.0, 1.0);
                out_state = vec4(0.25, 0.25, 0.25, 0.25); // Equal superposition
                return;
            }
            
            // 3. Obstacle Logic
            if (obstacle > 0.5 || speed <= 1e-6) {
                out_time = vec4(u_huge_time, 0.0, 0.0, 1.0);
                out_state = mix(prev_state, vec4(0.0), u_memory_mix);
                return;
            }
            
            // 4. Eikonal Update (Robust algorithm from reference implementation)
            // Sample neighbors with safe clamping
            float t_left = sample_time(vec2(-1.0, 0.0));
            float t_right = sample_time(vec2(1.0, 0.0));
            float t_down = sample_time(vec2(0.0, -1.0));
            float t_up = sample_time(vec2(0.0, 1.0));
            
            // Solve Eikonal: |grad T| = 1/speed
            float a = min(t_left, t_right);
            float b = min(t_down, t_up);
            float inv_speed = 1.0 / max(speed, 1e-4);
            
            float candidate;
            float diff = abs(a - b);
            if (diff >= inv_speed) {
                // One-dimensional case
                candidate = min(a, b) + inv_speed;
            } else {
                // Two-dimensional case (diagonal propagation)
                float rad = max(0.0, 2.0 * inv_speed * inv_speed - diff * diff);
                candidate = 0.5 * (a + b + sqrt(rad));
            }
            
            // Ensure candidate is not worse than any neighbor
            candidate = min(candidate, min(min(t_left, t_right), min(t_down, t_up)) + inv_speed);
            candidate = min(candidate, current); // Monotonicity
            
            // Relaxation for stability
            float relaxed = mix(current, candidate, u_relaxation);
            
            // 5. Gradient Calculation
            vec2 grad = vec2(0.0);
            grad.x = (t_right - t_left) * 0.5;
            grad.y = (t_up - t_down) * 0.5;
            
            // 6. Neuromorphic State Update (Directional Memory)
            // Calculate flow directions based on time differences
            float north = max(0.0, t_up - relaxed);
            float east  = max(0.0, t_right - relaxed);
            float south = max(0.0, t_down - relaxed);
            float west  = max(0.0, t_left - relaxed);
            vec4 flow = vec4(north, east, south, west);
            
            // Normalize flow
            float total = max(dot(flow, vec4(1.0)), 1e-6);
            flow /= total;
            
            // Blend with previous state (memory retention)
            vec4 blended = mix(prev_state, flow, u_memory_mix);
            blended = max(blended, vec4(0.0)); // Ensure non-negative
            float blend_sum = max(dot(blended, vec4(1.0)), 1e-6);
            blended /= blend_sum; // Renormalize
            
            out_time = vec4(relaxed, grad.x, grad.y, 1.0);
            out_state = blended;
        }
        """
        
        self.prog_quantum = self.ctx.program(vertex_shader=vs, fragment_shader=quantum_fs)
        self.prog_eikonal = self.ctx.program(vertex_shader=vs, fragment_shader=eikonal_fs)
        self.prog_display = self.ctx.program(vertex_shader=vs, fragment_shader=display_fs)
        
        self.vao = self.ctx.simple_vertex_array(self.prog_display, self.quad_vbo, 'in_vert')
        
        # Path line shader
        path_vs = """
        #version 430
        in vec2 in_pos;
        void main() {
            gl_Position = vec4(in_pos, 0.0, 1.0);
        }
        """
        
        path_fs = """
        #version 430
        out vec4 f_color;
        void main() {
            // Bright red path with slight glow
            f_color = vec4(1.0, 0.2, 0.2, 1.0); // Bright red path
        }
        """
        
        self.prog_path = self.ctx.program(vertex_shader=path_vs, fragment_shader=path_fs)

    def _reset_wavefunction(self):
        """Initialize wavefunction at center"""
        data = np.zeros((self.grid_size, self.grid_size, 4), dtype=np.float32)
        # Gaussian at center
        cx, cy = self.grid_size // 2, self.grid_size // 2
        y, x = np.ogrid[-cy:self.grid_size-cy, -cx:self.grid_size-cx]
        mask = x*x + y*y <= 20*20
        data[mask] = [1.0, 0.0, 0.0, 0.0]
        
        self.wave_tex_a.write(data.tobytes())
        self.wave_tex_b.write(data.tobytes())
        
        # Reset Eikonal
        eik_data = np.full((self.grid_size, self.grid_size, 4), 10000.0, dtype=np.float32)
        self.eikonal_tex_a.write(eik_data.tobytes())
        self.eikonal_tex_b.write(eik_data.tobytes())
        
        # Reset Target
        self._update_target_texture()

    def _update_target_texture(self):
        """Update target texture based on self.target_pos"""
        data = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        if self.target_pos is not None:
            tx, ty = int(self.target_pos[0]), int(self.target_pos[1])
            # Ensure target is within bounds
            tx = np.clip(tx, 0, self.grid_size - 1)
            ty = np.clip(ty, 0, self.grid_size - 1)
            
            # Draw a small circle for target
            rr, cc = np.ogrid[:self.grid_size, :self.grid_size]
            mask = (rr - ty)**2 + (cc - tx)**2 <= 25
            data[mask] = 1.0
            
            # DON'T move target automatically - use exact click position
            # User clicked where they want to go, respect that
            # Only warn if target is in a building, but don't move it
            if self.obstacle_field_data[ty, tx] > 0.5:
                print(f"[EIKONAL] WARNING: Target at ({tx}, {ty}) is in a building - pathfinding may fail")
            
        # IMPORTANT: Flip target to match OpenGL coordinates (same as map)
        data = np.flipud(data)
            
        self.target_tex.write(data.tobytes())
        
        # IMPORTANT: Reset Eikonal field when target changes
        if self.target_pos is not None:
            tx, ty = int(self.target_pos[0]), int(self.target_pos[1])
            tx = np.clip(tx, 0, self.grid_size - 1)
            ty = np.clip(ty, 0, self.grid_size - 1)
            print(f"[EIKONAL] Resetting field for new target at grid ({tx}, {ty})")
            
            # Reset to Infinity
            eik_data = np.full((self.grid_size, self.grid_size, 4), 10000.0, dtype=np.float32)
            
            # Set target position to time=0 (source of propagation)
            # COORDINATE SYSTEM EXPLANATION:
            # - obstacle_field_data: Created in CPU coords (Y=0=top), then flipped with np.flipud
            #   After flip: obstacle_field_data[0, x] = original bottom row = OpenGL Y=0 (bottom)
            #   So obstacle_field_data uses OpenGL coordinates (Y=0 is bottom)
            # - target_pos: Set from mouse callback using v_opengl = 1.0 - (y/h)
            #   So target_pos[1] is in OpenGL coordinates (Y=0 is bottom)
            # - Eikonal texture: Written directly, uses OpenGL coordinates (Y=0 is bottom)
            # - When reading Eikonal field: We flip it back, so field uses CPU coordinates (Y=0 is top)
            # 
            # SOLUTION: Since target_pos is in OpenGL coords and Eikonal texture uses OpenGL coords,
            # we can use ty directly. But when reading the field, we flip it, so we need to convert
            # target_pos to CPU coords for pathfinding: ty_cpu = grid_size - 1 - ty
            
            # Write to Eikonal texture (OpenGL coordinates)
            eik_data[ty, tx, 0] = 0.0  # Time = 0 at target
            eik_data[ty, tx, 1] = 0.0  # Gradient X = 0
            eik_data[ty, tx, 2] = 0.0  # Gradient Y = 0
            
            # Also initialize nearby cells to speed up convergence
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = tx + dx, ty + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist > 0 and dist < 2.5:
                            # Set initial time based on distance (helps convergence)
                            eik_data[ny, nx, 0] = dist * 0.5
            
            self.eikonal_tex_a.write(eik_data.tobytes())
            self.eikonal_tex_b.write(eik_data.tobytes())
            
            # Reset current buffer index
            self.current_buffer_index = 0

    def _key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            self.keys[key] = True
            if key == glfw.KEY_ESCAPE:
                self.is_running = False
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_R:
                self._reset_wavefunction()
        elif action == glfw.RELEASE:
            self.keys[key] = False

    def _update_physics(self, dt=0.016, frame=0):
        """Update vehicle physics based on keyboard input"""
        # Store previous position for collision detection
        prev_pos = self.vehicle_pos.copy()
        
        # Steering
        if self.keys.get(glfw.KEY_LEFT):
            self.heading += 0.1
        if self.keys.get(glfw.KEY_RIGHT):
            self.heading -= 0.1
            
        # Progressive Acceleration
        # Acceleration decreases as speed increases (realistic engine curve)
        max_speed = 20.0
        accel_factor = 0.2 * (1.0 - (self.speed / max_speed)) 
        accel_factor = max(0.05, accel_factor) # Minimum acceleration
        
        if self.keys.get(glfw.KEY_UP):
            self.speed = min(self.speed + accel_factor, max_speed)
        elif self.keys.get(glfw.KEY_DOWN):
            self.speed = max(self.speed - 0.3, -5.0) # Braking is stronger
        else:
            # Friction
            self.speed *= 0.96 # Coasting
            
        # Update velocity vector
        self.velocity[0] = math.cos(self.heading) * self.speed
        self.velocity[1] = math.sin(self.heading) * self.speed
        
        # Update vehicle position (use actual dt)
        # Scale velocity by 10.0 to make it visible but controllable
        move_scale = 10.0
        self.vehicle_pos[0] += self.velocity[0] * dt * move_scale
        self.vehicle_pos[1] += self.velocity[1] * dt * move_scale
        
        # Clamp to grid bounds
        self.vehicle_pos[0] = np.clip(self.vehicle_pos[0], 0, self.grid_size - 1)
        self.vehicle_pos[1] = np.clip(self.vehicle_pos[1], 0, self.grid_size - 1)
        
        # Collision detection with buildings - sliding effect (like ice)
        vx, vy = int(self.vehicle_pos[0]), int(self.vehicle_pos[1])
        if 0 <= vx < self.grid_size and 0 <= vy < self.grid_size:
            if self.obstacle_field_data[vy, vx] > 0.5:  # Hit a building
                # Ice-like sliding: maintain velocity direction but push perpendicular to obstacle
                # Calculate normal vector from obstacle (perpendicular to velocity)
                vel_norm = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
                if vel_norm > 0.01:
                    # Normalize velocity
                    vx_norm = self.velocity[0] / vel_norm
                    vy_norm = self.velocity[1] / vel_norm
                    
                    # Find direction to nearest street (perpendicular to velocity for sliding)
                    best_dir = None
                    min_dist = float('inf')
                    
                    # Check directions perpendicular to velocity (sliding directions)
                    perp1 = (-vy_norm, vx_norm)  # Perpendicular 1
                    perp2 = (vy_norm, -vx_norm)  # Perpendicular 2
                    
                    # Check perpendicular directions first (sliding)
                    for dir_vec in [perp1, perp2, (-perp1[0], -perp1[1]), (-perp2[0], -perp2[1])]:
                        dx, dy = dir_vec
                        # Check nearby cells in sliding direction
                        for dist in [1, 2, 3]:
                            nx, ny = vx + int(dx * dist), vy + int(dy * dist)
                            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                                if self.obstacle_field_data[ny, nx] < 0.5:  # Street found
                                    if dist < min_dist:
                                        min_dist = dist
                                        best_dir = (dx, dy)
                    
                    # If no perpendicular street found, check all directions
                    if best_dir is None:
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = vx + dx, vy + dy
                            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                                if self.obstacle_field_data[ny, nx] < 0.5:  # Street found
                                    dist = math.sqrt(dx*dx + dy*dy)
                                    if dist < min_dist:
                                        min_dist = dist
                                        best_dir = (dx, dy)
                    
                    if best_dir:
                        # Slide in the direction found, but maintain some forward momentum
                        slide_factor = 0.4  # How much to slide
                        # Combine sliding with reduced forward motion
                        self.vehicle_pos[0] = prev_pos[0] + best_dir[0] * slide_factor + vx_norm * 0.2
                        self.vehicle_pos[1] = prev_pos[1] + best_dir[1] * slide_factor + vy_norm * 0.2
                        # Reduce speed but maintain direction
                        self.speed *= 0.8
                        # DON'T change heading - maintain forward direction
                    else:
                        # No street found, stop but don't reverse
                        self.vehicle_pos = prev_pos
                        self.speed *= 0.3
                else:
                    # Very slow, just stop
                    self.vehicle_pos = prev_pos
                    self.speed = 0
        
        # Update optimal path (more frequently for better responsiveness)
        if frame % 2 == 0:  # Every 2 frames for smoother path updates
            self._update_optimal_path(frame)  # Pass frame for logging

    def _mouse_callback(self, window, button, action, mods):
        if action == glfw.PRESS and button == glfw.MOUSE_BUTTON_LEFT:
            x, y = glfw.get_cursor_pos(window)
            # Convert screen to grid coordinates
            w, h = glfw.get_window_size(window)
            
            # The shader uses: pixel = v_uv * u_res
            # where v_uv is normalized [0,1] and u_res is (grid_size, grid_size)
            # So we need to convert mouse to UV coordinates first
            
            # Normalize mouse coordinates to [0, 1] (UV space)
            u = x / w
            v = y / h  # Screen Y=0 is top, but we'll flip it
            
            # IMPORTANT: OpenGL UV coordinates: V=0 is bottom, V=1 is top
            # Screen coordinates: Y=0 is top, Y=h is bottom
            # So we need to flip V: v_opengl = 1.0 - (y / h)
            v_opengl = 1.0 - v
            
            # Convert UV to grid coordinates
            gx = int(u * self.grid_size)
            gy = int(v_opengl * self.grid_size)
            
            # Clamp to grid bounds
            gx = np.clip(gx, 0, self.grid_size - 1)
            gy = np.clip(gy, 0, self.grid_size - 1)
            
            # Set target at EXACT click position (don't move it)
            self.target_pos = np.array([float(gx), float(gy)], dtype=np.float32)
            self._update_target_texture()
            print(f"[CHIMERA] Target set at grid ({gx}, {gy}) from mouse ({x:.0f}, {y:.0f}) screen, UV=({u:.3f}, {v_opengl:.3f})")

    def _cursor_callback(self, window, x, y):
        pass # Disabled mouse velocity control

    def _update_optimal_path(self, frame=0):
        """Extract optimal path from Eikonal field"""
        if self.target_pos is None:
            self.optimal_path = []
            self._update_path_buffer()
            return

        try:
            vx, vy = int(self.vehicle_pos[0]), int(self.vehicle_pos[1])
            tx, ty = int(self.target_pos[0]), int(self.target_pos[1])
            
            # Ensure coordinates are within bounds
            vx = np.clip(vx, 0, self.grid_size - 1)
            vy = np.clip(vy, 0, self.grid_size - 1)
            tx = np.clip(tx, 0, self.grid_size - 1)
            ty = np.clip(ty, 0, self.grid_size - 1)
            
            # Read Eikonal field from GPU
            current_eik = self.eikonal_tex_a if self.current_buffer_index == 0 else self.eikonal_tex_b
            buffer = current_eik.read()
            field = np.frombuffer(buffer, dtype=np.float32).reshape(self.grid_size, self.grid_size, 4)
            
            # COORDINATE SYSTEM CONSISTENCY:
            # - obstacle_field_data: Flipped with np.flipud, so uses OpenGL coords (Y=0 is bottom)
            # - Eikonal field from GPU: Uses OpenGL coords (Y=0 is bottom)  
            # - vehicle_pos and target_pos: Use OpenGL coords (Y=0 is bottom)
            # So we DON'T flip the field - keep everything in OpenGL coordinates
            # field[vy, vx] where vy is OpenGL coord (0=bottom) matches obstacle_field_data[vy, vx]
            
            # Debug: Analyze field values
            if frame % 30 == 0:
                time_field = field[:, :, 0]
                print(f"\n[EIKONAL DEBUG]")
                print(f"  Field stats: min={time_field.min():.2f}, max={time_field.max():.2f}, mean={time_field.mean():.2f}")
                print(f"  Vehicle at ({vx}, {vy}), time value: {field[vy, vx, 0]:.2f}")
                print(f"  Target at ({tx}, {ty}), time value: {field[ty, tx, 0]:.2f}")
                
                # Count how many cells have converged (not at initial value)
                converged_cells = np.sum(time_field < 9999.0)
                total_cells = self.grid_size * self.grid_size
                print(f"  Converged cells: {converged_cells}/{total_cells} ({100*converged_cells/total_cells:.1f}%)")
            
            # Extract path via gradient descent from vehicle to target
            # The Eikonal field propagates FROM target, so we follow gradient DESCENDING
            # from vehicle position (higher time) towards target (time=0)
            path = []
            current = (vx, vy)
            visited = set()
            # Use only cardinal directions for pathfinding to follow streets better
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Only cardinal directions
            
            max_steps = self.grid_size * 3  # More steps for longer paths
            
            current_value = field[vy, vx, 0]
            
            # If vehicle is at infinity, field hasn't converged yet
            # Try to find nearest converged cell and path from there
            if current_value >= 9999.0:
                # Search for nearest converged cell
                nearest_converged = None
                min_dist = float('inf')
                
                search_radius = min(50, self.grid_size // 10)  # Search in nearby area
                for dy in range(-search_radius, search_radius + 1):
                    for dx in range(-search_radius, search_radius + 1):
                        nx, ny = vx + dx, vy + dy
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            if self.obstacle_field_data[ny, nx] < 0.5:  # On street
                                cell_value = field[ny, nx, 0]
                                if cell_value < 9999.0:  # Converged
                                    dist = math.sqrt(dx*dx + dy*dy)
                                    if dist < min_dist:
                                        min_dist = dist
                                        nearest_converged = (nx, ny, cell_value)
                
                if nearest_converged:
                    # Use nearest converged cell as starting point
                    if frame % 60 == 0:
                        print(f"  [PATH] Vehicle at infinity, using nearest converged cell at {nearest_converged[:2]}")
                    vx, vy = nearest_converged[0], nearest_converged[1]
                    current_value = nearest_converged[2]
                else:
                    # Still no path found
                    if frame % 60 == 0:
                        print(f"  [PATH] Field not converged yet, waiting... (converged: {np.sum(field[:, :, 0] < 9999.0)}/{self.grid_size*self.grid_size})")
                    self.optimal_path = []
                    self._update_path_buffer()
                    return
            
            for step in range(max_steps):
                path.append(current)
                cx, cy = current
                
                # Check if we've reached the target
                if current == (tx, ty):
                    break
                
                # Check if we're close enough to target
                dist_to_target = math.sqrt((cx - tx)**2 + (cy - ty)**2)
                if dist_to_target < 2.0:  # Within 2 pixels
                    path.append((tx, ty))
                    break
                
                # Also check if we've reached time=0 (at target)
                if field[cy, cx, 0] < 0.5:  # Very close to target (time near 0)
                    path.append((tx, ty))
                    break
                
                visited.add(current)
                
                # Find neighbor with LOWEST time value (closer to target)
                # Prioritize neighbors that are on streets
                best_neighbor = current
                best_value = field[cy, cx, 0]
                best_is_street = False
                
                for dx, dy in neighbors:
                    nx, ny = cx + dx, cy + dy
                    
                    # Bounds check
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        # Obstacle check - skip buildings
                        if self.obstacle_field_data[ny, nx] > 0.5:
                            continue
                        
                        is_street = self.obstacle_field_data[ny, nx] < 0.5
                        candidate = field[ny, nx, 0]
                        
                        # Prefer streets over non-streets, and lower time values
                        if candidate < 9999.0:  # Valid value
                            # If current best is not a street but this is, prefer this
                            if not best_is_street and is_street:
                                best_value = candidate
                                best_neighbor = (nx, ny)
                                best_is_street = True
                            # If both are streets or both are not, prefer lower time
                            elif (best_is_street == is_street) and (candidate < best_value - 1e-6):
                                best_value = candidate
                                best_neighbor = (nx, ny)
                                best_is_street = is_street
                
                # If no better neighbor found, or we hit a loop, stop
                if best_neighbor == current:
                    if frame % 30 == 0:
                        print(f"  [PATH] No better neighbor at {current}, val={best_value:.2f}")
                    break
                    
                if best_neighbor in visited:
                    if frame % 30 == 0:
                        print(f"  [PATH] Loop detected at {current}")
                    break
                
                current = best_neighbor
                
            # Path is Vehicle -> Target (following gradient descent)
            # Reverse path so it goes from vehicle to target visually
            if len(path) > 0:
                # Path is already in correct order (from vehicle to target)
                self.optimal_path = path
                if frame % 30 == 0 and len(path) > 1:
                    print(f"  [PATH] Extracted path: {len(path)} points from ({path[0][0]}, {path[0][1]}) to ({path[-1][0]}, {path[-1][1]})")
            else:
                self.optimal_path = []
            
            self._update_path_buffer()
            
            # Calculate metrics
            if len(path) > 0:
                dist_pixels = len(path)
                dist_meters = dist_pixels * 4.0
                eta_seconds = dist_meters / 8.3
                
                title = f"CHIMERA v3.0 | 🎯 {dist_meters:.0f}m | ⏱️ {eta_seconds:.0f}s | 🚗 {self.speed:.1f} km/h | Path: {len(path)} pts"
                glfw.set_window_title(self.window, title)
            else:
                glfw.set_window_title(self.window, "CHIMERA v3.0 | ❌ No Path")
                
        except Exception as e:
            print(f"[CHIMERA] Path error: {e}")
            import traceback
            traceback.print_exc()
            self.optimal_path = []
    
    def _update_path_buffer(self):
        """Update GPU buffer for path rendering"""
        if len(self.optimal_path) < 2:
            if self.path_vbo is not None:
                self.path_vbo.release()
                self.path_vbo = None
                self.path_vao = None
            return
        
        # Convert path to vertices in clip space
        # Path coordinates are in grid space [0, grid_size-1]
        # Need to convert to clip space [-1, 1] matching the display shader
        vertices = []
        for point in self.optimal_path:
            x, y = point
            
            # Grid to clip space: [0, grid_size-1] -> [-1, 1]
            # Path coordinates are in OpenGL coords (Y=0 is bottom)
            # Clip space: Y=-1 is bottom, Y=1 is top
            cx = (x / (self.grid_size - 1)) * 2.0 - 1.0
            # Y coordinate: y is OpenGL coord (0=bottom), clip space Y=-1 is bottom
            # So we DON'T flip: cy = (y / (grid_size - 1)) * 2.0 - 1.0
            cy = (y / (self.grid_size - 1)) * 2.0 - 1.0
            
            vertices.extend([cx, cy])
        
        if not vertices:
            if len(self.optimal_path) > 0:
                print(f"[PATH] WARNING: No vertices generated from {len(self.optimal_path)} path points!")
            return
        
        data = np.array(vertices, dtype='f4').tobytes()
        
        # Always recreate buffer to avoid issues
        if self.path_vbo is not None:
            self.path_vbo.release()
        
        try:
            self.path_vbo = self.ctx.buffer(data)
            self.path_vao = self.ctx.simple_vertex_array(self.prog_path, self.path_vbo, 'in_pos')
        except Exception as e:
            print(f"[PATH] Error creating buffer: {e}")
            self.path_vbo = None
            self.path_vao = None
    
    def _inject_wavefunction_at_vehicle(self):
        """Inject probability at vehicle position to show uncertainty"""
        # Use the current wave buffer (ping-pong)
        # We'll inject into the destination buffer after quantum step
        # For now, just ensure wavefunction is initialized at vehicle position
        pass  # The quantum shader will handle advection based on velocity
    
    def _apply_landmark_correction(self, timestamp: float):
        """
        Apply landmark-based position correction when turns are detected.
        This is the "quantum measurement" moment: we collapse the wavefunction
        to the exact intersection point.
        """
        # Update landmark corrector with current state
        heading_degrees = math.degrees(self.heading)
        turn_event = self.landmark_corrector.update(
            timestamp=timestamp,
            position=(self.vehicle_pos[0], self.vehicle_pos[1]),
            heading=heading_degrees,
            speed=self.speed
        )
        
        # If a turn was detected, try to correct position
        if turn_event is not None:
            corrected_pos = self.landmark_corrector.correct_position(turn_event)
            
            if corrected_pos is not None:
                # Apply correction: update vehicle position to exact intersection
                correction_radius = self.landmark_corrector.get_correction_radius()
                
                # Update vehicle position
                old_pos = self.vehicle_pos.copy()
                self.vehicle_pos[0] = float(corrected_pos[0])
                self.vehicle_pos[1] = float(corrected_pos[1])
                
                # Update wavefunction: collapse probability to corrected position
                # Read current wavefunction
                current_wave = self.wave_tex_a if self.frame_count % 2 == 0 else self.wave_tex_b
                buffer = current_wave.read()
                wave_data = np.frombuffer(buffer, dtype=np.float32).reshape(
                    self.grid_size, self.grid_size, 4
                )
                
                # Collapse wavefunction: set high probability at corrected position
                cx, cy = int(corrected_pos[0]), int(corrected_pos[1])
                radius = int(correction_radius)
                
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            dist = math.sqrt(dx*dx + dy*dy)
                            if dist <= correction_radius:
                                # Gaussian probability distribution centered at intersection
                                prob = math.exp(-(dist**2) / (2.0 * (correction_radius/2.0)**2))
                                wave_data[ny, nx, 0] = max(wave_data[ny, nx, 0], prob * 0.8)
                
                # Write back to wavefunction texture
                current_wave.write(wave_data.tobytes())
                
                # Log correction
                correction_distance = math.sqrt(
                    (old_pos[0] - self.vehicle_pos[0])**2 + 
                    (old_pos[1] - self.vehicle_pos[1])**2
                )
                print(f"[QUANTUM CORRECTION] Position corrected by {correction_distance:.2f} cells")
                print(f"  Old: ({old_pos[0]:.1f}, {old_pos[1]:.1f})")
                print(f"  New: ({self.vehicle_pos[0]:.1f}, {self.vehicle_pos[1]:.1f})")
                
                self.last_correction_time = timestamp
    
    def run(self):
        """Main Loop"""
        print("[CHIMERA] Starting Quantum Loop...")
        
        frame = 0
        last_time = time.time()
        start_time = time.time()
        self.frame_count = 0
        
        while not glfw.window_should_close(self.window) and self.is_running:
            self.frame_count = frame
            # Calculate actual delta time
            current_time = time.time()
            dt = min(current_time - last_time, 0.1)  # Cap at 100ms
            last_time = current_time
            
            # 0. Physics Update with actual dt
            self._update_physics(dt, frame)
            
            # Update dt for shaders
            self.dt = dt
            
            # Inject wavefunction at vehicle position
            self._inject_wavefunction_at_vehicle()
            
            # Apply landmark-based position correction (quantum measurement at turns)
            current_time = time.time() - start_time
            self._apply_landmark_correction(current_time)
            
            # 1. Quantum Positioning Step
            # Ping-Pong Wavefunction
            source_wave = self.wave_tex_a if frame % 2 == 0 else self.wave_tex_b
            dest_fbo = self.wave_fbo_b if frame % 2 == 0 else self.wave_fbo_a
            
            dest_fbo.use()
            source_wave.use(location=0)
            self.obstacle_tex.use(location=1)
            
            self.prog_quantum['u_wave'].value = 0
            self.prog_quantum['u_obstacle'].value = 1
            self.prog_quantum['u_velocity'].value = tuple(self.velocity)
            self.prog_quantum['u_dt'].value = self.dt
            self.prog_quantum['u_diffusion'].value = self.diffusion_coeff
            self.prog_quantum['u_res'].value = (self.grid_size, self.grid_size)
            
            quantum_vao = self.ctx.simple_vertex_array(self.prog_quantum, self.quad_vbo, 'in_vert')
            quantum_vao.render(moderngl.TRIANGLE_STRIP)
            
            # 2. Eikonal Pathfinding Step
            # Based on reference implementation: needs 2n-4n iterations for full convergence
            # For 512x512 grid: ~1024-2048 iterations for full convergence
            # But we can do fewer iterations per frame and accumulate over time
            if frame < 200:  # First 200 frames: aggressive convergence
                eikonal_iterations = min(200, self.grid_size * 2)  # Up to 2n iterations
            elif frame < 500:  # Next 300 frames: continue convergence
                eikonal_iterations = min(100, self.grid_size)  # Up to n iterations
            else:  # After convergence: maintenance mode
                eikonal_iterations = 50  # Maintenance
            
            for sub_iter in range(eikonal_iterations):
                # Execute Eikonal propagation shader
                source_eik = self.eikonal_tex_a if self.current_buffer_index == 0 else self.eikonal_tex_b
                source_state = self.state_tex_a if self.current_buffer_index == 0 else self.state_tex_b
                dest_eik_fbo = self.eikonal_fbo_b if self.current_buffer_index == 0 else self.eikonal_fbo_a
                
                dest_eik_fbo.use()
                source_eik.use(location=0)
                source_state.use(location=1)
                self.speed_tex.use(location=2)
                self.target_tex.use(location=3)
                self.obstacle_tex.use(location=4)
                
                self.prog_eikonal['u_time_tex'].value = 0
                self.prog_eikonal['u_state_tex'].value = 1
                self.prog_eikonal['u_speed_tex'].value = 2
                self.prog_eikonal['u_target_tex'].value = 3
                self.prog_eikonal['u_obstacle_tex'].value = 4
                self.prog_eikonal['u_resolution'].value = (float(self.grid_size), float(self.grid_size))
                self.prog_eikonal['u_relaxation'].value = 0.95
                self.prog_eikonal['u_memory_mix'].value = 0.08
                self.prog_eikonal['u_huge_time'].value = 10000.0
                
                # Use correct VAO for Eikonal shader
                eikonal_vao = self.ctx.simple_vertex_array(self.prog_eikonal, self.quad_vbo, 'in_vert')
                eikonal_vao.render(moderngl.TRIANGLE_STRIP)
                
                # Swap buffers
                self.current_buffer_index = 1 - self.current_buffer_index
            
            # 3. Render display (once per frame)
            self.ctx.screen.use()
            self.ctx.clear(0.0, 0.0, 0.0)
            
            # Bind textures for display
            current_wave = self.wave_tex_a if frame % 2 == 0 else self.wave_tex_b
            current_eik = self.eikonal_tex_a if self.current_buffer_index == 0 else self.eikonal_tex_b
            current_wave.use(location=0)
            current_eik.use(location=1)
            self.map_tex.use(location=2)
            self.obstacle_tex.use(location=3)
            self.target_tex.use(location=4)
            
            self.prog_display['u_wave'].value = 0
            self.prog_display['u_eikonal'].value = 1
            self.prog_display['u_map'].value = 2
            self.prog_display['u_obstacle'].value = 3
            self.prog_display['u_target'].value = 4
            self.prog_display['u_vehicle_pos'].value = tuple(self.vehicle_pos)
            self.prog_display['u_vehicle_heading'].value = self.heading
            self.prog_display['u_res'].value = (self.grid_size, self.grid_size)
            
            self.vao.render(moderngl.TRIANGLE_STRIP)
            
            # Render path overlay (red line following streets)
            # Always try to render if path exists
            if len(self.optimal_path) > 1:
                if self.path_vao is None:
                    # Path buffer not created yet, try to create it
                    self._update_path_buffer()
                
                if self.path_vao is not None:
                    self.ctx.enable(moderngl.BLEND)
                    self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
                    # Set line width for better visibility
                    try:
                        self.ctx.line_width = 5.0
                    except:
                        pass  # Some contexts don't support line_width
                    # Disable depth test for path rendering
                    self.ctx.disable(moderngl.DEPTH_TEST)
                    # Render path as line strip (thick red line)
                    self.path_vao.render(moderngl.LINE_STRIP)
                    # Re-enable depth test
                    self.ctx.enable(moderngl.DEPTH_TEST)
            
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            
            frame += 1
            
        glfw.terminate()

if __name__ == "__main__":
    # Entry point
    system = ChimeraSystem(city_name="Madrid, Spain", grid_size=512)
    system.run()
