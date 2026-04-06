#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified GPU Eikonal Solver with Quantum Qubits (4-state simulation)
===================================================================

GPU-accelerated pathfinding using optical neuromorphic architecture.
- 4-state directional memory per cell (quantum-inspired qubits)
- Real-time wavefront propagation via ModernGL
- Automatic path extraction via gradient descent
- Optimized for RTX 3090
"""

import numpy as np
import moderngl
import glfw
from typing import List, Tuple, Optional
import heapq
import math


class GpuEikonalSolver:
    """GPU-based Eikonal solver with quantum 4-state architecture"""
    
    def __init__(self, grid_size: int = 512, headless: bool = False):
        """
        Initialize GPU Eikonal solver
        
        Args:
            grid_size: Resolution of computation grid
            headless: If True, no window rendering (compute only)
        """
        self.grid_size = grid_size
        self.headless = headless
        self.huge_time = 1.0e6
        self.relaxation = 0.95
        self.memory_mix = 0.08
        self.iteration_counter = 0
        
        # State
        self.source_pos = (grid_size // 8, grid_size // 2)
        self.target_pos = (grid_size - grid_size // 8, grid_size // 2)
        self.path_points: List[Tuple[int, int]] = []
        self.path_length: Optional[float] = None
        self.path_cost: Optional[float] = None
        
        # Data fields
        self.speed_field = np.ones((grid_size, grid_size), dtype=np.float32)
        self.obstacle_field = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.source_field = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.target_field = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        self.source_field[self.source_pos[1], self.source_pos[0]] = 1.0
        self.target_field[self.target_pos[1], self.target_pos[0]] = 1.0
        
        # GPU resources
        self.window = None
        self.ctx = None
        self.time_textures = []
        self.state_textures = []
        self.framebuffers = []
        self.current_buffer_index = 0
        self.time_range = (0.0, 1.0)
        
        self._init_gpu(headless)
    
    def _init_gpu(self, headless: bool) -> None:
        """Initialize GPU and context"""
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        
        if headless:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            self.window = glfw.create_window(1, 1, "", None, None)
        else:
            self.window = glfw.create_window(
                self.grid_size * 2, self.grid_size * 2,
                "GPU Eikonal Solver", None, None
            )
        
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        
        glfw.make_context_current(self.window)
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND)
        
        # Create textures and shaders
        self._create_textures()
        self._create_programs()
        self._create_quad()
        self._upload_static_textures()
        self._reset_time_field()
    
    def _create_textures(self) -> None:
        """Create GPU textures for time and state fields"""
        size = (self.grid_size, self.grid_size)
        
        for _ in range(2):
            # Time field texture (R32F)
            time_tex = self.ctx.texture(size, 1, dtype='f4')
            time_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            time_tex.repeat_x = False
            time_tex.repeat_y = False
            
            # State texture (RGBA32F) - 4 quantum states per cell
            state_tex = self.ctx.texture(size, 4, dtype='f4')
            state_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            state_tex.repeat_x = False
            state_tex.repeat_y = False
            
            fbo = self.ctx.framebuffer(color_attachments=[time_tex, state_tex])
            
            self.time_textures.append(time_tex)
            self.state_textures.append(state_tex)
            self.framebuffers.append(fbo)
        
        # Static textures
        self.speed_texture = self.ctx.texture(size, 1, dtype='f4')
        self.speed_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        
        self.source_texture = self.ctx.texture(size, 1, dtype='f4')
        self.source_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        
        self.target_texture = self.ctx.texture(size, 1, dtype='f4')
        self.target_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        
        self.obstacle_texture = self.ctx.texture(size, 1, dtype='f4')
        self.obstacle_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
    
    def _create_programs(self) -> None:
        """Create GPU compute shaders"""
        vs = """
        #version 430 core
        in vec2 in_pos;
        out vec2 v_uv;
        void main() {
            v_uv = in_pos * 0.5 + 0.5;
            gl_Position = vec4(in_pos, 0.0, 1.0);
        }
        """
        
        # Propagation shader (4-qubit neuromorphic)
        propagation_fs = """
        #version 430 core
        
        in vec2 v_uv;
        layout(location = 0) out float out_time;
        layout(location = 1) out vec4 out_state;
        
        uniform sampler2D u_time_tex;
        uniform sampler2D u_state_tex;
        uniform sampler2D u_speed_tex;
        uniform sampler2D u_source_tex;
        uniform sampler2D u_obstacle_tex;
        
        uniform vec2 u_resolution;
        uniform float u_relaxation;
        uniform float u_memory_mix;
        uniform float u_huge_time;
        
        float sample_time(vec2 offset) {
            vec2 texel = 1.0 / u_resolution;
            vec2 clamped = clamp(v_uv + offset * texel, texel * 0.5, 1.0 - texel * 0.5);
            return texture(u_time_tex, clamped).r;
        }
        
        void main() {
            float speed = texture(u_speed_tex, v_uv).r;
            float source = texture(u_source_tex, v_uv).r;
            float obstacle = texture(u_obstacle_tex, v_uv).r;
            float current = texture(u_time_tex, v_uv).r;
            vec4 prev_state = texture(u_state_tex, v_uv);
            
            // Source point
            if (source > 0.5) {
                out_time = 0.0;
                out_state = vec4(0.25, 0.25, 0.25, 0.25);  // Equal superposition
                return;
            }
            
            // Obstacle/blocked
            if (obstacle > 0.5 || speed <= 1e-6) {
                out_time = u_huge_time;
                out_state = mix(prev_state, vec4(0.0), u_memory_mix);
                return;
            }
            
            // Sample neighbors
            float t_left = sample_time(vec2(-1.0, 0.0));
            float t_right = sample_time(vec2(1.0, 0.0));
            float t_down = sample_time(vec2(0.0, -1.0));
            float t_up = sample_time(vec2(0.0, 1.0));
            
            // Eikonal equation solve (fast marching)
            float a = min(t_left, t_right);
            float b = min(t_down, t_up);
            float inv_speed = 1.0 / max(speed, 1e-4);
            
            float candidate;
            float diff = abs(a - b);
            if (diff >= inv_speed) {
                candidate = min(a, b) + inv_speed;
            } else {
                float rad = max(0.0, 2.0 * inv_speed * inv_speed - diff * diff);
                candidate = 0.5 * (a + b + sqrt(rad));
            }
            
            candidate = min(candidate, current);
            float relaxed = mix(current, candidate, u_relaxation);
            out_time = relaxed;
            
            // 4-state quantum directional memory (neuromorphic guidance)
            float north = max(0.0, t_up - relaxed);
            float east = max(0.0, t_right - relaxed);
            float south = max(0.0, t_down - relaxed);
            float west = max(0.0, t_left - relaxed);
            
            vec4 flow = vec4(north, east, south, west);
            float total = max(dot(flow, vec4(1.0)), 1e-6);
            flow /= total;
            
            // Blend with previous state (neuromorphic decay)
            vec4 blended = mix(prev_state, flow, u_memory_mix);
            blended = max(blended, vec4(0.0));
            float blend_sum = max(dot(blended, vec4(1.0)), 1e-6);
            out_state = blended / blend_sum;
        }
        """
        
        self.propagation_program = self.ctx.program(
            vertex_shader=vs, fragment_shader=propagation_fs
        )
        self.propagation_vao = self.ctx.simple_vertex_array(
            self.propagation_program, self.quad_buffer, 'in_pos'
        )
    
    def _create_quad(self) -> None:
        """Create fullscreen quad"""
        quad_vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
             1.0,  1.0,
            -1.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0,
        ], dtype='f4')
        self.quad_buffer = self.ctx.buffer(quad_vertices.tobytes())
    
    def _upload_static_textures(self) -> None:
        """Upload static field textures to GPU"""
        self.speed_texture.write(self.speed_field.tobytes())
        self.source_texture.write(self.source_field.tobytes())
        self.target_texture.write(self.target_field.tobytes())
        self.obstacle_texture.write(self.obstacle_field.tobytes())
        
        # Initial quantum state: equal superposition
        initial_state = np.full((self.grid_size, self.grid_size, 4), 0.25, dtype=np.float32)
        state_bytes = initial_state.tobytes()
        for tex in self.state_textures:
            tex.write(state_bytes)
    
    def _reset_time_field(self) -> None:
        """Reset time field for new computation"""
        base = np.full((self.grid_size, self.grid_size), self.huge_time, dtype=np.float32)
        base[self.source_pos[1], self.source_pos[0]] = 0.0
        data = base.tobytes()
        for tex in self.time_textures:
            tex.write(data)
        self.iteration_counter = 0
    
    def _set_uniform(self, program: moderngl.Program, name: str, value) -> None:
        """Set shader uniform"""
        try:
            program[name].value = value
        except KeyError:
            pass
    
    def _propagation_step(self) -> None:
        """Execute one GPU propagation iteration"""
        read_idx = self.current_buffer_index
        write_idx = 1 - read_idx
        
        # Bind input textures
        self.time_textures[read_idx].use(location=0)
        self.state_textures[read_idx].use(location=1)
        self.speed_texture.use(location=2)
        self.source_texture.use(location=3)
        self.obstacle_texture.use(location=4)
        
        # Set uniforms
        self._set_uniform(self.propagation_program, 'u_time_tex', 0)
        self._set_uniform(self.propagation_program, 'u_state_tex', 1)
        self._set_uniform(self.propagation_program, 'u_speed_tex', 2)
        self._set_uniform(self.propagation_program, 'u_source_tex', 3)
        self._set_uniform(self.propagation_program, 'u_obstacle_tex', 4)
        self._set_uniform(self.propagation_program, 'u_resolution', 
                         (float(self.grid_size), float(self.grid_size)))
        self._set_uniform(self.propagation_program, 'u_relaxation', float(self.relaxation))
        self._set_uniform(self.propagation_program, 'u_memory_mix', float(self.memory_mix))
        self._set_uniform(self.propagation_program, 'u_huge_time', float(self.huge_time))
        
        # Render to output framebuffer
        self.framebuffers[write_idx].use()
        self.propagation_vao.render(moderngl.TRIANGLES)
        
        self.current_buffer_index = write_idx
        self.iteration_counter += 1
    
    def set_source_target(self, source: Tuple[int, int], target: Tuple[int, int]) -> None:
        """Set source and target points"""
        self.source_pos = source
        self.target_pos = target
        
        self.source_field.fill(0.0)
        self.target_field.fill(0.0)
        
        sx, sy = np.clip(source[0], 0, self.grid_size - 1), np.clip(source[1], 0, self.grid_size - 1)
        tx, ty = np.clip(target[0], 0, self.grid_size - 1), np.clip(target[1], 0, self.grid_size - 1)
        
        self.source_field[sy, sx] = 1.0
        self.target_field[ty, tx] = 1.0
        
        self.source_texture.write(self.source_field.tobytes())
        self.target_texture.write(self.target_field.tobytes())
        self._reset_time_field()
    
    def set_obstacle_field(self, obstacles: np.ndarray) -> None:
        """Set obstacle field"""
        self.obstacle_field = np.clip(obstacles, 0.0, 1.0).astype(np.float32)
        self.obstacle_texture.write(self.obstacle_field.tobytes())
    
    def set_speed_field(self, speeds: np.ndarray) -> None:
        """Set speed field"""
        self.speed_field = np.clip(speeds, 0.01, 2.0).astype(np.float32)
        self.speed_texture.write(self.speed_field.tobytes())
    
    def compute_path(self, num_iterations: int = None) -> List[Tuple[int, int]]:
        """Compute path via GPU propagation"""
        if num_iterations is None:
            num_iterations = self.grid_size * 2
        
        print(f"   [GpuEikonal] Running {num_iterations} propagation iterations...")
        
        for _ in range(num_iterations):
            self._propagation_step()
        
        # Extract path
        field = self._read_time_field()
        self.path_points = self._extract_path(field)
        self._compute_path_metrics()
        
        print(f"   [GpuEikonal] Path found: {len(self.path_points)} waypoints")
        return self.path_points
    
    def _read_time_field(self) -> np.ndarray:
        """Read time field from GPU"""
        tex = self.time_textures[self.current_buffer_index]
        buffer = tex.read()
        field = np.frombuffer(buffer, dtype=np.float32).reshape(
            self.grid_size, self.grid_size
        )
        return field
    
    def _extract_path(self, field: np.ndarray) -> List[Tuple[int, int]]:
        """Extract optimal path from time field via gradient descent"""
        tx, ty = self.target_pos
        sx, sy = self.source_pos
        
        tx = np.clip(tx, 0, self.grid_size - 1)
        ty = np.clip(ty, 0, self.grid_size - 1)
        sx = np.clip(sx, 0, self.grid_size - 1)
        sy = np.clip(sy, 0, self.grid_size - 1)
        
        path = []
        current = (tx, ty)
        visited = set()
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for _ in range(self.grid_size * self.grid_size):
            path.append(current)
            if current == (sx, sy):
                break
            
            visited.add(current)
            cx, cy = current
            best_neighbor = current
            best_value = field[cy, cx]
            
            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if self.obstacle_field[ny, nx] > 0.5:
                        continue
                    candidate = field[ny, nx]
                    if candidate < best_value - 1e-6:
                        best_value = candidate
                        best_neighbor = (nx, ny)
            
            if best_neighbor == current or best_neighbor in visited:
                break
            
            current = best_neighbor
        
        return path if path and path[-1] == (sx, sy) else []
    
    def _compute_path_metrics(self) -> None:
        """Compute path length and cost"""
        if len(self.path_points) < 2:
            self.path_length = None
            self.path_cost = None
            return
        
        total_cost = 0.0
        total_length = 0.0
        
        for (x0, y0), (x1, y1) in zip(self.path_points[:-1], self.path_points[1:]):
            s0 = self.speed_field[y0, x0]
            s1 = self.speed_field[y1, x1]
            step_speed = max(0.5 * (s0 + s1), 1e-4)
            total_cost += 1.0 / step_speed
            total_length += math.hypot(x1 - x0, y1 - y0)
        
        self.path_length = total_length
        self.path_cost = total_cost
