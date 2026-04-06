#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Professional OpenGL Visualization Interface
===========================================

Real-time rendering of:
- OSM map tiles
- Vehicle position and heading
- Route visualization
- Navigation HUD overlay
"""

import numpy as np
import moderngl
import glfw
from PIL import Image
from typing import Optional, Tuple


class VisualizationEngine:
    """GPU-based visualization with real maps"""
    
    def __init__(self, map_manager, navigator, width: int = 1920, height: int = 1080):
        """Initialize visualization"""
        self.map_manager = map_manager
        self.navigator = navigator
        self.width = width
        self.height = height
        
        # State
        self.running = True
        self.fps = 0.0
        self.frame_count = 0
        
        # GPU resources
        self.window = None
        self.ctx = None
        self.map_texture = None
        self.quad_vao = None
        self.quad_program = None
        self.hud_program = None
        
        self._init_glfw()
        self._init_gpu()
        self._create_shaders()
        self._upload_map_texture()
    
    def _init_glfw(self) -> None:
        """Initialize GLFW window"""
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, 4)
        
        self.window = glfw.create_window(
            self.width, self.height,
            "Quantum GPS Navigator - Professional",
            None, None
        )
        
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Window creation failed")
        
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_mouse_button_callback(self.window, self._on_mouse)
    
    def _init_gpu(self) -> None:
        """Initialize GPU context"""
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
    
    def _create_shaders(self) -> None:
        """Create rendering shaders"""
        # Quad vertex shader
        vs = """
        #version 430 core
        in vec2 in_pos;
        out vec2 v_uv;
        void main() {
            v_uv = in_pos * 0.5 + 0.5;
            gl_Position = vec4(in_pos, 0.0, 1.0);
        }
        """
        
        # Map fragment shader
        map_fs = """
        #version 430 core
        in vec2 v_uv;
        out vec4 fragColor;
        
        uniform sampler2D u_map;
        uniform vec2 u_vehicle_pos;
        uniform float u_vehicle_heading;
        uniform vec2 u_dest_pos;
        uniform int u_has_dest;
        
        // Route points array
        uniform vec2 u_route[1000];
        uniform int u_route_len;
        
        void main() {
            vec3 color = texture(u_map, v_uv).rgb;
            
            // Draw route
            if (u_route_len > 1) {
                float min_dist = 1.0;
                for (int i = 0; i < u_route_len - 1; i++) {
                    vec2 p1 = u_route[i];
                    vec2 p2 = u_route[i + 1];
                    
                    vec2 pa = v_uv - p1;
                    vec2 ba = p2 - p1;
                    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
                    float dist = length(pa - ba * h);
                    
                    if (dist < 0.004) {
                        color = mix(color, vec3(0.2, 0.5, 1.0), 0.8);
                    }
                }
            }
            
            // Vehicle marker (green circle with heading arrow)
            float dist_vehicle = length(v_uv - u_vehicle_pos);
            if (dist_vehicle < 0.012) {
                vec2 to_vehicle = normalize(v_uv - u_vehicle_pos);
                float heading_rad = radians(u_vehicle_heading);
                float angle = atan(to_vehicle.y, to_vehicle.x);
                float angle_diff = abs(mod(angle - heading_rad + 3.14159, 6.28318) - 3.14159);
                
                if (dist_vehicle < 0.006 || (dist_vehicle < 0.01 && angle_diff < 0.4)) {
                    color = vec3(0.1, 0.95, 0.2);
                }
            }
            
            // Destination marker (red circle)
            if (u_has_dest > 0) {
                float dist_dest = length(v_uv - u_dest_pos);
                if (dist_dest < 0.01) {
                    color = vec3(1.0, 0.2, 0.2);
                } else if (dist_dest < 0.012) {
                    color = mix(color, vec3(1.0, 0.2, 0.2), 0.5);
                }
            }
            
            fragColor = vec4(color, 1.0);
        }
        """
        
        self.quad_program = self.ctx.program(vertex_shader=vs, fragment_shader=map_fs)
        
        # Create quad VAO
        quad_verts = np.array([
            -1, -1,
             1, -1,
             1,  1,
            -1, -1,
             1,  1,
            -1,  1,
        ], dtype='f4')
        quad_vbo = self.ctx.buffer(quad_verts.tobytes())
        self.quad_vao = self.ctx.simple_vertex_array(self.quad_program, quad_vbo, 'in_pos')
    
    def _upload_map_texture(self) -> None:
        """Upload map image to GPU"""
        if self.map_manager.map_image is None:
            print("[Visualization] WARNING: No map image")
            return
        
        img = self.map_manager.map_image.convert('RGB')
        img_array = np.array(img, dtype='u1')
        img_array = np.flipud(img_array)  # OpenGL expects bottom-up
        
        self.map_texture = self.ctx.texture(
            (img.width, img.height), 3, img_array.tobytes()
        )
        self.map_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.map_texture.build_mipmaps()
        
        print(f"[Visualization] Map texture loaded: {img.width}x{img.height}")
    
    def _latlon_to_uv(self, lat: float, lon: float) -> Tuple[float, float]:
        """Convert lat/lon to UV coordinates"""
        if self.map_manager.bounds is None or self.map_manager.map_image is None:
            return 0.5, 0.5
        
        b = self.map_manager.bounds
        
        # Mercator projection
        lat_rad = np.radians(lat)
        x = (lon + 180.0) / 360.0
        y = (1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / np.pi) / 2.0
        
        # Normalize to bounds
        lat_rad_n = np.radians(b['north'])
        lat_rad_s = np.radians(b['south'])
        x_n = (b['west'] + 180) / 360
        x_s = (b['east'] + 180) / 360
        y_n = (1.0 - np.log(np.tan(lat_rad_n) + 1.0 / np.cos(lat_rad_n)) / np.pi) / 2.0
        y_s = (1.0 - np.log(np.tan(lat_rad_s) + 1.0 / np.cos(lat_rad_s)) / np.pi) / 2.0
        
        u = (x - x_n) / (x_s - x_n)
        v = (y - y_n) / (y_s - y_n)
        
        return np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)
    
    def render(self, dt: float) -> None:
        """Render frame"""
        self.ctx.clear(0.1, 0.1, 0.1)
        
        # Convert vehicle position to UV
        vx, vy = self._latlon_to_uv(
            self.navigator.current_position[0],
            self.navigator.current_position[1]
        )
        
        # Convert route to UV coordinates
        route_uvs = []
        for lat, lon in self.navigator.route_waypoints[:1000]:
            u, v = self._latlon_to_uv(lat, lon)
            route_uvs.extend([u, v])
        
        # Destination
        has_dest = 1 if self.navigator.destination else 0
        if self.navigator.destination:
            dx, dy = self._latlon_to_uv(
                self.navigator.destination[0],
                self.navigator.destination[1]
            )
        else:
            dx, dy = 0.5, 0.5
        
        # Set uniforms
        self.quad_program['u_vehicle_pos'].value = (vx, vy)
        self.quad_program['u_vehicle_heading'].value = self.navigator.heading
        self.quad_program['u_dest_pos'].value = (dx, dy)
        self.quad_program['u_has_dest'].value = has_dest
        self.quad_program['u_route_len'].value = len(route_uvs) // 2
        
        # Write route array
        if route_uvs:
            route_array = np.array(route_uvs, dtype='f4')
            # Pad to 2000 floats (1000 vec2)
            if len(route_array) < 2000:
                route_array = np.pad(route_array, (0, 2000 - len(route_array)))
            self.quad_program['u_route'].write(route_array.tobytes())
        
        # Bind and render map
        if self.map_texture:
            self.map_texture.use(0)
            self.quad_program['u_map'].value = 0
            self.quad_vao.render(moderngl.TRIANGLE_STRIP)
        
        # Draw HUD overlay
        self._draw_hud(dt)
        
        glfw.swap_buffers(self.window)
    
    def _draw_hud(self, dt: float) -> None:
        """Draw HUD overlay with information"""
        # Update FPS
        self.frame_count += 1
        if self.frame_count >= 30:
            self.fps = 30.0 / dt if dt > 0 else 0.0
            self.frame_count = 0
        
        # Update window title
        title = f"Quantum GPS Navigator | "
        title += f"FPS: {self.fps:.0f} | "
        title += f"Speed: {self.navigator.velocity * 3.6:.0f} km/h | "
        title += f"Heading: {self.navigator.heading:.0f}°"
        
        if self.navigator.destination:
            dist = self.navigator.get_distance_to_destination()
            title += f" | Distance: {dist:.0f}m"
        
        title += " | OPTICAL/QUANTUM ARCHITECTURE"
        glfw.set_window_title(self.window, title)
    
    def _on_key(self, window, key, scancode, action, mods) -> None:
        """Handle keyboard input"""
        if action == glfw.RELEASE:
            if key in [glfw.KEY_UP, glfw.KEY_DOWN]:
                self.navigator.velocity = 0.0
            return
        
        if action in [glfw.PRESS, glfw.REPEAT]:
            if key == glfw.KEY_UP:
                self.navigator.velocity = 10.0  # m/s
            elif key == glfw.KEY_DOWN:
                self.navigator.velocity = 0.0
            elif key == glfw.KEY_LEFT:
                self.navigator.heading += 3.0
            elif key == glfw.KEY_RIGHT:
                self.navigator.heading -= 3.0
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
    
    def _on_mouse(self, window, button, action, mods) -> None:
        """Handle mouse input"""
        if action != glfw.PRESS:
            return
        
        if button == glfw.MOUSE_BUTTON_LEFT:
            # Get mouse position
            xpos, ypos = glfw.get_cursor_pos(window)
            
            # Convert to UV
            u = xpos / self.width
            v = 1.0 - (ypos / self.height)
            
            # Convert to lat/lon
            if self.map_manager.bounds and self.map_manager.map_image:
                b = self.map_manager.bounds
                
                lon_w = b['west']
                lon_e = b['east']
                lon = lon_w + u * (lon_e - lon_w)
                
                lat_rad_n = np.radians(b['north'])
                lat_rad_s = np.radians(b['south'])
                
                y_n = (1.0 - np.log(np.tan(lat_rad_n) + 1.0 / np.cos(lat_rad_n)) / np.pi) / 2.0
                y_s = (1.0 - np.log(np.tan(lat_rad_s) + 1.0 / np.cos(lat_rad_s)) / np.pi) / 2.0
                
                y_p = y_n + v * (y_s - y_n)
                lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * y_p)))
                lat = np.degrees(lat_rad)
                
                print(f"\n[Visualization] Setting destination: ({lat:.4f}, {lon:.4f})")
                self.navigator.set_destination(lat, lon)
    
    def update(self, dt: float) -> bool:
        """Update and check if still running"""
        glfw.poll_events()
        
        if glfw.window_should_close(self.window):
            return False
        
        # Update navigator
        self.navigator.update(dt)
        
        return True
    
    def close(self) -> None:
        """Cleanup"""
        glfw.terminate()
