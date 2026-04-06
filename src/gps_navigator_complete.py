#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete GPS-Free Navigator with Manual Driving
================================================

Features:
- Origin and destination input
- Optimal route calculation using Eikonal solver
- Manual vehicle control (arrow keys)
- Turn-by-turn navigation instructions
- Automatic rerouting when off-track
- One-way street support
- Traffic direction awareness
"""

import sys
import time
import numpy as np
import glfw
import moderngl
from typing import Optional, Tuple, List
from dataclasses import dataclass

from quantum_navigation import QuantumStreetNavigator
from map_loader import RealStreetMapLoader


@dataclass
class Vehicle:
    """Vehicle state"""
    lat: float
    lon: float
    heading: float  # degrees (0 = North, 90 = East)
    speed: float  # m/s

    def to_grid(self, map_loader, grid_size):
        """Convert to grid coordinates"""
        return map_loader.latlon_to_grid(self.lat, self.lon, grid_size)


class GPSNavigator:
    """Complete GPS Navigator with manual driving"""

    def __init__(self):
        print("\n" + "="*80)
        print("GPS-FREE STREET NAVIGATOR".center(80))
        print("="*80)

        # Configuration
        self.origin = None
        self.destination = None
        self.map_loader = None
        self.navigator = None
        self.grid_size = 512
        self.radius_meters = 1500  # Map radius

        # Vehicle state
        self.vehicle = None
        self.target_speed = 0.0
        self.acceleration = 5.0  # m/s²
        self.max_speed = 20.0  # m/s (72 km/h)
        self.turn_rate = 60.0  # degrees per second

        # Route
        self.route_cells = []
        self.route_grid = None
        self.instructions = []
        self.current_instruction_idx = 0
        self.off_route = False
        self.last_reroute_time = 0.0
        self.reroute_cooldown = 3.0  # seconds

        # UI state
        self.mode = "config"  # "config", "loading", "navigating"
        self.origin_text = ""
        self.dest_text = ""
        self.input_focus = "origin"  # "origin" or "destination"
        self.status_message = "Enter origin and destination"

        # Graphics
        self.window = None
        self.ctx = None
        self.program = None
        self.text_program = None

        # Timing
        self.last_time = time.time()
        self.fps = 60.0

    def init_window(self):
        """Initialize GLFW window"""
        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.width = 1400
        self.height = 900

        self.window = glfw.create_window(
            self.width, self.height,
            "GPS-Free Street Navigator",
            None, None
        )

        if not self.window:
            glfw.terminate()
            raise RuntimeError("Window creation failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # Callbacks
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_char_callback(self.window, self._on_char)
        glfw.set_mouse_button_callback(self.window, self._on_mouse)

        # OpenGL context
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        print("[OK] Window initialized")

    def load_map_and_route(self):
        """Load map and calculate route"""
        self.mode = "loading"
        self.status_message = "Loading map..."

        print(f"[1/5] Loading map around destination area...")
        self.map_loader = RealStreetMapLoader()

        # Try to geocode destination to center the map there
        success = False
        try:
            if "," in self.dest_text:
                # Try as place name
                success = self.map_loader.load_place(self.dest_text, dist=self.radius_meters)
        except:
            pass

        if not success:
            # Default to Madrid center
            print("   Using default location (Madrid center)")
            success = self.map_loader.load_coordinates(
                center_lat=40.4168,
                center_lon=-3.7038,
                radius_meters=self.radius_meters,
                network_type='drive'
            )

        if not success:
            self.status_message = "ERROR: Failed to load map!"
            return False

        print(f"   [OK] Map loaded: {self.map_loader.bounds.width_meters:.0f}m x {self.map_loader.bounds.height_meters:.0f}m")

        # Initialize navigator
        print(f"[2/5] Initializing navigator...")
        center_lat, center_lon = self.map_loader.bounds.center

        self.navigator = QuantumStreetNavigator(
            map_loader=self.map_loader,
            grid_size=self.grid_size,
            initial_lat=center_lat,
            initial_lon=center_lon
        )

        print(f"   [OK] Navigator ready")

        # Parse origin and destination
        print(f"[3/5] Geocoding addresses...")
        origin_coords = self._parse_location(self.origin_text)
        dest_coords = self._parse_location(self.dest_text)

        if origin_coords is None:
            # Find a valid street location for origin
            origin_coords = self._find_valid_street_location(center_lat, center_lon, offset_distance=0.001)
            print(f"   Origin: Auto-selected street location ({origin_coords[0]:.6f}, {origin_coords[1]:.6f})")
        else:
            # Make sure origin is on a street
            origin_coords = self._find_valid_street_location(origin_coords[0], origin_coords[1])
            print(f"   Origin: ({origin_coords[0]:.6f}, {origin_coords[1]:.6f})")

        if dest_coords is None:
            # Find a valid street location for destination (far from origin)
            dest_coords = self._find_valid_street_location(center_lat, center_lon, offset_distance=0.004)
            print(f"   Destination: Auto-selected street location ({dest_coords[0]:.6f}, {dest_coords[1]:.6f})")
        else:
            # Make sure destination is on a street
            dest_coords = self._find_valid_street_location(dest_coords[0], dest_coords[1])
            print(f"   Destination: ({dest_coords[0]:.6f}, {dest_coords[1]:.6f})")

        # Initialize vehicle at origin
        self.vehicle = Vehicle(
            lat=origin_coords[0],
            lon=origin_coords[1],
            heading=0.0,
            speed=0.0
        )

        # Calculate route
        print(f"[4/5] Calculating optimal route...")
        success = self.navigator.set_destination(dest_coords[0], dest_coords[1])

        if success:
            self._update_route()
            print(f"   [OK] Route calculated!")
            print(f"   Distance: {self._calculate_route_distance():.0f}m")
            print(f"   Instructions: {len(self.instructions)}")
            self.status_message = "Route ready! Use arrow keys to drive"
            self.mode = "navigating"
        else:
            self.status_message = "ERROR: Could not find route!"
            return False

        # Don't create graphics here - will be done after window is created
        self.mode = "ready"
        return True

    def _parse_location(self, text: str) -> Optional[Tuple[float, float]]:
        """Parse location text into lat/lon"""
        text = text.strip()

        # Check if it's coordinates (lat, lon)
        if "," in text:
            try:
                parts = text.split(",")
                lat = float(parts[0].strip())
                lon = float(parts[1].strip())
                return (lat, lon)
            except:
                pass

        # Try to geocode using OSM (if it's a place name within loaded map)
        # For simplicity, return None and use defaults
        return None

    def _find_valid_street_location(self, lat: float, lon: float, offset_distance: float = 0.0) -> Tuple[float, float]:
        """
        Find a valid street location near the given coordinates.

        Args:
            lat: Target latitude
            lon: Target longitude
            offset_distance: Distance to offset from center (in degrees, ~111m per 0.001)

        Returns:
            Tuple of (lat, lon) on a valid street
        """
        # Apply offset if requested
        if offset_distance > 0:
            # Random offset in a circle
            import random
            angle = random.uniform(0, 2 * np.pi)
            lat += offset_distance * np.cos(angle)
            lon += offset_distance * np.sin(angle)

        # Convert to grid coordinates
        gx, gy = self.map_loader.latlon_to_grid(lat, lon, self.grid_size)

        # Search in expanding circles for a street cell
        max_search_radius = 50  # cells

        for radius in range(1, max_search_radius):
            for angle in np.linspace(0, 2*np.pi, 8*radius):
                test_x = int(gx + radius * np.cos(angle))
                test_y = int(gy + radius * np.sin(angle))

                # Check if in bounds
                if 0 <= test_x < self.grid_size and 0 <= test_y < self.grid_size:
                    # Check if it's a street (value < 0.5)
                    if self.navigator.street_grid[test_y, test_x] < 0.5:
                        # Found a street! Convert back to lat/lon
                        found_lat, found_lon = self.map_loader.grid_to_latlon(
                            test_x, test_y, self.grid_size
                        )
                        return (found_lat, found_lon)

        # If no street found, return original (fallback)
        print(f"   WARNING: Could not find street near ({lat:.6f}, {lon:.6f}), using original")
        return (lat, lon)

    def _update_route(self):
        """Update route visualization from navigator"""
        # Get waypoints from navigator
        if hasattr(self.navigator, 'route_waypoints') and self.navigator.route_waypoints:
            # Convert waypoints (lat/lon) to grid cells
            self.route_cells = []

            for waypoint in self.navigator.route_waypoints:
                lat, lon = waypoint
                gx, gy = self.map_loader.latlon_to_grid(lat, lon, self.grid_size)
                self.route_cells.append((gx, gy))

            # Create route grid
            if self.route_grid is None:
                self.route_grid = np.zeros((self.grid_size, self.grid_size), dtype='f4')
            else:
                self.route_grid.fill(0.0)

            # Draw route with line interpolation between waypoints
            for i in range(len(self.route_cells) - 1):
                x1, y1 = self.route_cells[i]
                x2, y2 = self.route_cells[i + 1]

                # Bresenham line algorithm to fill all cells between waypoints
                points = self._bresenham_line(int(x1), int(y1), int(x2), int(y2))

                for x, y in points:
                    if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                        self.route_grid[y, x] = 1.0

            # Update texture if it exists
            if hasattr(self, 'route_tex') and self.route_tex is not None:
                self.route_tex.write(self.route_grid.tobytes())

        # Get instructions
        status = self.navigator.get_status()
        self.instructions = status.get('instructions', [])
        self.current_instruction_idx = 0

    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """
        Bresenham's line algorithm to get all cells between two points.
        Returns list of (x, y) coordinates.
        """
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0

        while True:
            points.append((x, y))

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return points

    def _calculate_route_distance(self) -> float:
        """Calculate route distance in meters"""
        if not self.route_cells or len(self.route_cells) < 2:
            return 0.0

        from geopy.distance import geodesic

        total_dist = 0.0
        for i in range(len(self.route_cells) - 1):
            x1, y1 = self.route_cells[i]
            x2, y2 = self.route_cells[i + 1]

            lat1, lon1 = self.map_loader.grid_to_latlon(x1, y1, self.grid_size)
            lat2, lon2 = self.map_loader.grid_to_latlon(x2, y2, self.grid_size)

            dist = geodesic((lat1, lon1), (lat2, lon2)).meters
            total_dist += dist

        return total_dist

    def _create_graphics(self):
        """Create OpenGL resources"""
        # Create textures
        size = (self.grid_size, self.grid_size)

        self.map_tex = self.ctx.texture(size, 1, dtype='f4')
        self.map_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.map_tex.write(self.navigator.street_grid.tobytes())

        self.route_tex = self.ctx.texture(size, 1, dtype='f4')
        self.route_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        if self.route_grid is not None:
            self.route_tex.write(self.route_grid.tobytes())

        # Create shader
        vs = """
        #version 430
        in vec2 in_pos;
        out vec2 v_uv;
        void main() {
            v_uv = in_pos * 0.5 + 0.5;
            gl_Position = vec4(in_pos, 0.0, 1.0);
        }
        """

        fs = """
        #version 430
        in vec2 v_uv;
        out vec4 fragColor;

        uniform sampler2D u_map;
        uniform sampler2D u_route;
        uniform vec2 u_vehicle_pos;
        uniform float u_vehicle_heading;
        uniform vec2 u_dest_pos;

        void main() {
            float map_val = texture(u_map, v_uv).r;
            float route_val = texture(u_route, v_uv).r;

            // Base color
            vec3 color;
            if (map_val < 0.5) {
                color = vec3(0.92, 0.92, 0.88);  // Streets (beige)
            } else {
                color = vec3(0.35, 0.38, 0.42);  // Buildings (dark gray)
            }

            // Route overlay (blue)
            if (route_val > 0.5) {
                color = mix(color, vec3(0.3, 0.5, 1.0), 0.5);
            }

            vec2 px_size = 1.0 / vec2(textureSize(u_map, 0));
            vec2 uv_px = v_uv * vec2(textureSize(u_map, 0));

            // Vehicle (green arrow)
            vec2 vehicle_px = u_vehicle_pos * vec2(textureSize(u_map, 0));
            float dist_vehicle = length(uv_px - vehicle_px);
            if (dist_vehicle < 12.0) {
                // Draw arrow pointing in heading direction
                vec2 to_vehicle = uv_px - vehicle_px;
                float angle = atan(to_vehicle.y, to_vehicle.x);
                float heading_rad = radians(u_vehicle_heading - 90.0); // Adjust for coordinate system
                float angle_diff = abs(mod(angle - heading_rad + 3.14159, 6.28318) - 3.14159);

                if (dist_vehicle < 8.0 || (dist_vehicle < 12.0 && angle_diff < 0.5)) {
                    color = mix(color, vec3(0.2, 1.0, 0.3), smoothstep(12.0, 6.0, dist_vehicle));
                }
            }

            // Destination (red flag)
            vec2 dest_px = u_dest_pos * vec2(textureSize(u_map, 0));
            float dist_dest = length(uv_px - dest_px);
            if (dist_dest < 10.0) {
                color = mix(color, vec3(1.0, 0.2, 0.2), smoothstep(10.0, 4.0, dist_dest));
            }

            fragColor = vec4(color, 1.0);
        }
        """

        self.program = self.ctx.program(vertex_shader=vs, fragment_shader=fs)

        # Create fullscreen quad
        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype='f4')

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.program, self.vbo, 'in_pos')

    def update(self, dt: float):
        """Update simulation"""
        if self.mode != "navigating" or self.vehicle is None:
            return

        # Update speed towards target
        speed_diff = self.target_speed - self.vehicle.speed
        if abs(speed_diff) > 0.01:
            accel = self.acceleration * np.sign(speed_diff)
            self.vehicle.speed += accel * dt
            self.vehicle.speed = np.clip(self.vehicle.speed, 0.0, self.max_speed)

        # Update position based on heading and speed
        if self.vehicle.speed > 0.1:
            # Convert heading to radians (0° = North = +Y)
            heading_rad = np.radians(self.vehicle.heading)

            # Calculate velocity components
            dx = np.sin(heading_rad) * self.vehicle.speed * dt
            dy = np.cos(heading_rad) * self.vehicle.speed * dt

            # Convert to lat/lon delta (approximate)
            meters_per_degree_lat = 111320.0
            meters_per_degree_lon = 111320.0 * np.cos(np.radians(self.vehicle.lat))

            dlat = dy / meters_per_degree_lat
            dlon = dx / meters_per_degree_lon

            # Calculate new position
            new_lat = self.vehicle.lat + dlat
            new_lon = self.vehicle.lon + dlon

            # Check collision with buildings
            if self._check_collision(new_lat, new_lon):
                # Stop the vehicle if collision detected
                self.vehicle.speed = 0.0
                self.target_speed = 0.0
                self.status_message = "COLLISION! Can't drive through buildings"
            else:
                # Update position
                self.vehicle.lat = new_lat
                self.vehicle.lon = new_lon

                # Check if off route
                self._check_off_route()

    def _check_collision(self, lat: float, lon: float) -> bool:
        """Check if position collides with building"""
        # Convert to grid coordinates
        gx, gy = self.map_loader.latlon_to_grid(lat, lon, self.grid_size)

        # Check bounds
        if gx < 0 or gx >= self.grid_size or gy < 0 or gy >= self.grid_size:
            return True

        # Check if on street (street_grid: 0 = street, 1 = building)
        grid_val = self.navigator.street_grid[int(gy), int(gx)]

        # Allow driving on streets (value < 0.5)
        return grid_val >= 0.5

    def _check_off_route(self):
        """Check if vehicle is off the planned route"""
        if not self.route_cells:
            return

        vx, vy = self.vehicle.to_grid(self.map_loader, self.grid_size)

        # Find distance to nearest route cell
        min_dist = float('inf')
        for rx, ry in self.route_cells:
            dist = np.sqrt((vx - rx)**2 + (vy - ry)**2)
            if dist < min_dist:
                min_dist = dist

        # If too far from route (>20 cells), reroute
        if min_dist > 20.0:
            if not self.off_route:
                self.off_route = True
                self.status_message = "OFF ROUTE - Recalculating..."

                # Reroute if cooldown expired
                current_time = time.time()
                if current_time - self.last_reroute_time > self.reroute_cooldown:
                    self._reroute()
                    self.last_reroute_time = current_time
        else:
            if self.off_route:
                self.off_route = False
                self.status_message = "Back on route"

    def _reroute(self):
        """Recalculate route from current position"""
        print(f"\n[REROUTE] Calculating new route from current position...")

        # Update navigator position
        self.navigator.current_lat = self.vehicle.lat
        self.navigator.current_lon = self.vehicle.lon

        # Recalculate to destination
        if self.navigator.destination_lat and self.navigator.destination_lon:
            dest_lat = self.navigator.destination_lat
            dest_lon = self.navigator.destination_lon
            success = self.navigator.set_destination(dest_lat, dest_lon)

            if success:
                self._update_route()
                print(f"   [OK] New route calculated: {len(self.route_cells)} cells")
                self.status_message = "Route updated!"
            else:
                print(f"   [ERROR] Could not find new route")
                self.status_message = "Cannot find route!"

    def render(self):
        """Render frame"""
        self.ctx.clear(0.15, 0.15, 0.18)

        if self.mode == "config":
            self._render_config_screen()
        elif self.mode == "loading":
            self._render_loading_screen()
        elif self.mode == "navigating":
            self._render_navigation()

    def _render_config_screen(self):
        """Render configuration screen"""
        # For now, just clear with dark background
        # In a full implementation, you'd render text inputs
        pass

    def _render_loading_screen(self):
        """Render loading screen"""
        pass

    def _render_navigation(self):
        """Render navigation view"""
        if self.vehicle is None or self.navigator is None:
            return

        # Get positions in normalized grid coordinates
        vx, vy = self.vehicle.to_grid(self.map_loader, self.grid_size)
        vehicle_norm = (vx / self.grid_size, vy / self.grid_size)

        if self.navigator.destination_lat and self.navigator.destination_lon:
            dest_lat = self.navigator.destination_lat
            dest_lon = self.navigator.destination_lon
            dx, dy = self.map_loader.latlon_to_grid(dest_lat, dest_lon, self.grid_size)
            dest_norm = (dx / self.grid_size, dy / self.grid_size)
        else:
            dest_norm = (0.5, 0.5)

        # Set uniforms
        self.program['u_vehicle_pos'].value = vehicle_norm
        self.program['u_vehicle_heading'].value = self.vehicle.heading
        self.program['u_dest_pos'].value = dest_norm

        # Bind textures
        self.map_tex.use(0)
        self.route_tex.use(1)
        self.program['u_map'].value = 0
        self.program['u_route'].value = 1

        # Render
        self.vao.render(moderngl.TRIANGLE_STRIP)

    def _on_key(self, window, key, scancode, action, mods):
        """Handle keyboard input"""
        if action == glfw.RELEASE:
            # Stop acceleration when key released
            if key in [glfw.KEY_UP, glfw.KEY_DOWN]:
                self.target_speed = 0.0
            return

        if self.mode == "config":
            if key == glfw.KEY_ENTER and action == glfw.PRESS:
                if self.input_focus == "origin" and self.origin_text:
                    self.input_focus = "destination"
                elif self.input_focus == "destination" and self.dest_text:
                    # Start loading
                    self.load_map_and_route()
            elif key == glfw.KEY_TAB and action == glfw.PRESS:
                self.input_focus = "destination" if self.input_focus == "origin" else "origin"
            elif key == glfw.KEY_BACKSPACE:
                if self.input_focus == "origin" and self.origin_text:
                    self.origin_text = self.origin_text[:-1]
                elif self.input_focus == "destination" and self.dest_text:
                    self.dest_text = self.dest_text[:-1]
            elif key == glfw.KEY_ESCAPE and action == glfw.PRESS:
                glfw.set_window_should_close(window, True)

        elif self.mode == "navigating":
            if key == glfw.KEY_UP:
                # Accelerate
                self.target_speed = self.max_speed
            elif key == glfw.KEY_DOWN:
                # Brake
                self.target_speed = 0.0
            elif key == glfw.KEY_LEFT:
                # Turn left (counterclockwise)
                self.vehicle.heading += self.turn_rate * 0.016  # Assume ~60fps
                self.vehicle.heading = self.vehicle.heading % 360.0
            elif key == glfw.KEY_RIGHT:
                # Turn right (clockwise)
                self.vehicle.heading -= self.turn_rate * 0.016
                self.vehicle.heading = self.vehicle.heading % 360.0
            elif key == glfw.KEY_R and action == glfw.PRESS:
                # Manual reroute
                self._reroute()
            elif key == glfw.KEY_ESCAPE and action == glfw.PRESS:
                glfw.set_window_should_close(window, True)

    def _on_char(self, window, codepoint):
        """Handle character input"""
        if self.mode == "config":
            char = chr(codepoint)
            if self.input_focus == "origin":
                self.origin_text += char
            elif self.input_focus == "destination":
                self.dest_text += char

    def _on_mouse(self, window, button, action, mods):
        """Handle mouse clicks"""
        pass

    def run(self):
        """Main loop"""
        # Get input BEFORE creating window
        print("\n" + "="*80)
        print("CONFIGURATION".center(80))
        print("="*80)
        print("\nEnter origin and destination in the console:")
        print("(You can use coordinates like '40.4168, -3.7038' or press Enter for defaults)\n")

        # Get input from console (simple version)
        try:
            origin_input = input("Origin (or Enter for map center): ").strip()
            self.origin_text = origin_input if origin_input else "40.4168, -3.7038"

            dest_input = input("Destination (or Enter for nearby point): ").strip()
            self.dest_text = dest_input if dest_input else "40.4200, -3.7000"

            print("\nLoading map and calculating route...\n")

            # Load map and route BEFORE creating window
            success = self.load_map_and_route()

            if not success:
                print("Failed to initialize navigation!")
                return

        except KeyboardInterrupt:
            print("\nCancelled")
            return

        # NOW create window after everything is loaded
        print("\n[5/5] Creating window...")
        self.init_window()
        print("   [OK] Window ready")

        # Create graphics NOW that window exists
        print("\n[6/6] Creating graphics resources...")
        self._create_graphics()
        print("   [OK] Graphics ready")

        self.mode = "navigating"

        # Main loop
        print("\n" + "="*80)
        print("NAVIGATION CONTROLS".center(80))
        print("="*80)
        print("  UP ARROW    : Accelerate")
        print("  DOWN ARROW  : Brake")
        print("  LEFT ARROW  : Turn left")
        print("  RIGHT ARROW : Turn right")
        print("  R           : Force reroute")
        print("  ESC         : Exit")
        print("="*80 + "\n")

        while not glfw.window_should_close(self.window):
            current = time.time()
            dt = current - self.last_time
            self.last_time = current

            # Update title with status
            if self.mode == "navigating" and self.vehicle:
                title = (f"GPS Navigator | Speed: {self.vehicle.speed*3.6:.0f} km/h | "
                        f"Heading: {self.vehicle.heading:.0f}° | {self.status_message}")
                glfw.set_window_title(self.window, title)

            self.update(dt)
            self.render()

            glfw.swap_buffers(self.window)
            glfw.poll_events()

        glfw.terminate()


def main():
    try:
        app = GPSNavigator()
        app.run()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
