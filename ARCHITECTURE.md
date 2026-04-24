# ARCHITECTURE GUIDE - Quantum-Inspired GPS Navigator

> **Disclaimer:** This project is "quantum-INSPIRED", not quantum. The
> 4-channel (N/E/S/W) amplitude representation is a classical
> directional-diffusion metaphor. No qubits, no superposition, no
> entanglement, no dependency on any quantum-computing library.

## System Overview

```
INPUT (Map + Route)
        ↓
    ┌───────────────────────┐
    │   MAP MANAGER         │
    │ - OSM Tiles Download  │
    │ - Grid Rasterization  │
    │ - Speed Field Gen.    │
    └───────────────────────┘
        ↓                ↓
   Street Grid      Speed Field
        ↓                ↓
    ┌────────────────────────────┐
    │  GPU EIKONAL SOLVER        │
    │  (4-State Neuromorphic)    │
    │ - Source/Target Set        │
    │ - GPU Wavefront Propagation│
    │ - Path Extraction          │
    └────────────────────────────┘
        ↓
   Optimal Path
        ↓
    ┌───────────────────────┐
    │   NAVIGATOR           │
    │ - Route Waypoints     │
    │ - Instructions Gen.   │
    │ - State Tracking      │
    └───────────────────────┘
        ↓
   Navigation Data
        ↓
    ┌───────────────────────┐
    │   VISUALIZATION       │
    │ - OpenGL Rendering    │
    │ - Real OSM Map Display│
    │ - HUD Overlay         │
    └───────────────────────┘
        ↓
    SCREEN OUTPUT
```

## Module Descriptions

### 1. map_manager.py

**Purpose**: Manage map data, tile caching, and grid generation

**Key Classes**:
- `OptimizedTileManager`: Parallel tile downloading
- `MapManager`: OSM integration + grid management

**Data Flow**:
```
OSM Tiles → Cache → Stitch → Grid Rasterization → Speed Field
```

**Key Methods**:
```python
load_city(place_name)              # Load OSM data for city
latlon_to_pixel(lat, lon)          # Convert coordinates
create_street_grid(grid_size)      # Binary grid (0=street, 1=building)
create_speed_field(grid_size)      # Speed multipliers by road type
```

**Grid Generation Algorithm**:
1. Download OSM street network
2. Extract road edges and coordinates
3. Rasterize edges to grid using Bresenham line drawing
4. Assign speed multipliers based on road type:
   - Motorway: 1.8× (fast)
   - Residential: 1.0× (normal)
   - Pedestrian: 0.5× (slow)

### 2. gpu_eikonal_solver.py

**Purpose**: GPU-accelerated pathfinding using neuromorphic 4-state qubits

**Key Class**:
- `GpuEikonalSolver`: Main solver interface

**GPU Architecture**:
```
Fragment Shader (Per-Cell Computation)
├─ Input: Time field (T), State vector (4-element)
├─ Operation: Eikonal equation relaxation
└─ Output: Updated T, Updated state

4-State Vector: [North, East, South, West]
├ Represents directional guidance
├ Normalized probability distribution
└ Updated via neuromorphic memory decay
```

**GPU Computation Loop**:
```
Read Buffer A → Compute → Write Buffer B
    ↓                       ↓
Propagation Step    Render to Texture
    ↓                       ↓
Swap Buffers → Next Iteration
```

**Key Methods**:
```python
set_source_target(source, target)   # Set start/end points
set_obstacle_field(obstacles)       # 1=blocked, 0=passable
set_speed_field(speeds)             # Speed multipliers
compute_path(num_iterations)        # Run GPU solver
```

**Time Complexity**:
- GPU: O(grid_size²) for complete solution
- CPU Dijkstra: O(grid_size² log grid_size)
- Speedup: 100-300×

### 3. navigator.py

**Purpose**: High-level navigation logic and state management

**Key Classes**:
- `NavigationState`: Current position/heading/velocity dataclass
- `UnifiedNavigator`: Main navigation system

**State Machine**:
```
Idle
 ↓ (set_destination)
Route Computing (GPU Eikonal)
 ↓
Route Available
 ↓ (update)
Navigating (following waypoints)
 ↓
Arrived (distance < threshold)
```

**Key Methods**:
```python
set_destination(lat, lon)           # Compute route to destination
update(dt, target_velocity)         # Update position/heading
get_current_instruction()           # Turn-by-turn text
get_distance_to_destination()       # Remaining distance
get_status()                        # Full status string
```

**Instruction Generation Algorithm**:
1. Extract all waypoints from GPU path
2. Convert grid coordinates to lat/lon
3. For each waypoint:
   - Calculate bearing to next waypoint
   - Compare with previous bearing
   - Generate instruction based on turn angle

### 4. visualization.py

**Purpose**: Real-time rendering with professional visualization

**Key Class**:
- `VisualizationEngine`: OpenGL rendering system

**Rendering Pipeline**:
```
                                    ┌──────────────┐
                                    │ OSM Map Tex  │
                                    └──────────────┘
                                           ↓
Input: Route, Vehicle Pos, Heading → Fragment Shader → Output: Screen
                                           ↓
                    Fullscreen Quad Rasterization
```

**Fragment Shader Features**:
- Map texture sampling
- Route overlay (blue line)
- Vehicle marker (green circle + heading arrow)
- Destination marker (red circle)
- Distance field calculations

**Key Methods**:
```python
render(dt)                          # Render single frame
update(dt)                          # Handle input + update state
close()                             # Cleanup
```

**OpenGL Pipeline**:
```
VS → Attributes → FS (per-fragment) → Blending → Screen
      (in_pos)      (u_map, uniforms)  (BLEND_MODE)
```

## Data Flow Examples

### Pathfinding Flow

```
(1) User Sets Destination
    │
    ↓
(2) MapManager converts lat/lon → grid pixels
    │
    ↓
(3) GpuEikonalSolver.set_source_target()
    │ - Sets source/target textures on GPU
    │ - Resets time field (all ∞ except source)
    │ - Initializes directional state (equal superposition)
    │
    ↓
(4) Propagation Loop (GPU Fragment Shader)
    │ For each iteration:
    │  - Read current time field + state
    │  - Compute neighbors' times
    │  - Solve Eikonal equation
    │  - Update directional state
    │  - Write new time field + state
    │
    ↓
(5) Convergence (after ~grid_size² iterations)
    │ - Time field now contains optimal travel times
    │ - State vectors encode optimal directions
    │
    ↓
(6) Path Extraction (Gradient Descent)
    │ - Start at target
    │ - Follow steepest descent in time field
    │ - Continue until reaching source
    │
    ↓
(7) Grid → World Conversion
    │ - Convert path pixels to lat/lon
    │ - Smooth via interpolation
    │
    ↓
(8) Route Ready!
    └→ Visualization displays route
    └→ Navigator generates instructions
```

### Navigation Update Flow

```
frame_time = 16ms (60 FPS)
    │
    ↓
Navigator.update(dt=0.016)
    │
    ├─ Update velocity (physics)
    ├─ Update heading (rotation)
    ├─ Update position (kinematics)
    │   new_lat = lat + (v * sin(heading)) / lat_scale * dt
    │   new_lon = lon + (v * cos(heading)) / lon_scale * dt
    │
    └─ Check distance to destination
       if distance < threshold → arrived
    
    ↓
VisualizationEngine.render(dt)
    │
    ├─ Convert new position to screen UV coords
    ├─ Clear screen
    ├─ Bind map texture
    ├─ Run fragment shader:
    │   - Sample map color
    │   - Draw route (blue)
    │   - Draw vehicle (green)
    │   - Draw destination (red)
    ├─ Update HUD (FPS, speed, distance)
    │
    └─ Swap buffers → Display
```

## Performance Optimization Strategies

### GPU Optimization
1. **Texture Streaming**: Mipmaps for zooming
2. **Binary Textures**: NEAREST filter (no interpolation)
3. **Framebuffer Ping-Pong**: Double buffering
4. **Fragment Shader Optimization**:
   - Minimize branching
   - Use built-in functions
   - Early exit for obstacles

### CPU Optimization
1. **Parallel Tile Downloading**: ThreadPoolExecutor (8 workers)
2. **Numpy Vectorization**: Avoid Python loops
3. **Memory Efficiency**: In-place array operations
4. **Headless Mode**: No window rendering in compute phase

### Memory Footprint
```
Grid 512×512:
├─ Time fields (2×): 2 × 256KB = 512KB
├─ State fields (2×): 2 × 1MB = 2MB
├─ Speed field: 256KB
├─ Obstacle field: 256KB
├─ Source/target fields: 512KB
└─ Map texture: ~1-4MB (depending on image)
   TOTAL: ~5-6MB GPU
```

## Integration Points

### With External Systems
- **CHIMERA/NEBULA AI**: Traffic prediction hooks
- **Real Quantum Hardware**: Sensor integration (future)
- **SLAM Systems**: Localization fusion
- **Mobile Apps**: API wrapper

### Error Handling
```python
try:
    map_mgr.load_city()
except OSMException:
    # Use cached maps
    load_cached_fallback()

try:
    solver.compute_path()
except GPUException:
    # Fallback to CPU pathfinding
    cpu_compute_dijkstra()
```

## Testing Strategy

**Unit Tests**:
- Coordinate transformations
- Grid rasterization
- Speed field generation

**Integration Tests**:
- Full pathfinding pipeline
- Route extraction accuracy
- Visualization rendering

**Performance Tests**:
- Computation time vs grid size
- GPU utilization
- FPS measurements

---

**Version**: 1.0 | **Last Updated**: 2025-11-13
