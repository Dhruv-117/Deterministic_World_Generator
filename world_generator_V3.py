"""
Deterministic Seed-Based Procedural Fantasy World Generator
============================================================
A fully reproducible world generation system using seeded random generators.
Generates a 200x200 grid with tectonics, climate, lakes, and biomes.
"""

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import Voronoi
from collections import deque
import random
import math
import builtins
import sys

# Optional: noise library for Perlin/Simplex noise
try:
    from noise import snoise2, pnoise2
    NOISE_AVAILABLE = True
except ImportError:
    NOISE_AVAILABLE = False
    print("Warning: 'noise' library not available. Using fallback noise generation.")

# Optional: matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# =============================================================================
# CONSTANTS
# =============================================================================
WORLD_WIDTH = 200
WORLD_HEIGHT = 200
TOTAL_TILES = WORLD_WIDTH * WORLD_HEIGHT  # 40,000

REGION_SIZE = 8  # 8x8 tile regions
REGIONS_X = WORLD_WIDTH // REGION_SIZE   # 25
REGIONS_Y = WORLD_HEIGHT // REGION_SIZE  # 25

# Target percentages
LAND_PERCENT = 0.50
OCEAN_PERCENT = 0.50
MOUNTAIN_PERCENT_OF_LAND = 0.10
LAKE_PERCENT_MIN = 0.02  # Min 2% of land for lakes
LAKE_PERCENT_MAX = 0.05  # Max 5% of land for lakes

# Lake constraints
LAKE_COAST_BUFFER = 8  # Lakes cannot be within 8 tiles of coastline
MIN_LAKE_SIZE = 4  # Minimum tiles per lake (no single-tile lakes)
MAX_LAKE_SIZE = 80  # Maximum tiles per lake (smaller, more numerous lakes)
MIN_LAKE_SPACING = 14  # Minimum distance between lake centers (spread them out)

# Desert lake constraint (0-2 lakes in desert: 70% 0, 20% 1, 10% 2)
DESERT_LAKE_PROBABILITIES = {0: 0.70, 1: 0.20, 2: 0.10}

# Biome constraints
MIN_BIOME_CLUSTER_SIZE = 4  # Minimum connected tiles for any biome

# Biome targets (as percentage of non-mountain land)
# Climate-based system with temperature/moisture/elevation dependencies
BIOME_TARGETS = {
    'plains': (0.25, 0.35),
    'forest': (0.18, 0.30),    # Needs high moisture, near water
    'desert': (0.05, 0.15),    # Climate-constrained (hot, dry areas)
    'rocky_hills': (0.03, 0.07),     # Low moisture hills
    'grassy_hills': (0.03, 0.07),    # Medium moisture hills  
    'forest_hills': (0.03, 0.07),    # High moisture hills
    'snow_tundra': (0.15, 0.28),  # Cold areas are common
    'mountain': (0.10, 0.10),  # Exactly 10%
}

# Moisture decay constants (distance-based)
MOISTURE_OCEAN_DECAY = 30.0  # exp(-dist/30) for ocean (reduced from 40)
MOISTURE_RIVER_DECAY = 10.0  # exp(-dist/10) for rivers - PLACEHOLDER (river system disabled)
MOISTURE_LAKE_DECAY = 18.0   # exp(-dist/18) for lakes (reduced from 25)
MOISTURE_OCEAN_WEIGHT = 1.0
MOISTURE_RIVER_WEIGHT = 0.6   # PLACEHOLDER (river system disabled) - no effect when dist_river is max
MOISTURE_LAKE_WEIGHT = 0.8    # Reduced from 0.9

# Temperature constants
TEMP_ELEVATION_FACTOR = 0.4  # Elevation cooling
TEMP_RIVER_COOLING_WEIGHT = 0.05  # River cooling effect
TEMP_RIVER_COOLING_DECAY = 10.0   # River cooling decay distance
TEMP_LAKE_COOLING_WEIGHT = 0.08
TEMP_LAKE_COOLING_DECAY = 15.0

# Terminal logging controls
_ORIGINAL_PRINT = builtins.print
_SUPPRESSED_PRINT_COUNT = 0


def configure_terminal_logging(verbose=False):
    """Configure terminal logging mode.

    - verbose=True: keep all prints
    - verbose=False: compact output (suppresses low-value lines)
    """
    global _SUPPRESSED_PRINT_COUNT
    _SUPPRESSED_PRINT_COUNT = 0

    if verbose:
        return

    important_prefixes = (
        "=",
        "GENERATING WORLD WITH SEED",
        "[",
        "VALIDATION RESULTS",
        "RIVER NETWORK SUMMARY:",
        "ENVIRONMENT RESOURCES SUMMARY:",
        "World saved to:",
        "Visualization saved to:",
        "Atlas map saved to:",
        "Resource map saved to:",
        "World generation complete!",
    )

    def compact_print(*args, **kwargs):
        global _SUPPRESSED_PRINT_COUNT

        target_file = kwargs.get('file', None)
        if target_file is not None and target_file is not sys.stdout:
            _ORIGINAL_PRINT(*args, **kwargs)
            return

        text = " ".join(str(a) for a in args).strip()
        if text.startswith(important_prefixes):
            _ORIGINAL_PRINT(*args, **kwargs)
        else:
            _SUPPRESSED_PRINT_COUNT += 1

    builtins.print = compact_print


def restore_terminal_logging(show_summary=True):
    """Restore original print and optionally show compact logging summary."""
    global _SUPPRESSED_PRINT_COUNT
    builtins.print = _ORIGINAL_PRINT
    if show_summary and _SUPPRESSED_PRINT_COUNT > 0:
        _ORIGINAL_PRINT(f"[compact-logs] Suppressed {_SUPPRESSED_PRINT_COUNT} low-priority log lines. Use --verbose-logs for full output.")

# =============================================================================
# WATERSHED HYDROLOGY CONSTANTS
# =============================================================================
# Flow accumulation thresholds for river formation
RIVER_FLOW_THRESHOLD = 8               # Min flow accumulation to form a river (lowered for more rivers)
RIVER_MAJOR_THRESHOLD = 50             # Flow threshold for major rivers
RIVER_MIN_HEADWATER_FLOW = 8           # Same as threshold - headwater is where river "starts"

# River hierarchy thresholds (flow accumulation)
RIVER_STREAM_THRESHOLD = 8             # Small streams (threshold to major)
RIVER_RIVER_THRESHOLD = 30             # Regular rivers (stream to major)
RIVER_MAJOR_RIVER_THRESHOLD = 60       # Major rivers

# Snowmelt and spring source parameters
SNOWMELT_TEMP_THRESHOLD = 0.35         # Temperature below which snowmelt boosts flow
SNOWMELT_FLOW_MULTIPLIER = 2.5         # Flow boost multiplier for snowmelt areas (increased)
SPRING_MOISTURE_THRESHOLD = 0.45       # Min moisture for spring sources (lowered)
SPRING_ELEVATION_MAX = 0.70            # Max elevation for springs (raised)
SPRING_FLOW_BONUS = 20                 # Flow bonus for spring sources (increased)

# Headwater classification elevation thresholds
HEADWATER_MOUNTAIN_ELEV = 0.65         # Mountain headwater elevation threshold (lowered)
HEADWATER_HIGHLAND_ELEV = 0.45         # Highland/hill headwater threshold (lowered)
HEADWATER_SPRING_ELEV_MAX = 0.60       # Spring sources below this elevation

# Upstream extension parameters
UPSTREAM_MAX_LENGTH = 200              # Max tiles to trace upstream (increased for longer rivers)
UPSTREAM_MIN_SLOPE = 0.0003            # Min slope to continue upstream (very permissive)
UPSTREAM_STOP_AT_RIDGE = True          # Stop when reaching ridge lines

# System consolidation parameters
MIN_RIVER_SYSTEM_LENGTH = 6            # Min length for standalone river systems (lowered)
TRIBUTARY_MERGE_DISTANCE = 5           # Max distance to merge into larger river
TARGET_MAX_RIVERS = 35                 # Target number of main river systems (increased)
MIN_HEADWATER_SPACING = 10             # Min tiles between river headwaters (reduced)
TRIBUTARY_SPACING_FACTOR = 0.7         # Tributary spacing = MIN_HEADWATER_SPACING * this factor
MAX_TRIBUTARIES_PER_RIVER = 8          # Maximum tributaries per main river system
INLAND_RIVER_CHANCE = 0.50             # Chance to add additional inland rivers (50%)

# Basin carving parameters
BASIN_CARVE_AMOUNT = 0.008             # Base carving depth per tile
BASIN_CARVE_MAX_DEPTH = 0.20           # Maximum cumulative carving per path (increased for no-lake runs)
BASIN_MAX_ITERATIONS = 200             # Maximum basin carving iterations (increased for thorough drainage)
BASIN_FLOOD_LIMIT = 600                # Max tiles to flood-fill for basin (increased)

# River width calculation
RIVER_BASE_WIDTH = 1.0                 # Minimum river width
RIVER_WIDTH_LOG_SCALE = 0.4            # Logarithmic scaling factor
RIVER_MAX_WIDTH = 3.0                  # Maximum river width in tiles

# River climate effects - distance-based moisture boost
RIVER_MOISTURE_DIST_0_1 = 0.35         # Strong moisture boost (distance 0-1)
RIVER_MOISTURE_DIST_2_3 = 0.20         # Moderate boost (distance 2-3)
RIVER_MOISTURE_DIST_4_6 = 0.10         # Weak boost (distance 4-6)
RIVER_MOISTURE_DIST_7_8 = 0.03         # Minimal boost (distance 7-8)
RIVER_TEMP_REDUCTION = 0.04            # Temperature reduction near rivers

# Enhanced biome effects near rivers by type
MAJOR_RIVER_MOISTURE_BOOST = 0.15      # Extra moisture for major river proximity
MAJOR_RIVER_INFLUENCE_DIST = 12        # Influence distance for major rivers

# Validation thresholds
RIVER_MIN_LENGTH = 5                   # Minimum tiles for a valid river segment (increased)


# =============================================================================
# SEEDED NOISE GENERATION
# =============================================================================
class SeededNoiseGenerator:
    """Generates deterministic noise using seeded random or noise library."""
    
    def __init__(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # Generate random offsets for noise functions
        self.offset_x = self.rng.uniform(0, 10000)
        self.offset_y = self.rng.uniform(0, 10000)
    
    def perlin_like(self, x, y, scale=1.0, octaves=4, persistence=0.5):
        """Generate Perlin-like noise at coordinates."""
        if NOISE_AVAILABLE:
            return pnoise2(
                (x + self.offset_x) / scale,
                (y + self.offset_y) / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=2.0,
                repeatx=WORLD_WIDTH,
                repeaty=WORLD_HEIGHT,
                base=self.seed % 1024
            )
        else:
            return self._fallback_noise(x, y, scale, octaves, persistence)
    
    def simplex_like(self, x, y, scale=1.0, octaves=4, persistence=0.5):
        """Generate Simplex-like noise at coordinates."""
        if NOISE_AVAILABLE:
            return snoise2(
                (x + self.offset_x) / scale,
                (y + self.offset_y) / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=2.0,
                base=self.seed % 1024
            )
        else:
            return self._fallback_noise(x, y, scale, octaves, persistence)
    
    def _fallback_noise(self, x, y, scale, octaves, persistence):
        """Fallback noise generation using seeded random interpolation."""
        value = 0.0
        amplitude = 1.0
        max_value = 0.0
        frequency = 1.0 / scale
        
        for _ in range(octaves):
            nx = (x + self.offset_x) * frequency
            ny = (y + self.offset_y) * frequency
            
            hash_val = self._hash_2d(int(nx), int(ny))
            value += hash_val * amplitude
            
            max_value += amplitude
            amplitude *= persistence
            frequency *= 2.0
        
        return value / max_value if max_value > 0 else 0.0
    
    def _hash_2d(self, x, y):
        """Deterministic 2D hash function."""
        h = (x * 374761393 + y * 668265263 + self.seed) & 0xFFFFFFFF
        h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
        return ((h & 0xFFFFFF) / 0xFFFFFF) * 2 - 1


def generate_heightmap_noise(width, height, noise_gen, scale=50.0, octaves=6):
    """Generate a full heightmap using noise."""
    heightmap = np.zeros((height, width))
    
    for y in range(height):
        for x in range(width):
            value = noise_gen.simplex_like(x, y, scale, octaves, 0.5)
            value += 0.3 * noise_gen.perlin_like(x, y, scale * 0.5, octaves=3, persistence=0.6)
            heightmap[y, x] = value
    
    return heightmap


# =============================================================================
# TECTONIC PLATE GENERATION
# =============================================================================
def generate_tectonic_plates(width, height, seed):
    """Generate 2-4 tectonic plates using noise-distorted Voronoi tessellation."""
    rng = np.random.default_rng(seed)
    py_random = random.Random(seed)
    
    num_plates = rng.integers(2, 5)
    dominant_continent = py_random.random() < 0.70
    
    plate_centers = []
    for i in range(num_plates):
        cx = rng.uniform(width * 0.1, width * 0.9)
        cy = rng.uniform(height * 0.1, height * 0.9)
        plate_centers.append([cx, cy])
    
    plate_centers = np.array(plate_centers)
    noise_gen = SeededNoiseGenerator(seed + 500)
    
    plate_map = np.zeros((height, width), dtype=np.int32)
    
    for y in range(height):
        for x in range(width):
            min_dist = float('inf')
            closest_plate = 0
            
            noise_val = noise_gen.simplex_like(x, y, scale=30.0, octaves=4, persistence=0.5)
            noise_val2 = noise_gen.perlin_like(x, y, scale=15.0, octaves=3, persistence=0.6)
            combined_noise = noise_val * 0.6 + noise_val2 * 0.4
            
            for i, (cx, cy) in enumerate(plate_centers):
                base_dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                plate_noise = noise_gen.simplex_like(x + i * 1000, y + i * 1000, scale=40.0, octaves=3, persistence=0.5)
                distortion = (combined_noise * 20 + plate_noise * 15)
                dist = base_dist + distortion
                
                if dist < min_dist:
                    min_dist = dist
                    closest_plate = i
            plate_map[y, x] = closest_plate
    
    plate_types = []
    if dominant_continent:
        plate_sizes = [(plate_map == i).sum() for i in range(num_plates)]
        largest_plate = np.argmax(plate_sizes)
        for i in range(num_plates):
            if i == largest_plate:
                plate_types.append('continental')
            else:
                plate_types.append('oceanic' if py_random.random() < 0.6 else 'continental_small')
    else:
        num_continental = rng.integers(2, min(4, num_plates + 1))
        continental_indices = rng.choice(num_plates, size=num_continental, replace=False)
        for i in range(num_plates):
            if i in continental_indices:
                plate_types.append('continental')
            else:
                plate_types.append('oceanic')
    
    plate_vectors = []
    for i in range(num_plates):
        angle = rng.uniform(0, 2 * np.pi)
        speed = rng.uniform(0.5, 2.0)
        plate_vectors.append((np.cos(angle) * speed, np.sin(angle) * speed))
    
    return plate_map, plate_centers, plate_types, plate_vectors, num_plates


def find_plate_boundaries(plate_map, width, height):
    """Find tiles that are on plate boundaries."""
    boundaries = np.zeros((height, width), dtype=bool)
    
    for y in range(height):
        for x in range(width):
            current_plate = plate_map[y, x]
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if plate_map[ny, nx] != current_plate:
                        boundaries[y, x] = True
                        break
    
    return boundaries


def determine_boundary_type(plate_map, plate_vectors, width, height):
    """Determine if boundaries are convergent, divergent, or transform."""
    boundaries = find_plate_boundaries(plate_map, width, height)
    convergent = np.zeros((height, width), dtype=bool)
    
    for y in range(height):
        for x in range(width):
            if not boundaries[y, x]:
                continue
            
            current_plate = plate_map[y, x]
            v1 = plate_vectors[current_plate]
            
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    other_plate = plate_map[ny, nx]
                    if other_plate != current_plate:
                        v2 = plate_vectors[other_plate]
                        rel_x = v1[0] - v2[0]
                        rel_y = v1[1] - v2[1]
                        dir_x = nx - x
                        dir_y = ny - y
                        if (rel_x * dir_x + rel_y * dir_y) > 0:
                            convergent[y, x] = True
                        break
    
    return boundaries, convergent


# =============================================================================
# ELEVATION GENERATION
# =============================================================================
def generate_elevation(width, height, seed, plate_map, convergent_boundaries, plate_types, num_plates):
    """Generate elevation with tectonic influence."""
    noise_gen = SeededNoiseGenerator(seed + 1000)
    
    heightmap = generate_heightmap_noise(width, height, noise_gen, scale=60.0, octaves=6)
    heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
    
    continental_bonus = np.zeros((height, width))
    for plate_id in range(num_plates):
        mask = plate_map == plate_id
        if plate_types[plate_id] in ['continental', 'continental_small']:
            bonus = 0.3 if plate_types[plate_id] == 'continental' else 0.15
            continental_bonus[mask] = bonus
    
    boundary_uplift = np.zeros((height, width))
    convergent_expanded = ndimage.binary_dilation(convergent_boundaries, iterations=3)
    boundary_uplift[convergent_expanded] = 0.25
    boundary_uplift[convergent_boundaries] = 0.4
    boundary_uplift = ndimage.gaussian_filter(boundary_uplift, sigma=2)
    
    heightmap = heightmap + continental_bonus + boundary_uplift
    heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
    
    return heightmap


def enforce_land_ocean_ratio(heightmap, target_land_ratio=0.50):
    """Use percentile threshold to enforce exact land/ocean ratio."""
    flat = heightmap.flatten()
    threshold = np.percentile(flat, (1.0 - target_land_ratio) * 100)
    is_land = heightmap >= threshold
    return is_land, threshold


# =============================================================================
# OCEAN TOPOLOGY VALIDATION
# =============================================================================
def validate_ocean_connectivity(is_land, width, height):
    """
    Ensure no inland ocean patches exist.
    Ocean tiles must be connected to map boundary.
    Use flood-fill from map edges to find connected ocean.
    Convert any unconnected ocean to land.
    """
    ocean_mask = ~is_land
    
    # Create mask of ocean connected to boundary using flood fill
    connected_ocean = np.zeros((height, width), dtype=bool)
    visited = np.zeros((height, width), dtype=bool)
    
    # Start flood fill from all boundary ocean tiles
    queue = deque()
    
    # Add all boundary tiles that are ocean
    for x in range(width):
        if ocean_mask[0, x]:  # Top edge
            queue.append((0, x))
            visited[0, x] = True
        if ocean_mask[height-1, x]:  # Bottom edge
            queue.append((height-1, x))
            visited[height-1, x] = True
    
    for y in range(height):
        if ocean_mask[y, 0]:  # Left edge
            queue.append((y, 0))
            visited[y, 0] = True
        if ocean_mask[y, width-1]:  # Right edge
            queue.append((y, width-1))
            visited[y, width-1] = True
    
    # Flood fill to find all connected ocean
    while queue:
        y, x = queue.popleft()
        
        if not ocean_mask[y, x]:
            continue
        
        connected_ocean[y, x] = True
        
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                if not visited[ny, nx] and ocean_mask[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
    
    # Find inland ocean tiles (ocean not connected to boundary)
    inland_ocean = ocean_mask & ~connected_ocean
    inland_count = inland_ocean.sum()
    
    # Convert inland ocean to land
    is_land_fixed = is_land.copy()
    is_land_fixed[inland_ocean] = True
    
    return is_land_fixed, inland_count, connected_ocean


def flood_fill_ocean_validation(is_land, width, height):
    """Wrapper for ocean validation with statistics."""
    is_land_fixed, inland_count, connected_ocean = validate_ocean_connectivity(is_land, width, height)
    return is_land_fixed, inland_count


# =============================================================================
# MOUNTAIN IDENTIFICATION
# =============================================================================
def identify_mountains(heightmap, is_land, convergent_boundaries, target_mountain_ratio=0.10):
    """
    Identify mountain tiles.
    Mountains must be exactly 10% of land tiles.
    Mountains form along convergent boundaries.
    """
    height, width = heightmap.shape
    land_count = is_land.sum()
    target_mountain_count = int(land_count * target_mountain_ratio)
    
    mountain_score = np.zeros((height, width))
    
    boundary_distance = ndimage.distance_transform_edt(~convergent_boundaries)
    max_dist = boundary_distance.max() if boundary_distance.max() > 0 else 1
    boundary_score = 1.0 - (boundary_distance / max_dist)
    
    mountain_score = heightmap * 0.4 + boundary_score * 0.6
    mountain_score[~is_land] = -1
    
    flat_scores = mountain_score.flatten()
    flat_indices = np.argsort(flat_scores)[::-1]
    
    is_mountain = np.zeros((height, width), dtype=bool)
    mountain_count = 0
    
    for idx in flat_indices:
        if mountain_count >= target_mountain_count:
            break
        y, x = idx // width, idx % width
        if is_land[y, x] and flat_scores[idx] >= 0:
            is_mountain[y, x] = True
            mountain_count += 1
    
    is_mountain = ensure_mountain_chains(is_mountain, heightmap, is_land, target_mountain_count)
    
    return is_mountain


def ensure_mountain_chains(is_mountain, heightmap, is_land, target_count):
    """Ensure mountains form continuous chains."""
    height, width = is_mountain.shape
    
    labeled, num_features = ndimage.label(is_mountain)
    
    if num_features <= 3:
        return is_mountain
    
    component_sizes = ndimage.sum(is_mountain, labeled, range(1, num_features + 1))
    large_threshold = target_count * 0.05
    
    for comp_id in range(1, num_features + 1):
        if component_sizes[comp_id - 1] < large_threshold:
            comp_tiles = np.argwhere(labeled == comp_id)
            
            for ty, tx in comp_tiles:
                for radius in range(1, 20):
                    found = False
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            if abs(dy) != radius and abs(dx) != radius:
                                continue
                            ny, nx = ty + dy, tx + dx
                            if 0 <= ny < height and 0 <= nx < width:
                                if is_mountain[ny, nx] and labeled[ny, nx] != comp_id:
                                    path = bresenham_line(tx, ty, nx, ny)
                                    for px, py in path:
                                        if 0 <= py < height and 0 <= px < width:
                                            if is_land[py, px]:
                                                is_mountain[py, px] = True
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break
    
    # Adjust to target count
    current_count = is_mountain.sum()
    while current_count > target_count:
        mountain_elevations = []
        for y in range(height):
            for x in range(width):
                if is_mountain[y, x]:
                    mountain_elevations.append((heightmap[y, x], y, x))
        mountain_elevations.sort()
        
        if mountain_elevations:
            _, y, x = mountain_elevations[0]
            is_mountain[y, x] = False
            current_count -= 1
    
    while current_count < target_count:
        candidates = []
        for y in range(height):
            for x in range(width):
                if is_land[y, x] and not is_mountain[y, x]:
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if is_mountain[ny, nx]:
                                candidates.append((heightmap[y, x], y, x))
                                break
        
        if not candidates:
            break
        
        candidates.sort(reverse=True)
        _, y, x = candidates[0]
        is_mountain[y, x] = True
        current_count += 1
    
    return is_mountain


def bresenham_line(x0, y0, x1, y1):
    """Generate points on a line using Bresenham's algorithm."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    
    return points


# =============================================================================
# DISTANCE FIELD COMPUTATION
# =============================================================================

def is_valid_lake_inlet(y, x, flow_dir, is_lake, width, height):
    """
    Check if a tile adjacent to a lake is a valid inlet (flows into the lake).
    
    Rivers should only touch lakes if they actually flow INTO them.
    Returns True if:
    - The tile is not adjacent to any lake, OR
    - The tile's flow direction points into a lake tile
    
    Returns False if:
    - The tile is adjacent to a lake but doesn't flow into it (just passing by)
    """
    if is_lake is None:
        return True
    
    # Check if adjacent to any lake tile
    adjacent_to_lake = False
    for dy, dx in D8_DIRECTIONS:
        ny, nx = y + dy, x + dx
        if 0 <= ny < height and 0 <= nx < width:
            if is_lake[ny, nx]:
                adjacent_to_lake = True
                break
    
    if not adjacent_to_lake:
        return True  # Not near a lake, always valid
    
    # Adjacent to lake - check if flow direction points into the lake
    dir_idx = flow_dir[y, x]
    if dir_idx < 0:
        return False  # No flow direction, don't place river here near lake
    
    dy, dx = D8_DIRECTIONS[dir_idx]
    ny, nx = y + dy, x + dx
    
    if 0 <= ny < height and 0 <= nx < width:
        if is_lake[ny, nx]:
            return True  # Flow points into lake - valid inlet
    
    return False  # Adjacent to lake but flow doesn't point into it


def compute_distance_from_ocean(is_land, width, height):
    """Compute distance from ocean for each tile using EDT."""
    # Ocean tiles are False in is_land, so distance from ocean = distance transform of is_land
    distance = ndimage.distance_transform_edt(is_land)
    return distance


def compute_distance_from_rivers(is_river, width, height):
    """
    Compute distance from rivers for each tile using EDT.
    PLACEHOLDER: River system disabled - this function is kept for API compatibility
    but will return max distance everywhere when is_river is all False.
    """
    # is_river is True for river tiles, so invert for EDT
    river_mask = ~is_river
    distance = ndimage.distance_transform_edt(river_mask)
    return distance


def compute_distance_from_lakes(is_lake, width, height):
    """Compute distance from lakes for each tile using EDT."""
    lake_mask = ~is_lake
    distance = ndimage.distance_transform_edt(lake_mask)
    return distance


# =============================================================================
# DERIVED TERRAIN MAPS (for expanded biome system)
# =============================================================================
def compute_slope_map(elevation, width, height):
    """
    Compute slope map from elevation gradients.
    Uses Sobel operators for gradient calculation.
    Returns normalized slope values 0-1.
    """
    # Compute gradients using Sobel operators
    grad_y = ndimage.sobel(elevation, axis=0, mode='reflect')
    grad_x = ndimage.sobel(elevation, axis=1, mode='reflect')
    
    # Calculate slope magnitude
    slope = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize to 0-1 based on distribution
    if slope.max() > 0:
        # Use percentile-based normalization for robustness
        p95 = np.percentile(slope, 95)
        if p95 > 0:
            slope = np.clip(slope / p95, 0, 1)
        else:
            slope = slope / slope.max() if slope.max() > 0 else slope
    
    return slope


def compute_roughness_map(elevation, width, height, window_size=3):
    """
    Compute terrain roughness from local height variance.
    Roughness = standard deviation of elevation in local window.
    Returns normalized roughness values 0-1.
    """
    from scipy.ndimage import generic_filter
    
    def local_std(values):
        return np.std(values)
    
    # Compute local standard deviation
    roughness = generic_filter(elevation, local_std, size=window_size, mode='reflect')
    
    # Normalize to 0-1 based on distribution
    if roughness.max() > 0:
        p95 = np.percentile(roughness, 95)
        if p95 > 0:
            roughness = np.clip(roughness / p95, 0, 1)
        else:
            roughness = roughness / roughness.max() if roughness.max() > 0 else roughness
    
    return roughness


def compute_distance_to_coast(is_land, width, height):
    """
    Compute distance to coastline for each tile.
    Positive for land tiles, negative for ocean tiles (optional).
    Returns distance in tiles.
    """
    # Find coastline tiles (land tiles adjacent to ocean)
    from scipy.ndimage import binary_dilation
    
    # Dilate land mask
    dilated = binary_dilation(is_land)
    # Coastline is where dilated differs from original AND is land
    coastline = is_land & ~binary_dilation(~is_land, iterations=0)
    
    # Actually, coastline = land tiles adjacent to ocean
    # Use morphological gradient
    ocean_mask = ~is_land
    land_adjacent_to_ocean = binary_dilation(ocean_mask) & is_land
    
    # Compute distance from coastline for all tiles
    coast_mask = ~land_adjacent_to_ocean  # Invert for EDT
    distance = ndimage.distance_transform_edt(coast_mask)
    
    return distance


def classify_landforms(elevation, slope, is_land, is_mountain):
    """
    Classify each tile into a landform category using elevation and slope.
    Categories: 'plains', 'hills', 'mountains'
    Uses dynamic thresholds based on data distribution.
    """
    height, width = elevation.shape
    landforms = np.empty((height, width), dtype=object)
    landforms.fill('ocean')
    
    # Get land elevation statistics for dynamic thresholds
    land_elev = elevation[is_land]
    land_slope = slope[is_land]
    
    if len(land_elev) == 0:
        return landforms
    
    # Dynamic thresholds based on percentiles
    elev_low = np.percentile(land_elev, 30)    # Below this = plains
    elev_mid = np.percentile(land_elev, 70)    # Above this = mountains potential
    elev_high = np.percentile(land_elev, 85)   # Above this = definitely mountains
    
    slope_low = np.percentile(land_slope, 40)  # Below this = flat (plains)
    slope_mid = np.percentile(land_slope, 70)  # Above this = steep (hills/mountains)
    slope_high = np.percentile(land_slope, 85) # Above this = very steep (mountains)
    
    for y in range(height):
        for x in range(width):
            if not is_land[y, x]:
                continue
            
            elev = elevation[y, x]
            slp = slope[y, x]
            
            # Mountains: pre-determined OR very high elevation OR very steep
            if is_mountain[y, x]:
                landforms[y, x] = 'mountains'
            elif elev > elev_high or slp > slope_high:
                landforms[y, x] = 'mountains'
            # Hills: mid elevation OR moderate slope
            elif elev > elev_low and (slp > slope_low or elev > elev_mid):
                landforms[y, x] = 'hills'
            # Plains: low elevation AND low slope
            else:
                landforms[y, x] = 'plains'
    
    return landforms


# =============================================================================
# EXPANDED BIOME COLOR MAP
# =============================================================================
EXPANDED_BIOME_COLORS = {
    # Ocean
    'ocean': '#1a5276',
    
    # Plains biomes
    'grassland': '#90c965',
    'meadow': '#7dcea0',
    'steppe': '#c4b896',
    'savanna': '#d4b86a',
    
    # Forest biomes
    'temperate_forest': '#228b22',
    'woodland': '#6b8e23',
    'tropical_forest': '#006400',
    'rainforest': '#004d00',
    
    # Desert biomes
    'sand_desert': '#f4d03f',
    'rock_desert': '#a0522d',
    'badlands': '#cd853f',
    'oasis': '#32cd32',
    
    # Snow biomes
    'snow_plains': '#e8f4f8',
    'snow_forest': '#a8d8ea',
    'snow_hills': '#b8c9d4',
    'glacier': '#e0ffff',
    
    # Hill biomes
    'grassy_hills': '#5c8a4d',
    'forest_hills': '#2d5a3f',
    'rocky_hills': '#6b5344',
    
    # Mountain biomes
    'rocky_mountains': '#5d6d7e',
    'snow_mountains': '#d5d8dc',
    'forest_mountains': '#3d5c4f',
    'alpine_meadows': '#8fbc8f',
    
    # Wetland biomes
    'swamp': '#4a5d23',
    'marsh': '#6b7b3a',
    'mangrove': '#3d5e3a',
    
    # Water features
    'lake': '#3498db',
    'river': '#1e90ff',  # Dodger blue
    
    # Legacy biomes (for backward compatibility)
    'plains': '#90c965',
    'forest': '#228b22',
    'desert': '#f4d03f',
    'snow_tundra': '#e8f4f8',
    'mountain': '#5d6d7e',
}


def compute_expanded_biome_thresholds(temperature, moisture, elevation, slope, roughness, is_land):
    """
    Compute dynamic thresholds for biome assignment based on data distributions.
    Returns a dictionary of threshold values derived from percentiles.
    """
    # Get land-only values
    temp_land = temperature[is_land]
    moist_land = moisture[is_land]
    elev_land = elevation[is_land]
    slope_land = slope[is_land]
    rough_land = roughness[is_land]
    
    if len(temp_land) == 0:
        # Return default thresholds if no land
        return {
            'temp_cold': 0.25, 'temp_cool': 0.35, 'temp_warm': 0.55, 'temp_hot': 0.70,
            'moist_arid': 0.20, 'moist_dry': 0.35, 'moist_moderate': 0.50, 'moist_wet': 0.65, 'moist_very_wet': 0.80,
            'elev_low': 0.25, 'elev_mid': 0.50, 'elev_high': 0.70, 'elev_alpine': 0.85,
            'slope_flat': 0.15, 'slope_gentle': 0.30, 'slope_moderate': 0.50, 'slope_steep': 0.70,
            'rough_smooth': 0.20, 'rough_moderate': 0.40, 'rough_rough': 0.60,
        }
    
    thresholds = {
        # Temperature thresholds (percentile-based)
        'temp_cold': np.percentile(temp_land, 20),      # Cold
        'temp_cool': np.percentile(temp_land, 35),      # Cool
        'temp_warm': np.percentile(temp_land, 65),      # Warm
        'temp_hot': np.percentile(temp_land, 80),       # Hot
        
        # Moisture thresholds
        'moist_arid': np.percentile(moist_land, 15),    # Very dry
        'moist_dry': np.percentile(moist_land, 30),     # Dry
        'moist_moderate': np.percentile(moist_land, 50),# Moderate
        'moist_wet': np.percentile(moist_land, 70),     # Wet
        'moist_very_wet': np.percentile(moist_land, 85),# Very wet
        
        # Elevation thresholds
        'elev_low': np.percentile(elev_land, 25),
        'elev_mid': np.percentile(elev_land, 50),
        'elev_high': np.percentile(elev_land, 75),
        'elev_alpine': np.percentile(elev_land, 90),
        
        # Slope thresholds
        'slope_flat': np.percentile(slope_land, 25),
        'slope_gentle': np.percentile(slope_land, 45),
        'slope_moderate': np.percentile(slope_land, 65),
        'slope_steep': np.percentile(slope_land, 85),
        
        # Roughness thresholds
        'rough_smooth': np.percentile(rough_land, 30),
        'rough_moderate': np.percentile(rough_land, 55),
        'rough_rough': np.percentile(rough_land, 80),
    }
    
    return thresholds


def assign_expanded_biome(temp, moist, elev, slope, roughness, dist_river, dist_coast, 
                          landform, thresholds, is_lake_tile, is_river_tile):
    """
    Assign a specific biome to a single tile based on environmental conditions.
    Uses relative comparisons with dynamic thresholds.
    """
    # Special cases first
    if is_lake_tile:
        return 'lake'
    
    t = thresholds
    
    # River proximity factor (normalized)
    near_river = dist_river < 5
    very_near_river = dist_river < 2
    
    # Coast proximity factor
    near_coast = dist_coast < 8
    very_near_coast = dist_coast < 3
    
    # ==========================================================================
    # MOUNTAIN BIOMES (landform == 'mountains')
    # ==========================================================================
    if landform == 'mountains':
        # Glacier: coldest, highest
        if temp < t['temp_cold'] and elev > t['elev_alpine']:
            return 'glacier'
        # Snow mountains: cold, high
        if temp < t['temp_cool']:
            return 'snow_mountains'
        # Forest mountains: moderate temp, sufficient moisture
        if temp >= t['temp_cool'] and temp < t['temp_warm'] and moist > t['moist_moderate']:
            return 'forest_mountains'
        # Alpine meadows: moderate temp, near snowline
        if temp >= t['temp_cool'] and temp < t['temp_warm'] and elev > t['elev_high']:
            return 'alpine_meadows'
        # Rocky mountains: default mountain
        return 'rocky_mountains'
    
    # ==========================================================================
    # HILL BIOMES (landform == 'hills')
    # ==========================================================================
    if landform == 'hills':
        # Snow hills: cold
        if temp < t['temp_cool']:
            return 'snow_hills'
        # Forest hills: high moisture, moderate temp
        if moist > t['moist_wet'] and temp >= t['temp_cool']:
            return 'forest_hills'
        # Rocky hills: low moisture, high roughness
        if moist < t['moist_dry'] or roughness > t['rough_rough']:
            return 'rocky_hills'
        # Grassy hills: default for moderate conditions
        return 'grassy_hills'
    
    # ==========================================================================
    # PLAINS BIOMES (landform == 'plains')
    # ==========================================================================
    # Check for snow biomes first (cold overrides other conditions)
    if temp < t['temp_cold']:
        if moist > t['moist_wet']:
            return 'snow_forest'
        return 'snow_plains'
    
    # Wetland biomes (very wet, low elevation, near water)
    if moist > t['moist_very_wet'] and elev < t['elev_mid']:
        # Mangrove: warm coastal
        if temp > t['temp_warm'] and very_near_coast:
            return 'mangrove'
        # Swamp: near rivers in very wet areas
        if very_near_river and temp >= t['temp_cool']:
            return 'swamp'
        # Marsh: wet lowlands near water
        if near_river and temp >= t['temp_cool']:
            return 'marsh'
    
    # Desert biomes (hot and dry)
    if temp > t['temp_warm'] and moist < t['moist_dry']:
        # Oasis: desert with river nearby
        if near_river:
            return 'oasis'
        # Badlands: dry, rough terrain
        if roughness > t['rough_moderate']:
            return 'badlands'
        # Rock desert: rough or high terrain in desert climate
        if roughness > t['rough_smooth'] or elev > t['elev_mid']:
            return 'rock_desert'
        # Sand desert: flat, hot, dry
        return 'sand_desert'
    
    # Forest biomes (sufficient moisture)
    if moist > t['moist_wet']:
        # Rainforest: hot, extremely wet
        if temp > t['temp_warm'] and moist > t['moist_very_wet']:
            return 'rainforest'
        # Tropical forest: warm, wet
        if temp > t['temp_warm']:
            return 'tropical_forest'
        # Temperate forest: moderate temp, wet
        if temp >= t['temp_cool']:
            return 'temperate_forest'
    
    # Woodland (moderate moisture with some trees)
    if moist > t['moist_moderate']:
        return 'woodland'
    
    # Plains biomes (remaining cases)
    # Steppe: dry plains
    if moist < t['moist_moderate']:
        return 'steppe'
    
    # Savanna: warm with moderate moisture
    if temp > t['temp_warm']:
        return 'savanna'
    
    # Meadow: near rivers with decent moisture
    if near_river and moist >= t['moist_moderate']:
        return 'meadow'
    
    # Grassland: default plains
    return 'grassland'


def assign_biomes_expanded(width, height, seed, is_land, is_mountain, is_lake, is_river,
                           temperature, moisture, elevation, slope, roughness,
                           dist_river, dist_coast, is_island=None):
    """
    Expanded biome assignment using environmental variables.
    
    This function replaces the old biome scoring system with a rule-based
    approach using dynamic thresholds derived from data distributions.
    """
    rng = np.random.default_rng(seed + 7000)
    
    biomes = np.empty((height, width), dtype=object)
    biomes.fill('ocean')
    
    # Combine land and islands
    if is_island is not None:
        land_mask = is_land | is_island
    else:
        land_mask = is_land
    
    # Classify landforms
    landforms = classify_landforms(elevation, slope, land_mask, is_mountain)
    
    # Compute dynamic thresholds
    thresholds = compute_expanded_biome_thresholds(
        temperature, moisture, elevation, slope, roughness, land_mask
    )
    
    # First pass: assign biomes
    for y in range(height):
        for x in range(width):
            if not land_mask[y, x]:
                biomes[y, x] = 'ocean'
                continue
            
            biome = assign_expanded_biome(
                temp=temperature[y, x],
                moist=moisture[y, x],
                elev=elevation[y, x],
                slope=slope[y, x],
                roughness=roughness[y, x],
                dist_river=dist_river[y, x],
                dist_coast=dist_coast[y, x],
                landform=landforms[y, x],
                thresholds=thresholds,
                is_lake_tile=is_lake[y, x],
                is_river_tile=is_river[y, x]
            )
            biomes[y, x] = biome
    
    # Smoothing pass for coherence
    biomes = smooth_expanded_biomes(biomes, land_mask, temperature, moisture, elevation,
                                     slope, roughness, dist_river, dist_coast, landforms,
                                     thresholds, is_lake, is_river, seed)
    
    return biomes


def smooth_expanded_biomes(biomes, land_mask, temperature, moisture, elevation,
                           slope, roughness, dist_river, dist_coast, landforms,
                           thresholds, is_lake, is_river, seed, num_passes=2):
    """
    Apply smoothing to expanded biomes for spatial coherence.
    Isolated tiles are reassigned based on neighbors.
    """
    height, width = biomes.shape
    rng = np.random.default_rng(seed + 7100)
    
    for _ in range(num_passes):
        changes = []
        
        for y in range(height):
            for x in range(width):
                if not land_mask[y, x] or is_lake[y, x]:
                    continue
                
                current = biomes[y, x]
                
                # Count neighbor biomes
                neighbor_counts = {}
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if land_mask[ny, nx]:
                                nb = biomes[ny, nx]
                                neighbor_counts[nb] = neighbor_counts.get(nb, 0) + 1
                
                if not neighbor_counts:
                    continue
                
                # Check if current biome is isolated (no matching neighbors)
                if current not in neighbor_counts or neighbor_counts[current] < 2:
                    # Find most common valid neighbor
                    sorted_neighbors = sorted(neighbor_counts.items(), key=lambda x: x[1], reverse=True)
                    
                    for candidate, count in sorted_neighbors:
                        if candidate == 'lake' or candidate == 'ocean':
                            continue
                        # Validate candidate is compatible with this tile's conditions
                        if is_biome_compatible(candidate, landforms[y, x], temperature[y, x], 
                                              moisture[y, x], thresholds):
                            changes.append((y, x, candidate))
                            break
        
        # Apply changes
        for y, x, new_biome in changes:
            biomes[y, x] = new_biome
    
    return biomes


def is_biome_compatible(biome, landform, temp, moist, thresholds):
    """Check if a biome is compatible with the tile's environmental conditions."""
    t = thresholds
    
    # Mountain biomes only on mountains
    mountain_biomes = {'glacier', 'snow_mountains', 'rocky_mountains', 'forest_mountains', 'alpine_meadows'}
    if biome in mountain_biomes:
        return landform == 'mountains'
    
    # Hill biomes only on hills
    hill_biomes = {'grassy_hills', 'forest_hills', 'rocky_hills', 'snow_hills'}
    if biome in hill_biomes:
        return landform == 'hills'
    
    # Snow biomes need cold temp
    snow_biomes = {'snow_plains', 'snow_forest', 'snow_hills', 'glacier', 'snow_mountains'}
    if biome in snow_biomes:
        return temp < t['temp_cool']
    
    # Desert biomes need hot + dry
    desert_biomes = {'sand_desert', 'rock_desert', 'badlands', 'oasis'}
    if biome in desert_biomes:
        return temp > t['temp_warm'] and moist < t['moist_moderate']
    
    # Wetland biomes need high moisture
    wetland_biomes = {'swamp', 'marsh', 'mangrove'}
    if biome in wetland_biomes:
        return moist > t['moist_wet']
    
    # Forest biomes need moderate+ moisture
    forest_biomes = {'temperate_forest', 'woodland', 'tropical_forest', 'rainforest'}
    if biome in forest_biomes:
        return moist > t['moist_moderate']
    
    # Plains biomes are generally compatible with plains landform
    return True


# =============================================================================
# NATURAL CLIMATE SIMULATION SYSTEM
# =============================================================================

def compute_wind_direction_field(seed, width, height):
    """
    Compute a consistent prevailing wind direction for the world.
    Wind patterns influence moisture propagation and rain shadows.
    
    Returns:
        wind_angle: Angle in radians
        wind_dx, wind_dy: Unit vector components
    """
    rng = np.random.default_rng(seed + 3000)
    
    # Primary wind direction (deterministic from seed)
    # Tend towards trade wind patterns (east-to-west in tropics, west-to-east mid-latitudes)
    base_angle = rng.uniform(0, 2 * np.pi)
    
    wind_dx = np.cos(base_angle)
    wind_dy = np.sin(base_angle)
    
    return base_angle, wind_dx, wind_dy


def compute_slope_direction(elevation, width, height):
    """
    Compute slope direction (aspect) for each tile.
    Returns angle indicating which direction the slope faces (0-2pi).
    Also returns a unit vector field for the slope direction.
    """
    # Compute gradients using Sobel operators
    grad_y = ndimage.sobel(elevation, axis=0, mode='reflect')  # North-south gradient
    grad_x = ndimage.sobel(elevation, axis=1, mode='reflect')  # East-west gradient
    
    # Aspect angle (direction slope faces, in radians)
    aspect = np.arctan2(-grad_y, -grad_x)  # Points downhill
    
    return aspect, grad_x, grad_y


def generate_natural_temperature(width, height, seed, elevation, is_land, is_mountain,
                                  dist_ocean, dist_river, dist_lake, slope, aspect,
                                  forest_hint=None):
    """
    Generate temperature map using natural geographic influences.
    
    Components:
    1. Latitude influence (equator warm, poles cold)
    2. Elevation cooling (higher = colder)
    3. Ocean moderation (coastal areas have milder temperatures)
    4. River/lake cooling (water bodies cool nearby land)
    5. Forest cooling (shading and evapotranspiration)
    6. Slope orientation (equator-facing slopes warmer)
    
    All thresholds derived dynamically from data distributions.
    """
    temperature = np.zeros((height, width))
    
    # Get land elevation stats for dynamic scaling
    land_elev = elevation[is_land]
    elev_mean = np.mean(land_elev) if len(land_elev) > 0 else 0.5
    elev_std = np.std(land_elev) if len(land_elev) > 0 else 0.2
    elev_max = np.max(land_elev) if len(land_elev) > 0 else 1.0
    
    # Distance stats for dynamic decay rates
    land_dist_ocean = dist_ocean[is_land]
    ocean_dist_median = np.median(land_dist_ocean) if len(land_dist_ocean) > 0 else 20.0
    
    land_dist_river = dist_river[is_land]
    river_dist_p75 = np.percentile(land_dist_river, 75) if len(land_dist_river) > 0 else 15.0
    
    land_dist_lake = dist_lake[is_land]
    lake_dist_p75 = np.percentile(land_dist_lake, 75) if len(land_dist_lake) > 0 else 20.0
    
    # Dynamic decay rates based on world geography
    ocean_moderation_decay = max(10.0, ocean_dist_median * 0.8)
    river_cooling_decay = max(5.0, river_dist_p75 * 0.3)
    lake_cooling_decay = max(8.0, lake_dist_p75 * 0.3)
    
    # Elevation cooling factor (scaled to elevation distribution)
    elev_cooling_scale = 0.5 / max(0.3, elev_max)
    
    for y in range(height):
        for x in range(width):
            # 1. LATITUDE INFLUENCE
            # Temperature varies smoothly from equator (warm) to poles (cold)
            # Using cosine for smoother transition
            lat_normalized = (y - height / 2) / (height / 2)  # -1 to 1
            latitude_temp = 0.5 * (1.0 + np.cos(np.pi * abs(lat_normalized)))  # 1 at equator, 0 at poles
            
            # 2. ELEVATION COOLING
            # Higher terrain is colder - scale based on elevation distribution
            elev = elevation[y, x]
            elevation_cooling = elev * elev_cooling_scale
            
            # 3. OCEAN TEMPERATURE MODERATION
            # Coastal areas have milder temperatures (less extreme hot/cold)
            # Ocean acts as thermal mass
            d_ocean = dist_ocean[y, x]
            ocean_moderation = np.exp(-d_ocean / ocean_moderation_decay)
            # Moderation pulls temperature toward moderate (0.5)
            moderation_target = 0.5
            
            # 4. RIVER COOLING
            d_river = dist_river[y, x]
            river_cooling = 0.08 * np.exp(-d_river / river_cooling_decay)
            
            # 5. LAKE COOLING
            d_lake = dist_lake[y, x]
            lake_cooling = 0.10 * np.exp(-d_lake / lake_cooling_decay)
            
            # 6. SLOPE ORIENTATION (sun exposure)
            # Slopes facing the equator receive more sunlight
            if is_land[y, x] and slope[y, x] > 0.1:
                # Determine if slope faces toward equator
                slope_aspect = aspect[y, x]
                
                # Equator is at y = height/2
                # North hemisphere (y < height/2): equator-facing = south-facing (aspect ~ pi/2)
                # South hemisphere (y > height/2): equator-facing = north-facing (aspect ~ -pi/2)
                if y < height / 2:
                    equator_angle = np.pi / 2  # South
                else:
                    equator_angle = -np.pi / 2  # North
                
                # Calculate how much the slope faces the equator
                angle_diff = np.abs(slope_aspect - equator_angle)
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                
                # Equator-facing slopes are warmer, pole-facing are cooler
                slope_sun_factor = np.cos(angle_diff) * slope[y, x] * 0.1
            else:
                slope_sun_factor = 0.0
            
            # 7. FOREST COOLING (if forest hint available)
            forest_cooling = 0.0
            if forest_hint is not None and is_land[y, x]:
                forest_cooling = forest_hint[y, x] * 0.05
            
            # Combine all factors
            base_temp = latitude_temp - elevation_cooling + slope_sun_factor
            base_temp = base_temp - river_cooling - lake_cooling - forest_cooling
            
            # Apply ocean moderation (blend toward moderate temperature)
            if is_land[y, x]:
                base_temp = base_temp * (1 - ocean_moderation * 0.3) + moderation_target * ocean_moderation * 0.3
            
            temperature[y, x] = base_temp
    
    # Normalize temperature to [0, 1] based on land values
    if is_land.any():
        land_temp = temperature[is_land]
        t_min, t_max = land_temp.min(), land_temp.max()
        if t_max > t_min:
            temperature[is_land] = (temperature[is_land] - t_min) / (t_max - t_min)
        temperature = np.clip(temperature, 0.0, 1.0)
    
    # Ocean gets moderate temperature
    temperature[~is_land] = 0.5
    
    return temperature


def propagate_moisture_with_wind(moisture_sources, elevation, is_land, is_mountain,
                                  wind_dx, wind_dy, width, height, iterations=50):
    """
    Propagate moisture from sources (ocean, lakes, rivers) following wind direction.
    Implements advection-diffusion with terrain blocking.
    
    Returns moisture field after propagation.
    """
    moisture = moisture_sources.copy()
    
    # Diffusion coefficient (how much moisture spreads locally)
    diffusion = 0.15
    
    # Advection strength (how much wind carries moisture)
    advection = 0.25
    
    # Elevation loss factor (moisture condenses as air rises)
    elev_loss_rate = 0.3
    
    # Get elevation statistics for dynamic scaling
    land_elev = elevation[is_land]
    elev_p75 = np.percentile(land_elev, 75) if len(land_elev) > 0 else 0.6
    
    for _ in range(iterations):
        new_moisture = moisture.copy()
        
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if not is_land[y, x]:
                    continue
                
                current = moisture[y, x]
                current_elev = elevation[y, x]
                
                # Local diffusion (average of neighbors)
                neighbors = [
                    moisture[y-1, x], moisture[y+1, x],
                    moisture[y, x-1], moisture[y, x+1]
                ]
                diffused = np.mean(neighbors) * diffusion + current * (1 - diffusion)
                
                # Advection from upwind direction
                upwind_x = int(x - wind_dx)
                upwind_y = int(y - wind_dy)
                
                if 0 <= upwind_x < width and 0 <= upwind_y < height:
                    upwind_moisture = moisture[upwind_y, upwind_x]
                    upwind_elev = elevation[upwind_y, upwind_x]
                    
                    # Elevation change causes moisture loss (orographic precipitation)
                    elev_change = max(0, current_elev - upwind_elev)
                    elev_loss = elev_change * elev_loss_rate / max(0.1, elev_p75)
                    
                    # Advected moisture with elevation loss
                    advected = upwind_moisture * (1 - min(0.8, elev_loss))
                    
                    # Blend advection with diffusion
                    new_moisture[y, x] = diffused * (1 - advection) + advected * advection
                else:
                    new_moisture[y, x] = diffused
                
                # Mountain blocking (sharp elevation reduces moisture more)
                if is_mountain[y, x]:
                    new_moisture[y, x] *= 0.7
        
        moisture = new_moisture
    
    return moisture


def compute_rain_shadow(elevation, is_mountain, wind_dx, wind_dy, width, height):
    """
    Compute rain shadow intensity map.
    Areas downwind of mountains receive less moisture.
    
    Returns shadow_strength map (0 = no shadow, 1 = full shadow).
    """
    shadow = np.zeros((height, width))
    
    # Get mountain elevation stats for dynamic thresholds
    if is_mountain.any():
        mountain_elev = elevation[is_mountain]
        mountain_elev_median = np.median(mountain_elev)
    else:
        mountain_elev_median = 0.7
    
    # Maximum distance to check for upwind mountains
    max_shadow_dist = 60
    
    for y in range(height):
        for x in range(width):
            # Check upwind for mountains
            shadow_strength = 0.0
            
            for dist in range(3, max_shadow_dist, 2):
                # Position upwind
                check_x = int(x - wind_dx * dist)
                check_y = int(y - wind_dy * dist)
                
                if not (0 <= check_x < width and 0 <= check_y < height):
                    break
                
                # Check for high terrain (mountains or high elevation)
                if is_mountain[check_y, check_x] or elevation[check_y, check_x] > mountain_elev_median:
                    # Calculate shadow strength based on:
                    # - Distance (closer mountains cast stronger shadow)
                    # - Elevation difference (higher mountains cast stronger shadow)
                    dist_factor = 1.0 - (dist / max_shadow_dist)
                    elev_factor = elevation[check_y, check_x] / mountain_elev_median
                    
                    local_shadow = dist_factor * min(1.0, elev_factor) * 0.9
                    shadow_strength = max(shadow_strength, local_shadow)
            
            shadow[y, x] = shadow_strength
    
    # Smooth for natural transitions
    shadow = ndimage.gaussian_filter(shadow, sigma=2)
    
    return shadow


def generate_natural_moisture(width, height, seed, elevation, is_land, is_mountain,
                               dist_ocean, dist_river, dist_lake, temperature,
                               slope=None, forest_hint=None):
    """
    Generate moisture map using natural atmospheric simulation.
    
    Components:
    1. Ocean moisture source (primary humidity source)
    2. Wind-based moisture advection
    3. Mountain rain shadow effect
    4. Elevation moisture loss
    5. River moisture contribution
    6. Lake moisture contribution
    7. Forest transpiration feedback
    8. Temperature interaction
    
    All parameters derived dynamically from data distributions.
    """
    # Get wind direction
    wind_angle, wind_dx, wind_dy = compute_wind_direction_field(seed, width, height)
    
    # Get statistics for dynamic scaling
    land_dist_ocean = dist_ocean[is_land]
    ocean_dist_max = np.max(land_dist_ocean) if len(land_dist_ocean) > 0 else 50.0
    ocean_dist_median = np.median(land_dist_ocean) if len(land_dist_ocean) > 0 else 20.0
    
    land_dist_river = dist_river[is_land]
    river_dist_p90 = np.percentile(land_dist_river, 90) if len(land_dist_river) > 0 else 25.0
    
    land_dist_lake = dist_lake[is_land]
    lake_dist_p90 = np.percentile(land_dist_lake, 90) if len(land_dist_lake) > 0 else 30.0
    
    # Dynamic decay rates
    ocean_decay = max(15.0, ocean_dist_median * 0.7)
    river_decay = max(5.0, river_dist_p90 * 0.2)
    lake_decay = max(8.0, lake_dist_p90 * 0.25)
    
    # 1. OCEAN MOISTURE SOURCE
    # Exponential decay from coast
    ocean_moisture = np.exp(-dist_ocean / ocean_decay)
    
    # 2. RIVER MOISTURE CONTRIBUTION
    river_moisture = 0.5 * np.exp(-dist_river / river_decay)
    
    # 3. LAKE MOISTURE CONTRIBUTION
    lake_moisture = 0.6 * np.exp(-dist_lake / lake_decay)
    
    # Combined base moisture sources
    moisture_sources = np.maximum(ocean_moisture, np.maximum(river_moisture, lake_moisture))
    
    # Ocean tiles are saturated
    moisture_sources[~is_land] = 1.0
    
    # 4. WIND-BASED MOISTURE PROPAGATION
    moisture = propagate_moisture_with_wind(
        moisture_sources, elevation, is_land, is_mountain,
        wind_dx, wind_dy, width, height, iterations=30
    )
    
    # 5. RAIN SHADOW EFFECT
    rain_shadow = compute_rain_shadow(elevation, is_mountain, wind_dx, wind_dy, width, height)
    
    # Apply rain shadow (reduce moisture on lee side of mountains)
    shadow_reduction = 1.0 - rain_shadow * 0.75  # Up to 75% reduction
    moisture = moisture * shadow_reduction
    
    # 6. ELEVATION MOISTURE LOSS
    # Higher elevations lose moisture due to precipitation
    land_elev = elevation[is_land]
    elev_p75 = np.percentile(land_elev, 75) if len(land_elev) > 0 else 0.6
    
    elev_loss = np.clip(elevation / elev_p75, 0, 1) * 0.25
    moisture = moisture * (1 - elev_loss)
    
    # 7. SUBTROPICAL DRY BELT
    # Hadley cell dynamics create dry zones at ~25-30 degrees latitude
    y_coords = np.arange(height).reshape(-1, 1) / height
    lat_from_equator = np.abs(y_coords - 0.5) * 2  # 0 at equator, 1 at poles
    
    # Subtropical high pressure belt
    subtropical_center = 0.30  # ~30% from equator
    subtropical_width = 0.12
    subtropical_dryness = np.exp(-((lat_from_equator - subtropical_center) ** 2) / (2 * subtropical_width ** 2))
    subtropical_factor = 1.0 - np.tile(subtropical_dryness, (1, width)) * 0.35
    
    moisture = moisture * subtropical_factor
    
    # 8. TEMPERATURE INTERACTION
    # Warm air holds more moisture, cold air less
    # Warm regions retain moisture, cold regions precipitate it out
    temp_factor = 0.7 + 0.3 * temperature  # 0.7-1.0 based on temperature
    moisture = moisture * temp_factor
    
    # 9. FOREST TRANSPIRATION FEEDBACK (if available)
    if forest_hint is not None:
        # Forests contribute humidity through transpiration
        forest_moisture_boost = forest_hint * 0.15
        moisture = np.clip(moisture + forest_moisture_boost, 0, 1)
    
    # 10. CONTINENTAL INTERIOR DRYNESS
    # Even with wind propagation, deep interiors tend to be drier
    continental_factor = 1.0 - np.clip(dist_ocean / ocean_dist_max, 0, 1) * 0.15
    moisture = moisture * continental_factor
    
    # Normalize to [0, 1] on land
    if is_land.any():
        land_moisture = moisture[is_land]
        m_min, m_max = land_moisture.min(), land_moisture.max()
        if m_max > m_min:
            moisture[is_land] = (moisture[is_land] - m_min) / (m_max - m_min)
    
    moisture = np.clip(moisture, 0.0, 1.0)
    
    # Ocean tiles have full moisture
    moisture[~is_land] = 1.0
    
    return moisture, (wind_dx, wind_dy), rain_shadow


# =============================================================================
# LEGACY CLIMATE GENERATION (kept for compatibility)
# =============================================================================
def generate_temperature(width, height, elevation, is_land, dist_river, dist_lake):
    """
    Generate temperature based on:
    - Latitude (latitude_factor)
    - Elevation cooling (-elevation * TEMP_ELEVATION_FACTOR)
    - Water cooling from rivers (-TEMP_RIVER_COOLING_WEIGHT * exp(-dist_river / TEMP_RIVER_COOLING_DECAY))
    - Water cooling from lakes (-TEMP_LAKE_COOLING_WEIGHT * exp(-dist_lake / TEMP_LAKE_COOLING_DECAY))
    
    Formula:
    temperature = latitude_factor - elevation*0.4 - 0.05*exp(-dist_river/10) - 0.08*exp(-dist_lake/15)
    """
    temperature = np.zeros((height, width))
    
    for y in range(height):
        for x in range(width):
            # Latitude factor: 1.0 at equator (y=height/2), 0.0 at poles
            latitude_factor = 1.0 - abs(y - height / 2) / (height / 2)
            
            # Elevation cooling
            elev = elevation[y, x]
            elevation_cooling = elev * TEMP_ELEVATION_FACTOR
            
            # Water cooling effects
            river_cooling = TEMP_RIVER_COOLING_WEIGHT * np.exp(-dist_river[y, x] / TEMP_RIVER_COOLING_DECAY)
            lake_cooling = TEMP_LAKE_COOLING_WEIGHT * np.exp(-dist_lake[y, x] / TEMP_LAKE_COOLING_DECAY)
            
            # Combined temperature
            temp = latitude_factor - elevation_cooling - river_cooling - lake_cooling
            temperature[y, x] = np.clip(temp, 0.0, 1.0)
    
    return temperature


# =============================================================================
# MOISTURE GENERATION - DISTANCE-BASED (NO NOISE)
# =============================================================================
def generate_moisture(width, height, seed, is_land, is_mountain, elevation, dist_ocean, dist_river, dist_lake):
    """
    Generate moisture map using distance-based exponential decay.
    
    Formula:
    moisture_raw = 1.0*exp(-dist_ocean/30) + 0.6*exp(-dist_river/10) + 0.8*exp(-dist_lake/18)
    
    Then apply:
    - Rain shadow reduction behind mountains (up to 80%)
    - Subtropical dry belt (latitudes 20-35% from equator)
    - Continental interior dryness (far from coast)
    
    Normalize to [0, 1].
    """
    rng = np.random.default_rng(seed + 2000)
    
    # Determine prevailing wind direction (deterministic from seed)
    wind_angle = rng.uniform(0, 2 * np.pi)
    wind_dx = np.cos(wind_angle)
    wind_dy = np.sin(wind_angle)
    
    moisture = np.zeros((height, width))
    
    # Compute base moisture from distance fields
    ocean_contribution = MOISTURE_OCEAN_WEIGHT * np.exp(-dist_ocean / MOISTURE_OCEAN_DECAY)
    river_contribution = MOISTURE_RIVER_WEIGHT * np.exp(-dist_river / MOISTURE_RIVER_DECAY)
    lake_contribution = MOISTURE_LAKE_WEIGHT * np.exp(-dist_lake / MOISTURE_LAKE_DECAY)
    
    moisture = ocean_contribution + river_contribution + lake_contribution
    
    # Apply rain shadow effect behind mountains
    shadow_map = np.zeros((height, width))
    
    for y in range(height):
        for x in range(width):
            if not is_land[y, x]:
                continue
            
            # Check for mountains upwind
            shadow_strength = 0.0
            for dist in range(5, 60, 5):
                check_x = int(x - wind_dx * dist)
                check_y = int(y - wind_dy * dist)
                
                if 0 <= check_x < width and 0 <= check_y < height:
                    if is_mountain[check_y, check_x]:
                        # Closer mountains cast stronger shadow
                        shadow_strength = max(shadow_strength, 1.0 - (dist / 60.0))
            
            shadow_map[y, x] = shadow_strength
    
    # Smooth the shadow map for natural transition
    shadow_map = ndimage.gaussian_filter(shadow_map, sigma=3)
    
    # Apply rain shadow - reduce moisture by up to 80% in shadow (increased from 70%)
    rain_shadow_factor = 1.0 - shadow_map * 0.80
    moisture = moisture * rain_shadow_factor
    
    # Apply subtropical dry belt (latitudes ~20-35% from equator, where deserts form)
    # Normalized y: 0 = top, 1 = bottom; equator at 0.5
    y_coords = np.arange(height).reshape(-1, 1) / height
    lat_from_equator = np.abs(y_coords - 0.5) * 2  # 0 at equator, 1 at poles
    
    # Subtropical high pressure belt: peak dryness between 0.2 and 0.4 latitude
    subtropical_dryness = np.exp(-((lat_from_equator - 0.30) ** 2) / (2 * 0.12 ** 2))
    subtropical_factor = 1.0 - subtropical_dryness * 0.4  # Up to 40% moisture reduction
    
    moisture = moisture * subtropical_factor
    
    # Apply continental interior dryness (far from coast = drier)
    # This creates characteristic dry interiors
    continental_dryness = 1.0 - np.clip(dist_ocean / 20.0, 0, 1) * 0.3
    moisture = moisture * continental_dryness
    
    # Normalize moisture on land tiles to [0, 1]
    land_moisture = moisture[is_land]
    if land_moisture.max() > land_moisture.min():
        moisture[is_land] = (moisture[is_land] - land_moisture.min()) / (land_moisture.max() - land_moisture.min())
    
    # Ocean tiles have max moisture
    moisture[~is_land] = 1.0
    
    return moisture, dist_ocean, (wind_dx, wind_dy)


# =============================================================================
# WATERSHED HYDROLOGY SYSTEM
# =============================================================================

# D8 flow direction encoding: (dy, dx) pairs for 8 neighbors
# Index 0-7 represents directions, -1 means no flow (basin/ocean)
D8_DIRECTIONS = [
    (-1, 0),   # 0: North
    (-1, 1),   # 1: Northeast
    (0, 1),    # 2: East
    (1, 1),    # 3: Southeast
    (1, 0),    # 4: South
    (1, -1),   # 5: Southwest
    (0, -1),   # 6: West
    (-1, -1),  # 7: Northwest
]

# Distance weights for D8 (cardinal = 1, diagonal = sqrt(2))
D8_DISTANCES = [1.0, 1.414, 1.0, 1.414, 1.0, 1.414, 1.0, 1.414]


def compute_flow_direction_field(elevation, is_land, is_lake, width, height):
    """
    STEP 1: Compute D8 flow direction for every land tile.
    
    For each tile, find the neighboring tile with steepest downhill slope.
    Returns:
    - flow_dir: array of direction indices (0-7) or -1 for basins
    - is_basin: boolean mask of tiles with no downhill neighbor
    """
    flow_dir = np.full((height, width), -1, dtype=np.int8)
    is_basin = np.zeros((height, width), dtype=bool)
    
    for y in range(height):
        for x in range(width):
            if not is_land[y, x]:
                # Ocean tiles have no flow direction
                continue
            
            current_elev = elevation[y, x]
            best_slope = 0
            best_dir = -1
            
            for dir_idx, (dy, dx) in enumerate(D8_DIRECTIONS):
                ny, nx = y + dy, x + dx
                if not (0 <= ny < height and 0 <= nx < width):
                    continue
                
                # Ocean is always lowest (flows to sea)
                if not is_land[ny, nx]:
                    slope = (current_elev + 1.0) / D8_DISTANCES[dir_idx]
                    if slope > best_slope:
                        best_slope = slope
                        best_dir = dir_idx
                else:
                    neighbor_elev = elevation[ny, nx]
                    if neighbor_elev < current_elev:
                        slope = (current_elev - neighbor_elev) / D8_DISTANCES[dir_idx]
                        if slope > best_slope:
                            best_slope = slope
                            best_dir = dir_idx
            
            flow_dir[y, x] = best_dir
            
            # Mark as basin if no downhill neighbor found (but not lake tiles)
            if best_dir == -1 and is_land[y, x] and (is_lake is None or not is_lake[y, x]):
                is_basin[y, x] = True
    
    return flow_dir, is_basin


def identify_basin(y, x, elevation, is_land, flow_dir, width, height, max_tiles=BASIN_FLOOD_LIMIT):
    """
    STEP 2: Identify basin region by flood-fill.
    
    Find all tiles that drain into this basin point using rising water simulation.
    Returns:
    - basin_tiles: set of (y, x) coordinates in the basin
    - rim_tiles: list of (elevation, y, x) tiles on basin perimeter
    """
    basin_tiles = set()
    rim_tiles = []
    visited = set()
    
    # Rising water approach
    water_level = elevation[y, x]
    max_rise = 0.25
    rise_step = 0.01
    
    while water_level < elevation[y, x] + max_rise and len(basin_tiles) < max_tiles:
        queue = deque([(y, x)])
        level_visited = set()
        level_visited.add((y, x))
        
        while queue and len(basin_tiles) < max_tiles:
            cy, cx = queue.popleft()
            
            if elevation[cy, cx] <= water_level:
                basin_tiles.add((cy, cx))
                
                for dy, dx in D8_DIRECTIONS:
                    ny, nx = cy + dy, cx + dx
                    if (ny, nx) in level_visited:
                        continue
                    if not (0 <= ny < height and 0 <= nx < width):
                        continue
                    
                    level_visited.add((ny, nx))
                    
                    # Ocean is an escape point
                    if not is_land[ny, nx]:
                        rim_tiles.append((-1.0, ny, nx))
                        continue
                    
                    if elevation[ny, nx] <= water_level:
                        queue.append((ny, nx))
                    elif (ny, nx) not in visited:
                        rim_tiles.append((elevation[ny, nx], ny, nx))
                        visited.add((ny, nx))
        
        # Found ocean outlet - done
        if any(e < 0 for e, _, _ in rim_tiles):
            break
        
        water_level += rise_step
    
    return basin_tiles, rim_tiles


def find_basin_outlet(y, x, elevation, is_land, width, height):
    """
    Find the nearest outlet from a basin point using BFS.
    
    Searches through terrain to find ocean or lower ground.
    Returns:
    - outlet: (y, x) of escape tile
    - path: list of tiles from basin to outlet
    """
    queue = deque([(y, x, [(y, x)])])
    visited = set()
    visited.add((y, x))
    
    current_elev = elevation[y, x]
    max_search = max(width, height)
    
    best_ocean_path = None
    best_lower_path = None
    
    while queue:
        cy, cx, path = queue.popleft()
        
        if len(path) > max_search:
            continue
        if best_ocean_path and len(path) >= len(best_ocean_path):
            continue
        
        for dy, dx in D8_DIRECTIONS:
            ny, nx = cy + dy, cx + dx
            if (ny, nx) in visited:
                continue
            if not (0 <= ny < height and 0 <= nx < width):
                continue
            
            visited.add((ny, nx))
            new_path = path + [(ny, nx)]
            
            # Ocean - best outlet
            if not is_land[ny, nx]:
                if best_ocean_path is None or len(new_path) < len(best_ocean_path):
                    best_ocean_path = new_path
                continue
            
            # Lower terrain
            if elevation[ny, nx] < current_elev - 0.005:
                if best_lower_path is None or len(new_path) < len(best_lower_path):
                    best_lower_path = new_path
            
            # Continue search - prioritize lower/similar elevation, but allow climbing
            if elevation[ny, nx] <= current_elev + 0.05:  # More permissive climbing (was 0.02)
                queue.appendleft((ny, nx, new_path))
            elif elevation[ny, nx] <= current_elev + 0.15:  # Allow moderate climb
                queue.append((ny, nx, new_path))
    
    if best_ocean_path:
        return (best_ocean_path[-1][0], best_ocean_path[-1][1]), best_ocean_path
    if best_lower_path:
        return (best_lower_path[-1][0], best_lower_path[-1][1]), best_lower_path
    
    return None, []


def carve_basin_outlet_path(path, elevation, is_land, width, height):
    """
    STEP 3: Carve a drainage path from basin to outlet.
    
    Creates a smooth downhill gradient along the path.
    Returns number of tiles carved.
    """
    if len(path) < 2:
        return 0
    
    carved_count = 0
    
    # Get start and end elevations
    start_y, start_x = path[0]
    end_y, end_x = path[-1]
    
    start_elev = elevation[start_y, start_x]
    
    if not is_land[end_y, end_x]:
        end_elev = 0.005  # Ocean level
    else:
        end_elev = min(start_elev - 0.02, elevation[end_y, end_x] - 0.01)
        end_elev = max(0.005, end_elev)
    
    total_drop = start_elev - end_elev
    carve_per_tile = min(BASIN_CARVE_AMOUNT, total_drop / max(1, len(path)))
    
    # Ensure monotonically decreasing elevation
    prev_elev = start_elev
    cumulative_carve = 0
    
    for i, (cy, cx) in enumerate(path):
        if not (0 <= cy < height and 0 <= cx < width):
            continue
        if not is_land[cy, cx]:
            continue
        
        t = i / max(1, len(path) - 1)
        target_elev = start_elev * (1 - t) + end_elev * t
        
        # Must be below previous tile
        target_elev = min(target_elev, prev_elev - 0.001)
        target_elev = max(0.005, target_elev)
        
        if elevation[cy, cx] > target_elev:
            carve_amount = elevation[cy, cx] - target_elev
            if cumulative_carve + carve_amount <= BASIN_CARVE_MAX_DEPTH:
                elevation[cy, cx] = target_elev
                cumulative_carve += carve_amount
                carved_count += 1
        
        prev_elev = min(prev_elev, elevation[cy, cx])
    
    return carved_count


def resolve_all_basins(elevation, is_land, is_lake, flow_dir, is_basin, width, height):
    """
    STEP 3 (continued): Resolve all basins by carving outlets.
    
    Iteratively finds and carves basins until all drainage reaches ocean.
    Returns updated elevation and flow_dir, plus carve statistics.
    """
    total_carves = 0
    iterations = 0
    
    while iterations < BASIN_MAX_ITERATIONS:
        # Find remaining basins
        basin_points = np.argwhere(is_basin)
        if len(basin_points) == 0:
            break
        
        # Sort by elevation (process lowest basins first)
        basin_list = [(elevation[y, x], y, x) for y, x in basin_points]
        basin_list.sort()
        
        carved_this_iteration = 0
        
        for _, by, bx in basin_list[:80]:  # Process more basins per iteration for faster convergence
            if not is_basin[by, bx]:
                continue
            
            outlet, path = find_basin_outlet(by, bx, elevation, is_land, width, height)
            
            if outlet and path:
                carved = carve_basin_outlet_path(path, elevation, is_land, width, height)
                carved_this_iteration += carved
                total_carves += carved
        
        if carved_this_iteration == 0:
            break
        
        # Recompute flow directions after carving
        flow_dir, is_basin = compute_flow_direction_field(
            elevation, is_land, is_lake, width, height
        )
        
        iterations += 1
    
    # FINAL PASS: Force ALL remaining basins to drain by connecting them
    # Process in multiple passes to chain basins together
    for pass_num in range(5):  # Multiple passes to chain basins
        basin_points = np.argwhere(is_basin)
        if len(basin_points) == 0:
            break
            
        forced_this_pass = 0
        
        # Sort by elevation (process lowest basins first - they need outlet most)
        basin_list = [(elevation[by, bx], by, bx) for by, bx in basin_points]
        basin_list.sort()
        
        for _, by, bx in basin_list:
            if flow_dir[by, bx] >= 0:
                continue  # Already has flow direction
            if not is_land[by, bx]:
                continue
            
            # Find the best neighbor to flow to
            # Priority: 1) Ocean, 2) Tile with existing outward flow (no cycle), 3) Downhill
            best_dir = -1
            best_score = -9999
            tile_elev = elevation[by, bx]
            
            for di, (dy, dx) in enumerate(D8_DIRECTIONS):
                ny, nx = by + dy, bx + dx
                if not (0 <= ny < height and 0 <= nx < width):
                    continue
                    
                if not is_land[ny, nx]:
                    # Ocean - best target
                    best_dir = di
                    break
                    
                if is_lake is not None and is_lake[ny, nx]:
                    continue  # Skip lakes
                
                # CRITICAL: Check if flowing here would create a 2-tile cycle
                neighbor_dir = flow_dir[ny, nx]
                if neighbor_dir >= 0:
                    ndy, ndx = D8_DIRECTIONS[neighbor_dir]
                    if ny + ndy == by and nx + ndx == bx:
                        continue  # Would create cycle - skip this neighbor
                
                # Score based on: having flow direction + being downhill
                score = 0
                if flow_dir[ny, nx] >= 0:
                    score += 100  # Strong preference for tiles that already drain
                
                # Slope bonus (positive = downhill)
                slope = (tile_elev - elevation[ny, nx]) / D8_DISTANCES[di]
                score += slope * 10
                
                if score > best_score:
                    best_score = score
                    best_dir = di
            
            if best_dir >= 0:
                flow_dir[by, bx] = best_dir
                is_basin[by, bx] = False  # No longer a basin
                forced_this_pass += 1
        
        if forced_this_pass == 0:
            break
    
    # LAST RESORT: Force any remaining basins to flow to any neighbor at all
    # These are isolated basins that couldn't find a non-cycle path
    basin_points = np.argwhere(is_basin)
    for by, bx in basin_points:
        if flow_dir[by, bx] >= 0:
            continue
        if not is_land[by, bx]:
            continue
        
        # Find ANY neighbor that doesn't create an immediate cycle
        for di, (dy, dx) in enumerate(D8_DIRECTIONS):
            ny, nx = by + dy, bx + dx
            if not (0 <= ny < height and 0 <= nx < width):
                continue
            if not is_land[ny, nx]:
                # Ocean - take it
                flow_dir[by, bx] = di
                is_basin[by, bx] = False
                break
            # Check for cycle
            if flow_dir[ny, nx] >= 0:
                ndy, ndx = D8_DIRECTIONS[flow_dir[ny, nx]]
                if ny + ndy == by and nx + ndx == bx:
                    continue  # Would cycle
            # Accept this neighbor
            flow_dir[by, bx] = di
            is_basin[by, bx] = False
            break
    
    # ABSOLUTE LAST RESORT: For true sinks where ALL neighbors point inward,
    # we need to carve an outlet to the nearest lower/ocean tile
    basin_points = np.argwhere(is_basin & is_land)
    for by, bx in basin_points:
        if flow_dir[by, bx] >= 0:
            continue
        
        # This is a true sink - all neighbors flow into it
        # Find the neighbor with lowest elevation and force flow there
        # Then redirect that neighbor to break the cycle
        min_elev = float('inf')
        best_dir = -1
        best_neighbor = None
        
        for di, (dy, dx) in enumerate(D8_DIRECTIONS):
            ny, nx = by + dy, bx + dx
            if 0 <= ny < height and 0 <= nx < width:
                if not is_land[ny, nx]:
                    # Ocean adjacent! Direct path
                    flow_dir[by, bx] = di
                    is_basin[by, bx] = False
                    best_dir = -2  # Signal found
                    break
                elif elevation[ny, nx] < min_elev:
                    min_elev = elevation[ny, nx]
                    best_dir = di
                    best_neighbor = (ny, nx)
        
        if best_dir == -2:
            continue  # Already handled (ocean)
        
        if best_dir >= 0 and best_neighbor:
            ny, nx = best_neighbor
            # Set basin to flow to this neighbor
            flow_dir[by, bx] = best_dir
            is_basin[by, bx] = False
            
            # Now the neighbor flows back - we need to redirect it
            # Find a secondary outlet for the neighbor (not back to basin)
            basin_elev = elevation[by, bx]
            neighbor_elev = elevation[ny, nx]
            
            # Find tile the neighbor can flow to that isn't back to basin
            new_best_dir = -1
            new_best_score = -9999
            for ndi, (ndy, ndx) in enumerate(D8_DIRECTIONS):
                nny, nnx = ny + ndy, nx + ndx
                if not (0 <= nny < height and 0 <= nnx < width):
                    continue
                if nny == by and nnx == bx:
                    continue  # Skip original basin
                if not is_land[nny, nnx]:
                    # Ocean - great!
                    new_best_dir = ndi
                    break
                # Score by downhill-ness
                slope = neighbor_elev - elevation[nny, nnx]
                if slope > new_best_score:
                    new_best_score = slope
                    new_best_dir = ndi
            
            if new_best_dir >= 0:
                flow_dir[ny, nx] = new_best_dir
    
    # CYCLE BREAKING PASS: Break ALL cycles (any length)
    # Find tiles that don't reach ocean and are in cycles
    for cycle_pass in range(20):  # Multiple passes needed
        cycles_broken = 0
        
        # Find all tiles in cycles by tracing flow paths
        tiles_in_cycle = set()
        for y in range(height):
            for x in range(width):
                if not is_land[y, x]:
                    continue
                if (y, x) in tiles_in_cycle:
                    continue
                
                # Trace path to find if it reaches ocean or cycles
                path = []
                ty, tx = y, x
                path_set = set()
                for _ in range(200):
                    if (ty, tx) in path_set:
                        # Found cycle - mark all tiles from cycle start
                        cycle_start = path.index((ty, tx))
                        for cy, cx in path[cycle_start:]:
                            tiles_in_cycle.add((cy, cx))
                        break
                    if not is_land[ty, tx]:
                        break  # Reached ocean
                    path.append((ty, tx))
                    path_set.add((ty, tx))
                    d = flow_dir[ty, tx]
                    if d < 0:
                        break
                    dy, dx = D8_DIRECTIONS[d]
                    ty, tx = ty + dy, tx + dx
                    if not (0 <= ty < height and 0 <= tx < width):
                        break
        
        if not tiles_in_cycle:
            break  # No more cycles
        
        # Break cycles by redirecting highest-elevation tile in each cycle
        # Group tiles into connected cycles
        processed = set()
        for cy, cx in tiles_in_cycle:
            if (cy, cx) in processed:
                continue
            
            # Find all tiles in this particular cycle
            this_cycle = []
            ty, tx = cy, cx
            for _ in range(100):
                if (ty, tx) in processed:
                    break
                this_cycle.append((ty, tx))
                processed.add((ty, tx))
                d = flow_dir[ty, tx]
                if d < 0:
                    break
                dy, dx = D8_DIRECTIONS[d]
                ty, tx = ty + dy, tx + dx
            
            if not this_cycle:
                continue
            
            # Find highest elevation tile in cycle to redirect
            max_elev = -1
            break_tile = None
            for ty, tx in this_cycle:
                if elevation[ty, tx] > max_elev:
                    max_elev = elevation[ty, tx]
                    break_tile = (ty, tx)
            
            if break_tile:
                by, bx = break_tile
                # Find alternative direction that doesn't point back into cycle
                current_dir = flow_dir[by, bx]
                for alt_di, (alt_dy, alt_dx) in enumerate(D8_DIRECTIONS):
                    if alt_di == current_dir:
                        continue
                    alty, altx = by + alt_dy, bx + alt_dx
                    if not (0 <= alty < height and 0 <= altx < width):
                        continue
                    if not is_land[alty, altx]:
                        # Ocean - take it
                        flow_dir[by, bx] = alt_di
                        cycles_broken += 1
                        break
                    if (alty, altx) in this_cycle:
                        continue  # Still in cycle
                    # Accept if it has outward flow or is downhill
                    if flow_dir[alty, altx] >= 0 or elevation[alty, altx] < elevation[by, bx]:
                        flow_dir[by, bx] = alt_di  
                        cycles_broken += 1
                        break
        
        if cycles_broken == 0:
            break
    
    # Count remaining basins after forcing
    remaining = is_basin.sum()
    
    return elevation, flow_dir, is_basin, total_carves


def check_flow_cycles(flow_dir, is_land, is_lake, width, height, label=""):
    """Debug: Check for 2-tile cycles in flow direction."""
    cycles = 0
    lake_cycles = 0
    shown = 0
    for y in range(height):
        for x in range(width):
            if not is_land[y, x]:
                continue
            dir_idx = flow_dir[y, x]
            if dir_idx < 0:
                continue
            dy, dx = D8_DIRECTIONS[dir_idx]
            ny, nx = y + dy, x + dx
            if not (0 <= ny < height and 0 <= nx < width):
                continue
            if not is_land[ny, nx]:
                continue
            # Check if neighbor points back
            neighbor_dir = flow_dir[ny, nx]
            if neighbor_dir < 0:
                continue
            ndy, ndx = D8_DIRECTIONS[neighbor_dir]
            if ny + ndy == y and nx + ndx == x:
                cycles += 1
                # Check if either tile is a lake
                tile_in_lake = is_lake is not None and is_lake[y, x]
                neighbor_in_lake = is_lake is not None and is_lake[ny, nx]
                if tile_in_lake or neighbor_in_lake:
                    lake_cycles += 1
                # Show first cycle
                if shown < 1:
                    print(f"      Cycle: ({y},{x}) lake={tile_in_lake} dir={dir_idx} <-> ({ny},{nx}) lake={neighbor_in_lake} dir={neighbor_dir}")
                    shown += 1
    if cycles > 0:
        print(f"    {label}: Found {cycles // 2} 2-tile flow cycles ({lake_cycles // 2} involve lakes)!")
    return cycles // 2  # Each cycle counted twice


def handle_lake_outlets(elevation, is_land, is_lake, lake_ids, flow_dir, width, height):
    """
    STEP 8: Process lake tiles for proper drainage.
    
    Lakes act as local basins - water enters and exits through lowest perimeter.
    Updates flow_dir for lake tiles.
    """
    if is_lake is None or not is_lake.any():
        return flow_dir
    
    # Find unique lake IDs
    unique_lake_ids = np.unique(lake_ids[is_lake])
    
    for lake_id in unique_lake_ids:
        if lake_id < 0:
            continue
        
        lake_mask = lake_ids == lake_id
        lake_tiles = np.argwhere(lake_mask)
        
        if len(lake_tiles) == 0:
            continue
        
        # Find lake outlet - lowest perimeter tile
        perimeter = []
        checked = set()
        
        for ly, lx in lake_tiles:
            for dy, dx in D8_DIRECTIONS:
                ny, nx = ly + dy, lx + dx
                if (ny, nx) in checked:
                    continue
                checked.add((ny, nx))
                
                if 0 <= ny < height and 0 <= nx < width:
                    if is_land[ny, nx] and not is_lake[ny, nx]:
                        perimeter.append((elevation[ny, nx], ny, nx, ly, lx))
        
        if not perimeter:
            continue
        
        # Sort by elevation - lowest is outlet
        perimeter.sort()
        _, outlet_y, outlet_x, adj_lake_y, adj_lake_x = perimeter[0]
        
        # Set flow direction for adjacent lake tile toward outlet
        for dir_idx, (dy, dx) in enumerate(D8_DIRECTIONS):
            if adj_lake_y + dy == outlet_y and adj_lake_x + dx == outlet_x:
                # All lake tiles flow toward this outlet direction
                for ly, lx in lake_tiles:
                    flow_dir[ly, lx] = dir_idx
                break
        
        # CRITICAL: Fix cycles - ensure outlet tile doesn't point back into the lake
        # Check if the outlet tile points toward any lake tile
        outlet_dir = flow_dir[outlet_y, outlet_x]
        if outlet_dir >= 0:
            ody, odx = D8_DIRECTIONS[outlet_dir]
            ony, onx = outlet_y + ody, outlet_x + odx
            if 0 <= ony < height and 0 <= onx < width:
                if is_lake[ony, onx]:
                    # Outlet points into lake - need to redirect it!
                    # Find a non-lake neighbor to flow to
                    best_slope = 0
                    best_dir = -1
                    outlet_elev = elevation[outlet_y, outlet_x]
                    for di, (ddy, ddx) in enumerate(D8_DIRECTIONS):
                        nny, nnx = outlet_y + ddy, outlet_x + ddx
                        if not (0 <= nny < height and 0 <= nnx < width):
                            continue
                        if is_lake[nny, nnx]:
                            continue  # Don't flow into lake
                        if not is_land[nny, nnx]:
                            # Ocean - excellent outlet
                            slope = (outlet_elev + 1.0) / D8_DISTANCES[di]
                            if slope > best_slope:
                                best_slope = slope
                                best_dir = di
                        elif elevation[nny, nnx] < outlet_elev:
                            slope = (outlet_elev - elevation[nny, nnx]) / D8_DISTANCES[di]
                            if slope > best_slope:
                                best_slope = slope
                                best_dir = di
                    if best_dir >= 0:
                        flow_dir[outlet_y, outlet_x] = best_dir
                        # print(f"      -> changed to dir {best_dir}")
                    else:
                        # No valid downhill non-lake neighbor - mark as basin?
                        # Or allow same-level flow
                        for di, (ddy, ddx) in enumerate(D8_DIRECTIONS):
                            nny, nnx = outlet_y + ddy, outlet_x + ddx
                            if not (0 <= nny < height and 0 <= nnx < width):
                                continue
                            if is_lake[nny, nnx]:
                                continue
                            if not is_land[nny, nnx]:
                                # Ocean - take it even if uphill
                                flow_dir[outlet_y, outlet_x] = di
                                # print(f"      -> forced to ocean dir {di}")
                                break
                            elif elevation[nny, nnx] <= outlet_elev + 0.01:
                                # Accept flat/slight uphill to non-lake
                                flow_dir[outlet_y, outlet_x] = di
                                # print(f"      -> flat to non-lake dir {di}")
                                break
    
    # After processing all lakes, do a second pass to fix any remaining cycles
    # Find all tiles that point into lakes and redirect them
    for y in range(height):
        for x in range(width):
            if not is_land[y, x] or is_lake[y, x]:
                continue
            dir_idx = flow_dir[y, x]
            if dir_idx < 0:
                continue
            dy, dx = D8_DIRECTIONS[dir_idx]
            ny, nx = y + dy, x + dx
            if not (0 <= ny < height and 0 <= nx < width):
                continue
            if is_lake[ny, nx]:
                # This non-lake tile points into a lake - create cycle with lake flowing out
                # Find alternative direction away from the lake
                tile_elev = elevation[y, x]
                best_slope = 0
                best_dir = -1
                for di, (ddy, ddx) in enumerate(D8_DIRECTIONS):
                    nny, nnx = y + ddy, x + ddx
                    if not (0 <= nny < height and 0 <= nnx < width):
                        continue
                    if is_lake[nny, nnx]:
                        continue  # Don't flow into lake
                    if not is_land[nny, nnx]:
                        # Ocean - excellent
                        slope = (tile_elev + 1.0) / D8_DISTANCES[di]
                        if slope > best_slope:
                            best_slope = slope
                            best_dir = di
                    elif elevation[nny, nnx] < tile_elev:
                        slope = (tile_elev - elevation[nny, nnx]) / D8_DISTANCES[di]
                        if slope > best_slope:
                            best_slope = slope
                            best_dir = di
                if best_dir < 0:
                    # No downhill - allow flat or even slight uphill to escape lake
                    for di, (ddy, ddx) in enumerate(D8_DIRECTIONS):
                        nny, nnx = y + ddy, x + ddx
                        if not (0 <= nny < height and 0 <= nnx < width):
                            continue
                        if is_lake[nny, nnx]:
                            continue
                        if not is_land[nny, nnx]:
                            best_dir = di
                            break
                        elif elevation[nny, nnx] <= tile_elev + 0.02:
                            best_dir = di
                            break
                if best_dir >= 0:
                    flow_dir[y, x] = best_dir
    
    # Third pass: Fix ALL 2-tile cycles (A points to B, B points to A)
    # These prevent drainage and block rivers from reaching ocean
    # Iterate until no more cycles can be fixed (new cycles may form when fixing)
    total_cycles_fixed = 0
    for iteration in range(10):  # Max 10 iterations
        cycles_fixed = 0
        for y in range(height):
            for x in range(width):
                if not is_land[y, x]:
                    continue
                dir_idx = flow_dir[y, x]
                if dir_idx < 0:
                    continue
                dy, dx = D8_DIRECTIONS[dir_idx]
                ny, nx = y + dy, x + dx
                if not (0 <= ny < height and 0 <= nx < width):
                    continue
                if not is_land[ny, nx]:
                    continue  # Flows to ocean - good
                # Check if neighbor points back to us
                neighbor_dir = flow_dir[ny, nx]
                if neighbor_dir < 0:
                    continue
                ndy, ndx = D8_DIRECTIONS[neighbor_dir]
                back_y, back_x = ny + ndy, nx + ndx
                if back_y == y and back_x == x:
                    # Found a 2-tile cycle! Pick the higher elevation tile to redirect
                    if elevation[y, x] >= elevation[ny, nx]:
                        # Redirect (y, x) away from (ny, nx)
                        tile_y, tile_x = y, x
                        avoid_y, avoid_x = ny, nx
                    else:
                        # Redirect (ny, nx) away from (y, x)
                        tile_y, tile_x = ny, nx  
                        avoid_y, avoid_x = y, x
                    
                    tile_elev = elevation[tile_y, tile_x]
                    best_slope = -999
                    best_dir = -1
                    for di, (ddy, ddx) in enumerate(D8_DIRECTIONS):
                        nny, nnx = tile_y + ddy, tile_x + ddx
                        if not (0 <= nny < height and 0 <= nnx < width):
                            continue
                        if nny == avoid_y and nnx == avoid_x:
                            continue  # Don't flow back to cycle partner
                        if not is_land[nny, nnx]:
                            # Ocean - excellent
                            best_dir = di
                            break
                        if is_lake[nny, nnx]:
                            continue  # Avoid lakes to prevent new cycles
                        slope = (tile_elev - elevation[nny, nnx]) / D8_DISTANCES[di]
                        if slope > best_slope:
                            best_slope = slope
                            best_dir = di
                    
                    if best_dir >= 0:
                        flow_dir[tile_y, tile_x] = best_dir
                        cycles_fixed += 1
        
        total_cycles_fixed += cycles_fixed
        if cycles_fixed == 0:
            break
    
    # if total_cycles_fixed > 0:  # DEBUG
    #     print(f"      Fixed {total_cycles_fixed} 2-tile flow cycles")
    
    # Fourth pass: Find and fix any longer cycles by tracing paths
    # A tile that can't reach ocean within 200 steps has a cycle issue
    def trace_to_ocean(y0, x0, max_steps=200):
        """Trace flow from (y0, x0), return True if reaches ocean."""
        visited = set()
        ty, tx = y0, x0
        for _ in range(max_steps):
            if (ty, tx) in visited:
                return False  # Loop detected
            visited.add((ty, tx))
            if not is_land[ty, tx]:
                return True  # Reached ocean
            dir_idx = flow_dir[ty, tx]
            if dir_idx < 0:
                return False  # Basin
            dy, dx = D8_DIRECTIONS[dir_idx]
            ty, tx = ty + dy, tx + dx
            if not (0 <= ty < height and 0 <= tx < width):
                return False  # Out of bounds
        return False  # Too many steps, likely a loop
    
    # Find tiles that should drain but can't reach ocean
    longer_cycles_fixed = 0
    for y in range(height):
        for x in range(width):
            if not is_land[y, x]:
                continue
            if flow_dir[y, x] < 0:
                continue
            if not trace_to_ocean(y, x):
                # This tile can't reach ocean - try to fix it
                tile_elev = elevation[y, x]
                best_dir = -1
                best_score = -999
                
                for di, (dy, dx) in enumerate(D8_DIRECTIONS):
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < height and 0 <= nx < width):
                        continue
                    if not is_land[ny, nx]:
                        # Ocean - best choice
                        best_dir = di
                        best_score = 1000
                        break
                    if is_lake[ny, nx]:
                        continue
                    # Check if neighbor can reach ocean
                    if trace_to_ocean(ny, nx):
                        slope = (tile_elev - elevation[ny, nx]) / D8_DISTANCES[di]
                        score = slope * 100 + 1  # Prefer downhill but accept any draining neighbor
                        if score > best_score:
                            best_score = score
                            best_dir = di
                
                if best_dir >= 0 and best_dir != flow_dir[y, x]:
                    flow_dir[y, x] = best_dir
                    longer_cycles_fixed += 1
    
    # if longer_cycles_fixed > 0:  # DEBUG
    #     print(f"      Fixed {longer_cycles_fixed} tiles stuck in longer cycles")
    
    return flow_dir


def compute_flow_accumulation(flow_dir, is_land, width, height):
    """
    STEP 4: Compute flow accumulation using topological sort.
    
    Each tile contributes 1 unit of water flowing downstream.
    Uses reverse topological ordering for efficiency.
    
    Returns:
    - flow_accum: array with accumulated flow values
    """
    flow_accum = np.ones((height, width), dtype=np.float32)
    flow_accum[~is_land] = 0  # Ocean has no accumulation
    
    # Count upstream tiles for each cell (in-degree)
    in_degree = np.zeros((height, width), dtype=np.int32)
    
    for y in range(height):
        for x in range(width):
            if not is_land[y, x]:
                continue
            
            dir_idx = flow_dir[y, x]
            if dir_idx < 0:
                continue
            
            dy, dx = D8_DIRECTIONS[dir_idx]
            ny, nx = y + dy, x + dx
            
            if 0 <= ny < height and 0 <= nx < width:
                in_degree[ny, nx] += 1
    
    # Find all tiles with no upstream (headwaters)
    queue = deque()
    for y in range(height):
        for x in range(width):
            if is_land[y, x] and in_degree[y, x] == 0:
                queue.append((y, x))
    
    # Process in topological order (headwaters first)
    while queue:
        y, x = queue.popleft()
        
        if not is_land[y, x]:
            continue
        
        dir_idx = flow_dir[y, x]
        if dir_idx < 0:
            continue
        
        dy, dx = D8_DIRECTIONS[dir_idx]
        ny, nx = y + dy, x + dx
        
        if 0 <= ny < height and 0 <= nx < width:
            if is_land[ny, nx]:
                # Pass accumulated flow downstream
                flow_accum[ny, nx] += flow_accum[y, x]
            
            # Decrement in-degree
            in_degree[ny, nx] -= 1
            if in_degree[ny, nx] == 0:
                queue.append((ny, nx))
    
    return flow_accum


def apply_snowmelt_and_springs(flow_accum, elevation, is_land, is_mountain, dist_ocean, width, height):
    """
    Apply snowmelt and spring source boosts to flow accumulation.
    
    Snowmelt: High elevation areas (mountains, highlands) contribute extra flow
    due to seasonal snowmelt. This helps rivers originate in mountains.
    
    Springs: Mid-elevation areas with high moisture potential (closer to coast,
    in valleys) may have natural springs contributing additional water.
    
    Returns modified flow_accum array with boosted values.
    """
    boosted_flow = flow_accum.copy()
    
    # Estimate coldness from elevation (high elevation = cold = snow)
    # Mountains definitely have snow, high elevations likely have snow
    snow_zone = is_mountain.copy()
    
    # Add high elevation land as potential snow zones
    land_elevations = elevation[is_land]
    if len(land_elevations) > 0:
        high_elev_threshold = np.percentile(land_elevations, 75)  # Top 25% elevation
        snow_zone |= (is_land & (elevation >= high_elev_threshold))
    
    snowmelt_count = 0
    spring_count = 0
    
    # Apply snowmelt/elevation boost directly to flow accumulation
    for y in range(height):
        for x in range(width):
            if not is_land[y, x]:
                continue
            
            elev = elevation[y, x]
            
            # Snow zone: strong boost to flow (snowmelt runoff)
            if snow_zone[y, x]:
                boost_factor = SNOWMELT_FLOW_MULTIPLIER
                boosted_flow[y, x] = flow_accum[y, x] * boost_factor
                snowmelt_count += 1
            # Highland zone: moderate boost (partial snowmelt)
            elif elev >= HEADWATER_HIGHLAND_ELEV:
                boost_factor = 1.0 + (SNOWMELT_FLOW_MULTIPLIER - 1.0) * 0.5
                boosted_flow[y, x] = flow_accum[y, x] * boost_factor
    
    # Spring sources: Add bonus flow in mid-elevation, moist areas
    if dist_ocean is not None:
        max_dist = dist_ocean[is_land].max() if is_land.any() else 1
        moist_potential = 1.0 - (dist_ocean / (max_dist + 1))  # Closer to coast = higher
        
        for y in range(height):
            for x in range(width):
                if not is_land[y, x]:
                    continue
                if is_mountain[y, x]:
                    continue  # Mountains don't have springs
                
                elev = elevation[y, x]
                # Springs occur in mid-elevation areas with moisture
                if SPRING_ELEVATION_MAX >= elev >= 0.25:
                    if moist_potential[y, x] >= SPRING_MOISTURE_THRESHOLD:
                        # Add spring bonus
                        boosted_flow[y, x] += SPRING_FLOW_BONUS * moist_potential[y, x]
                        spring_count += 1
    
    if snowmelt_count > 0:
        print(f"    - Snowmelt applied to {snowmelt_count} tiles")
    if spring_count > 0:
        print(f"    - Spring sources at {spring_count} tiles")
    
    return boosted_flow


def find_river_headwaters(flow_accum, flow_dir, elevation, is_land, is_mountain, width, height, threshold=RIVER_FLOW_THRESHOLD, dist_ocean=None):
    """
    Find valid river source points (headwaters) with type classification.
    
    A headwater is a tile where:
    - Flow accumulation exceeds threshold
    - No upstream tile ALSO exceeds threshold (this is where river "starts")
    - Has valid terrain type for river source
    
    Headwater Types:
    - 'mountain': Originates in mountain tile or very high elevation (highest priority)
    - 'highland': Originates near mountains, high ground (high priority)  
    - 'interior_highland': High elevation in deep interior (far from coast)
    - 'hills': Originates in hilly terrain, moderate elevation (medium priority)
    - 'snow_spring': Snowmelt spring in cold/high latitude areas (lower priority)
    
    Rivers then flow THROUGH plains and other terrain on their way to the ocean.
    
    Returns list of (y, x, flow, headwater_type) tuples sorted by flow (highest first).
    """
    headwaters = []
    
    # Calculate elevation percentile thresholds for sources
    land_elevations = elevation[is_land]
    hills_elev_threshold = np.percentile(land_elevations, 40)   # Top 60% - hills territory
    highland_threshold = np.percentile(land_elevations, 65)     # Top 35% - highland
    very_high_threshold = np.percentile(land_elevations, 85)    # Top 15% - near mountain
    
    for y in range(height):
        for x in range(width):
            if not is_land[y, x]:
                continue
            if flow_accum[y, x] < threshold:
                continue
            
            # Check if this could be a headwater (no high-flow upstream)
            has_upstream_river = False
            for dy, dx in D8_DIRECTIONS:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if is_land[ny, nx] and flow_accum[ny, nx] >= threshold:
                        # Check if that tile flows into this one
                        neighbor_dir = flow_dir[ny, nx]
                        if neighbor_dir >= 0:
                            ndy, ndx = D8_DIRECTIONS[neighbor_dir]
                            if ny + ndy == y and nx + ndx == x:
                                has_upstream_river = True
                                break
            
            if not has_upstream_river:
                elev = elevation[y, x]
                
                # Determine headwater type with priority
                headwater_type = None
                
                # === MOUNTAIN HEADWATER (highest priority) ===
                if is_mountain[y, x] or elev >= HEADWATER_MOUNTAIN_ELEV:
                    headwater_type = 'mountain'
                
                # === HIGHLAND HEADWATER (near mountains) ===
                elif elev >= HEADWATER_HIGHLAND_ELEV:
                    # Check for mountains within 2 tiles
                    has_mountain_nearby = False
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < height and 0 <= nx < width:
                                if is_mountain[ny, nx] or elevation[ny, nx] >= very_high_threshold:
                                    has_mountain_nearby = True
                                    break
                        if has_mountain_nearby:
                            break
                    if has_mountain_nearby:
                        headwater_type = 'highland'
                
                # === INTERIOR HIGHLAND (high elevation far from coast, no mountains needed) ===
                if headwater_type is None and dist_ocean is not None:
                    # Deep interior with high elevation can spawn rivers even without mountains
                    # More lenient detection for inland rivers
                    is_interior = dist_ocean[y, x] >= 10  # Closer to interior threshold (was 12)
                    is_elevated = elev >= hills_elev_threshold * 0.9  # Slightly lower elevation OK
                    if is_interior and is_elevated and flow_accum[y, x] >= threshold * 0.7:
                        headwater_type = 'interior_highland'
                
                # === HILLS HEADWATER (moderate elevation, some distance from mountains) ===
                if headwater_type is None and elev >= hills_elev_threshold:
                    # Hills can spawn rivers if they have enough elevation and flow
                    # But require higher flow threshold than mountains
                    if flow_accum[y, x] >= threshold * 1.3:
                        # Check we're in a hilly area (not flat plains)
                        neighbors_elev = []
                        for dy, dx in D8_DIRECTIONS:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < height and 0 <= nx < width and is_land[ny, nx]:
                                neighbors_elev.append(elevation[ny, nx])
                        if neighbors_elev:
                            elev_variance = max(neighbors_elev) - min(neighbors_elev)
                            # Hills have elevation variance (not flat)
                            if elev_variance > 0.03:
                                headwater_type = 'hills'
                
                # === SNOW SPRING (cold regions - high latitude or near snow) ===
                # Estimate cold regions: high Y (north in typical map) - NOT elevation (that's highland)
                if headwater_type is None:
                    # Only latitude-based cold, not elevation (elevation caught by highland)
                    is_cold_latitude = (y < height * 0.25) or (y > height * 0.75)
                    if is_cold_latitude and flow_accum[y, x] >= threshold * 1.2:
                        # Snow springs - snowmelt in cold latitudes
                        headwater_type = 'snow_spring'
                
                # === FLOW FROM MOUNTAINS (rivers starting just below mountains) ===
                if headwater_type is None:
                    for dy, dx in D8_DIRECTIONS:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if is_land[ny, nx]:
                                if is_mountain[ny, nx] or elevation[ny, nx] >= very_high_threshold:
                                    # Verify it actually flows into us
                                    neighbor_dir = flow_dir[ny, nx]
                                    if neighbor_dir >= 0:
                                        ndy, ndx = D8_DIRECTIONS[neighbor_dir]
                                        if ny + ndy == y and nx + ndx == x:
                                            headwater_type = 'highland'
                                            break
                
                # Only include if properly classified
                if headwater_type is not None and flow_accum[y, x] >= RIVER_MIN_HEADWATER_FLOW:
                    headwaters.append((y, x, flow_accum[y, x], headwater_type))
    
    # === INTERIOR RIVER SOURCES ===
    # The flow accumulation model naturally biases toward edges.
    # We need to explicitly add sources from the INTERIOR of the landmass.
    # These will create rivers that flow from center through middle lands to ocean.
    # KEY INSIGHT: Don't rely on dist_ocean - instead find tiles with LONG actual drainage paths
    if dist_ocean is not None:
        # Find tiles with long drainage paths that reach ocean
        interior_candidates = []
        
        # We want sources with paths >= 12 tiles to ocean
        # But for very deep interior (d_ocean >= 20), we accept shorter paths
        min_path_length = 12
        deep_min_path_length = 8  # Shorter requirement for d_ocean >= 20
        
        # NO elevation restriction - we want sources from wherever can drain
        # Sample tiles efficiently (check every 2nd tile for coverage)
        sample_rate = 2
        
        for y in range(0, height, sample_rate):
            for x in range(0, width, sample_rate):
                if not is_land[y, x]:
                    continue
                if is_mountain[y, x]:
                    continue  # Mountains already create headwaters
                    
                # Must have valid flow direction
                if flow_dir[y, x] < 0:
                    continue
                
                # Debug: track a deep interior tile
                debug_tile = (dist_ocean[y, x] >= 25)
                
                # Trace downstream to see how long the path to ocean is
                path_length = 0
                ty, tx = y, x
                visited = set()
                failure_reason = ""
                
                for _ in range(100):  # Max 100 tiles
                    if (ty, tx) in visited:
                        failure_reason = "loop"
                        break
                    visited.add((ty, tx))
                    path_length += 1
                    
                    if not is_land[ty, tx]:  # Reached ocean
                        # This tile has a path to ocean!
                        break
                    
                    dir_idx = flow_dir[ty, tx]
                    if dir_idx < 0:
                        failure_reason = "basin"
                        path_length = 0  # Ends in basin - not useful
                        break
                    
                    dy, dx = D8_DIRECTIONS[dir_idx]
                    ty, tx = ty + dy, tx + dx
                    if not (0 <= ty < height and 0 <= tx < width):
                        failure_reason = "bounds"
                        path_length = 0
                        break
                
                # NOTE: Debug output for deep tiles disabled for cleaner output
                # Uncomment below to debug interior source detection issues
                # if debug_tile and hasattr(find_river_headwaters, 'debug_count'):
                #     ... (deep tile debugging code)
                
                # CRITICAL: Verify path actually reached ocean (not just long)
                # Check if the last tile in the trace is ocean
                reached_ocean = False
                if path_length >= min_path_length:
                    # Walk the path again to check the ending
                    ty, tx = y, x
                    visited = set()
                    for _ in range(path_length + 5):
                        if (ty, tx) in visited:
                            break
                        visited.add((ty, tx))
                        if not is_land[ty, tx]:
                            reached_ocean = True
                            break
                        dir_idx = flow_dir[ty, tx]
                        if dir_idx < 0:
                            break
                        dy, dx = D8_DIRECTIONS[dir_idx]
                        ty, tx = ty + dy, tx + dx
                        if not (0 <= ty < height and 0 <= tx < width):
                            break
                
                if path_length >= min_path_length and reached_ocean:
                    # Found a good interior source!
                    d_ocean = dist_ocean[y, x]
                    score = path_length + d_ocean * 0.5 + elev * 10
                    interior_candidates.append((y, x, score, path_length, d_ocean, elev))
                elif path_length >= deep_min_path_length and reached_ocean and dist_ocean[y, x] >= 20:
                    # Deep interior can use shorter path requirement
                    d_ocean = dist_ocean[y, x]
                    score = path_length + d_ocean * 0.5 + elev * 10
                    interior_candidates.append((y, x, score, path_length, d_ocean, elev))
        
        # Sort by score (highest first = best interior sources)
        # Prioritize by d_ocean first, then path length
        interior_candidates.sort(key=lambda c: (-c[4], -c[3]))  # Sort by d_ocean desc, then path_len desc
        
        # Debug: Show top candidate info (disabled for cleaner output)
        # print(f"    Interior source search: found {len(interior_candidates)} candidates")
        # if len(interior_candidates) > 0:
        #     print(f"    Top 5 candidates:")
        #     for i, (y, x, score, path_len, d_oc, elev) in enumerate(interior_candidates[:5]):
        #         print(f"      ({y},{x}): path_len={path_len}, d_ocean={d_oc:.0f}")
        
        # Add top interior candidates as new headwaters
        existing_headwater_positions = {(h[0], h[1]) for h in headwaters}
        added_interior = 0
        target_interior = 12  # Add up to 12 interior river sources
        
        for y, x, score, path_len, d_ocean, elev in interior_candidates:
            if added_interior >= target_interior:
                break
                
            if (y, x) in existing_headwater_positions:
                continue
                
            # Check spacing from existing headwaters - NO spacing check as these are valuable
            # Just check we're not exactly on an existing headwater
            
            # Path length already verified during candidate search
            # Add as interior source with VERY HIGH priority
            priority_flow = threshold * 5 + path_len * 10  # Prioritize by path length
            headwaters.append((y, x, priority_flow, 'interior_source'))
            existing_headwater_positions.add((y, x))
            added_interior += 1
        
        print(f"    Added {added_interior} interior source headwaters")
    
    # Sort by flow/priority (highest first = main rivers)
    headwaters.sort(key=lambda h: -h[2])
    
    return headwaters


def trace_river_upstream(start_y, start_x, flow_dir, elevation, is_land, is_mountain, width, height, max_length=UPSTREAM_MAX_LENGTH):
    """
    Trace upstream from a point to find the river source.
    
    Follows the steepest uphill path to find where the river originates.
    Stops when:
    - Reaching mountains (good source)
    - Slope becomes too gentle (river source area)
    - Reaching a ridge line (watershed boundary)
    - Max length reached
    
    Returns list of (y, x) tiles forming the upstream path.
    """
    path = []
    y, x = start_y, start_x
    visited = set()
    visited.add((y, x))
    prev_elev = elevation[y, x]
    
    for _ in range(max_length):
        # Find the neighbor that flows INTO this tile with highest elevation
        best_upstream = None
        best_elev = -1
        
        for dy, dx in D8_DIRECTIONS:
            ny, nx = y + dy, x + dx
            if (ny, nx) in visited:
                continue
            if not (0 <= ny < height and 0 <= nx < width):
                continue
            if not is_land[ny, nx]:
                continue
            
            # Check if this neighbor flows into current tile
            neighbor_dir = flow_dir[ny, nx]
            if neighbor_dir >= 0:
                ndy, ndx = D8_DIRECTIONS[neighbor_dir]
                if ny + ndy == y and nx + ndx == x:
                    # This neighbor flows into us
                    if elevation[ny, nx] > best_elev:
                        best_elev = elevation[ny, nx]
                        best_upstream = (ny, nx)
        
        if best_upstream is None:
            break  # No more upstream
        
        y, x = best_upstream
        curr_elev = elevation[y, x]
        
        # Check slope - if too gentle, we've found the source area
        slope = curr_elev - prev_elev
        if UPSTREAM_MIN_SLOPE > 0 and slope < UPSTREAM_MIN_SLOPE and len(path) > 5:
            # Low slope indicates we're in flatter headwater area
            break
        
        # Check for ridge line (if enabled) - local maximum in elevation
        if UPSTREAM_STOP_AT_RIDGE and len(path) > 3:
            is_ridge = True
            for dy, dx in D8_DIRECTIONS:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and is_land[ny, nx]:
                    if elevation[ny, nx] > curr_elev + 0.01:
                        is_ridge = False
                        break
            if is_ridge:
                path.append((y, x))
                break  # Reached ridge - good source
        
        visited.add((y, x))
        path.append((y, x))
        prev_elev = curr_elev
        
        # Stop if we reach mountains (excellent source)
        if is_mountain[y, x]:
            break
    
    return path


def trace_river_to_ocean(start_y, start_x, flow_dir, is_land, width, height, max_length=500, is_lake=None, is_river=None):
    """
    Trace a river from headwater to ocean following flow direction.
    
    When the trace hits a basin (no flow direction), it will attempt to
    continue along an existing river tile that has a valid flow direction.
    This allows tributaries to properly merge with main rivers.
    
    Returns list of (y, x) tiles forming the river path.
    """
    path = []
    y, x = start_y, start_x
    visited = set()
    
    for _ in range(max_length):
        if (y, x) in visited:
            break  # Loop detected
        
        visited.add((y, x))
        path.append((y, x))
        
        # Check if reached ocean
        if not is_land[y, x]:
            break
        
        dir_idx = flow_dir[y, x]
        
        # If no flow direction (basin), try to find adjacent river with valid flow
        if dir_idx < 0:
            # Look for adjacent tile that:
            # 1. Is already a river (is_river not None)
            # 2. Has a valid flow direction
            # 3. Or is adjacent to ocean
            found_continuation = False
            
            if is_river is not None:
                # First priority: find adjacent river tile with valid flow direction
                for di in range(8):
                    dy, dx = D8_DIRECTIONS[di]
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < height and 0 <= nx < width and 
                        (ny, nx) not in visited and
                        is_river[ny, nx] and is_land[ny, nx] and
                        flow_dir[ny, nx] >= 0):
                        y, x = ny, nx
                        found_continuation = True
                        break
                
                # Second priority: find adjacent river tile next to ocean
                if not found_continuation:
                    for di in range(8):
                        dy, dx = D8_DIRECTIONS[di]
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < height and 0 <= nx < width and 
                            (ny, nx) not in visited and
                            is_river[ny, nx] and is_land[ny, nx]):
                            # Check if this river tile is adjacent to ocean
                            for di2 in range(8):
                                dy2, dx2 = D8_DIRECTIONS[di2]
                                oy, ox = ny + dy2, nx + dx2
                                if (0 <= oy < height and 0 <= ox < width and
                                    not is_land[oy, ox]):
                                    y, x = ny, nx
                                    found_continuation = True
                                    break
                            if found_continuation:
                                break
            
            if found_continuation:
                continue
            
            # Check if current tile is adjacent to ocean
            for di in range(8):
                dy, dx = D8_DIRECTIONS[di]
                ny, nx = y + dy, x + dx
                if (0 <= ny < height and 0 <= nx < width and
                    not is_land[ny, nx]):
                    path.append((ny, nx))
                    return path
            
            break  # No flow direction and no continuation found
        
        dy, dx = D8_DIRECTIONS[dir_idx]
        ny, nx = y + dy, x + dx
        
        if not (0 <= ny < height and 0 <= nx < width):
            break
        
        # If next tile is ocean, add it and stop
        if not is_land[ny, nx]:
            path.append((ny, nx))
            break
        
        y, x = ny, nx
    
    return path


def compute_ocean_reachability(is_river, is_land, width, height):
    """
    Compute which river tiles can reach the ocean through connected river tiles.
    
    A river tile reaches ocean if:
    - It is adjacent to an ocean tile, OR
    - It is adjacent to another river tile that reaches ocean
    
    Uses flood-fill propagation from ocean-adjacent river tiles backwards.
    Returns boolean mask where True = this river tile can reach ocean.
    """
    reaches_ocean = np.zeros((height, width), dtype=bool)
    
    # Find all river tiles adjacent to ocean (these definitely reach ocean)
    ocean_adjacent_rivers = []
    for y in range(height):
        for x in range(width):
            if is_river[y, x] and is_land[y, x]:
                # Check if adjacent to ocean
                for di in range(8):
                    dy, dx = D8_DIRECTIONS[di]
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < height and 0 <= nx < width and 
                        not is_land[ny, nx]):  # Ocean tile
                        reaches_ocean[y, x] = True
                        ocean_adjacent_rivers.append((y, x))
                        break
    
    # Propagate backwards through river network
    # Any river tile adjacent to a "reaches ocean" tile also reaches ocean
    from collections import deque
    queue = deque(ocean_adjacent_rivers)
    
    while queue:
        y, x = queue.popleft()
        
        # Check all 8 neighbors
        for di in range(8):
            dy, dx = D8_DIRECTIONS[di]
            ny, nx = y + dy, x + dx
            
            if (0 <= ny < height and 0 <= nx < width and
                is_river[ny, nx] and is_land[ny, nx] and
                not reaches_ocean[ny, nx]):
                reaches_ocean[ny, nx] = True
                queue.append((ny, nx))
    
    return reaches_ocean


def extend_river_to_ocean(path, is_river, is_land, reaches_ocean, flow_dir, width, height, max_extension=200):
    """
    Extend a river path that ends in land by following connected river tiles toward ocean.
    
    Uses flow direction from existing river tiles to navigate downstream.
    Falls back to reaches_ocean mask when flow direction is unavailable.
    """
    if not path:
        return path
    
    last_y, last_x = path[-1]
    
    # If already at ocean, no extension needed
    if not is_land[last_y, last_x]:
        return path
    
    # If this tile doesn't reach ocean, check if adjacent to one that does
    if not reaches_ocean[last_y, last_x]:
        # Check if adjacent to ocean-reaching river
        found = False
        for di in range(8):
            dy, dx = D8_DIRECTIONS[di]
            ny, nx = last_y + dy, last_x + dx
            if (0 <= ny < height and 0 <= nx < width and
                is_river[ny, nx] and reaches_ocean[ny, nx]):
                last_y, last_x = ny, nx
                path = path + [(ny, nx)]
                found = True
                break
        if not found:
            return path  # No ocean-reaching river adjacent
    
    # Now follow river tiles toward ocean using flow direction
    visited = set(path)
    extension = []
    y, x = last_y, last_x
    
    for _ in range(max_extension):
        # Check if we reached ocean (adjacent to ocean tile)
        for di in range(8):
            dy, dx = D8_DIRECTIONS[di]
            ny, nx = y + dy, x + dx
            if (0 <= ny < height and 0 <= nx < width and
                not is_land[ny, nx]):  # Ocean!
                extension.append((ny, nx))
                return path + extension
        
        # Try to follow flow direction first
        dir_idx = flow_dir[y, x]
        if dir_idx >= 0:
            dy, dx = D8_DIRECTIONS[dir_idx]
            ny, nx = y + dy, x + dx
            
            if (0 <= ny < height and 0 <= nx < width and
                (ny, nx) not in visited):
                
                # Check if next tile reaches ocean
                if not is_land[ny, nx]:
                    extension.append((ny, nx))
                    return path + extension
                
                if is_river[ny, nx] and reaches_ocean[ny, nx]:
                    visited.add((ny, nx))
                    extension.append((ny, nx))
                    y, x = ny, nx
                    continue
        
        # Flow direction didn't work - find any adjacent river tile that reaches ocean
        found_next = False
        for di in range(8):
            dy, dx = D8_DIRECTIONS[di]
            ny, nx = y + dy, x + dx
            if (0 <= ny < height and 0 <= nx < width and
                (ny, nx) not in visited and
                is_river[ny, nx] and is_land[ny, nx] and
                reaches_ocean[ny, nx]):
                visited.add((ny, nx))
                extension.append((ny, nx))
                y, x = ny, nx
                found_next = True
                break
        
        if not found_next:
            break  # Dead end
    
    return path + extension


def create_river_from_accumulation(flow_accum, flow_dir, elevation, is_land, is_mountain, width, height, threshold=RIVER_FLOW_THRESHOLD, dist_ocean=None, seed=None, is_lake=None):
    """
    STEP 5: Create rivers by tracing from headwaters to ocean.
    
    Rivers are traced both upstream (to find true source) and downstream
    (to reach ocean), ensuring complete river paths from mountains to sea.
    Lakes are generated AFTER rivers, so rivers form complete systems.
    
    Includes:
    - Headwater type classification (mountain, highland, spring)
    - River hierarchy (stream, river, major river) based on flow accumulation
    - System consolidation to reduce excessive small rivers
    
    Returns:
    - is_river: boolean mask
    - river_map: flow values for river tiles (0 elsewhere)
    - river_paths: list of river paths for ID assignment
    - river_hierarchy: dict mapping (y,x) to hierarchy level ('stream', 'river', 'major')
    - headwater_types: dict mapping headwater (y,x) to type
    """
    is_river = np.zeros((height, width), dtype=bool)
    river_map = np.zeros((height, width), dtype=np.float32)
    river_paths = []
    river_hierarchy = {}
    headwater_types = {}
    
    # Find valid headwaters with type classification
    headwaters = find_river_headwaters(
        flow_accum, flow_dir, elevation, is_land, is_mountain, 
        width, height, threshold, dist_ocean=dist_ocean
    )
    
    # First pass: identify all potential rivers and their lengths
    potential_rivers = []
    
    for hy, hx, flow, hw_type in headwaters:
        # Debug interior_source processing (disabled for cleaner output)
        # if hw_type == 'interior_source':
        #     print(f"    Processing interior_source at ({hy},{hx}), flow={flow}, is_river={is_river[hy, hx]}")
        
        if is_river[hy, hx]:
            continue  # Already part of another river
        
        # For interior_source headwaters, they ARE the true source - no upstream tracing needed
        if hw_type == 'interior_source':
            upstream_path = [(hy, hx)]  # Source IS the headwater
        else:
            # Trace upstream to find true source for flow-accumulation-based headwaters
            upstream_path = trace_river_upstream(
                hy, hx, flow_dir, elevation, is_land, is_mountain, width, height
            )
        
        # Trace downstream to ocean (pass is_river to allow merging with existing rivers)
        downstream_path = trace_river_to_ocean(hy, hx, flow_dir, is_land, width, height, 
                                               is_lake=is_lake, is_river=is_river)
        
        # Combine: upstream (reversed) + downstream
        full_path = list(reversed(upstream_path)) + downstream_path
        
        # Debug for interior_source (disabled for cleaner output)
        # if hw_type == 'interior_source':
        #     print(f"      interior_source ({hy},{hx}): upstream={len(upstream_path)}, downstream={len(downstream_path)}, full={len(full_path)}")
        
        if len(full_path) < RIVER_MIN_LENGTH:
            continue  # Too short
        
        # Check if river reaches ocean or another river
        reaches_destination = False
        for py, px in full_path:
            if not is_land[py, px]:
                reaches_destination = True
                break
        
        if not reaches_destination:
            # Check if it would connect to an existing river
            can_merge = False
            for py, px in full_path:
                if is_river[py, px]:
                    can_merge = True
                    break
            if not can_merge:
                continue  # River goes nowhere
        
        # Calculate max flow in this river path
        max_flow = max(flow_accum[py, px] for py, px in full_path if is_land[py, px])
        
        potential_rivers.append({
            'path': full_path,
            'headwater': (hy, hx),
            'hw_type': hw_type,
            'max_flow': max_flow,
            'length': len(full_path),
            'reaches_ocean': reaches_destination,
            'headwater_dist_ocean': dist_ocean[hy, hx] if dist_ocean is not None else 0
        })
        
        # Temporarily mark as river to prevent overlap
        for py, px in full_path:
            if is_land[py, px]:
                is_river[py, px] = True
    
    # System consolidation: Prioritize longer river networks
    # Filter out very short systems even if they reach ocean
    consolidated_rivers = []
    
    # Debug: interior_source stats (disabled for cleaner output)
    # interior_source_count = sum(1 for r in potential_rivers if r['hw_type'] == 'interior_source')
    # print(f"    Interior source rivers in potential pool: {interior_source_count}")
    
    # Reset river mask for consolidation pass
    is_river.fill(False)
    
    # Sort by a score combining length and max_flow (favor longer, bigger rivers)
    for r in potential_rivers:
        # Heavily reward longer rivers - exponential bonus for length
        length_bonus = r['length'] ** 1.5  # Exponential bonus for longer rivers
        inland_bonus = 1.5 if not r['reaches_ocean'] else 1.0  # Bonus for inland rivers
        # IMPORTANT: Strong bonus for headwaters far from ocean (creates rivers through middle lands)
        dist_ocean_bonus = 1.0 + (r['headwater_dist_ocean'] / 15.0)  # Stronger bonus scales with distance
        # Extra bonus for interior_source type (these are our priority rivers)
        source_type_bonus = 2.0 if r['hw_type'] == 'interior_source' else 1.0
        r['score'] = length_bonus * 2 * inland_bonus * dist_ocean_bonus * source_type_bonus + r['max_flow'] * 0.1
    potential_rivers.sort(key=lambda r: -r['score'])
    
    # Keep track of how many rivers we've accepted and their headwater positions
    accepted_count = 0
    inland_river_count = 0
    target_max_rivers = TARGET_MAX_RIVERS  # Target fewer, longer rivers
    accepted_headwaters = []  # List of (y, x) for spacing check
    
    # Determine how many inland rivers to allow (probability-based)
    rng_inland = np.random.default_rng((seed or 0) + 9999)
    if rng_inland.random() < INLAND_RIVER_CHANCE:
        target_inland_rivers = rng_inland.integers(2, 5)  # 2-4 inland rivers
    else:
        target_inland_rivers = 1  # Always allow at least 1 inland river
    
    def headwater_too_close(hy, hx):
        """Check if a headwater is too close to existing accepted headwaters."""
        for ay, ax in accepted_headwaters:
            dist = np.sqrt((hy - ay)**2 + (hx - ax)**2)
            if dist < MIN_HEADWATER_SPACING:
                return True
        return False
    
    for river_data in potential_rivers:
        path = river_data['path']
        hy, hx = river_data['headwater']
        
        # Check if this river connects to an already-accepted river
        connects_to_larger = False
        for py, px in path:
            if is_river[py, px]:
                connects_to_larger = True
                break
        
        # For tributaries, check spacing requirement (controlled by TRIBUTARY_SPACING_FACTOR)
        if connects_to_larger:
            # Only accept tributaries if their headwater is somewhat spaced
            too_close = False
            for ay, ax in accepted_headwaters:
                dist = np.sqrt((hy - ay)**2 + (hx - ax)**2)
                if dist < MIN_HEADWATER_SPACING * TRIBUTARY_SPACING_FACTOR:
                    too_close = True
                    break
            if too_close:
                continue  # Skip this tributary - too clustered
            
            # Also limit total tributaries to prevent overcrowding
            tributary_count = len(consolidated_rivers) - accepted_count
            if tributary_count >= MAX_TRIBUTARIES_PER_RIVER * accepted_count:
                continue  # Skip - already have enough tributaries
        else:
            # For standalone rivers, apply strict spacing
            if headwater_too_close(hy, hx):
                continue  # Skip - too close to existing river
            
            # For standalone rivers, apply other strict filtering
            if river_data['length'] < MIN_RIVER_SYSTEM_LENGTH:
                continue  # Skip short standalone rivers
            elif not river_data['reaches_ocean']:
                # Non-ocean-reaching rivers (inland) need special handling
                if inland_river_count >= target_inland_rivers:
                    continue  # Already have enough inland rivers
                if river_data['length'] < MIN_RIVER_SYSTEM_LENGTH * 1.5:
                    continue  # Inland rivers need to be longer (12+ tiles)
                inland_river_count += 1  # Count this as an inland river
            # Limit total number of standalone rivers
            elif accepted_count >= target_max_rivers:
                # Only accept exceptionally long rivers beyond the limit
                if river_data['length'] < MIN_RIVER_SYSTEM_LENGTH * 2:
                    continue
        
        # Accept this river
        consolidated_rivers.append(river_data)
        accepted_headwaters.append((hy, hx))
        if not connects_to_larger:
            accepted_count += 1
        
        # Mark river tiles
        for py, px in path:
            if is_land[py, px]:
                is_river[py, px] = True
                river_map[py, px] = flow_accum[py, px]
        
        # Store headwater type
        headwater_types[(hy, hx)] = river_data['hw_type']
    
    # Debug: interior_source stats (disabled for cleaner output)
    # interior_consolidated = sum(1 for r in consolidated_rivers if r['hw_type'] == 'interior_source')
    # print(f"    Interior source rivers accepted: {interior_consolidated}")
    
    # POST-PROCESSING: Extend rivers that end in land but are connected to ocean-reaching rivers
    # Compute which river tiles can reach ocean through connected river network
    reaches_ocean = compute_ocean_reachability(is_river, is_land, width, height)
    
    # Extend each river path that doesn't reach ocean
    extended_count = 0
    for river_data in consolidated_rivers:
        path = river_data['path']
        
        # Check if river already reaches ocean
        if path and not is_land[path[-1][0], path[-1][1]]:
            continue  # Already at ocean
        
        # Try to extend this river using flow direction
        original_len = len(path)
        extended_path = extend_river_to_ocean(
            path, is_river, is_land, reaches_ocean, flow_dir, width, height
        )
        
        if len(extended_path) > original_len:
            # Update the path in place
            river_data['path'] = extended_path
            extended_count += 1
            
            # Mark new tiles as river
            for py, px in extended_path[original_len:]:
                if 0 <= py < height and 0 <= px < width and is_land[py, px]:
                    is_river[py, px] = True
                    river_map[py, px] = flow_accum[py, px]
    
    if extended_count > 0:
        print(f"    Extended {extended_count} rivers to reach ocean")
    
    # Build river paths list and assign hierarchy
    for river_data in consolidated_rivers:
        path = river_data['path']
        river_paths.append(path)
        
        # Assign hierarchy based on flow accumulation at each tile
        for py, px in path:
            if not is_land[py, px]:
                continue
            
            flow = flow_accum[py, px]
            
            if flow >= RIVER_MAJOR_RIVER_THRESHOLD:
                hierarchy = 'major'
            elif flow >= RIVER_RIVER_THRESHOLD:
                hierarchy = 'river'
            else:
                hierarchy = 'stream'
            
            # Upgrade if already assigned a higher level
            current = river_hierarchy.get((py, px))
            if current is None:
                river_hierarchy[(py, px)] = hierarchy
            elif current == 'stream' and hierarchy in ('river', 'major'):
                river_hierarchy[(py, px)] = hierarchy
            elif current == 'river' and hierarchy == 'major':
                river_hierarchy[(py, px)] = hierarchy
    
    # Calculate moisture intensity for each river tile
    # Intensity decreases as river flows downstream (1.0 at source, 0.2 near ocean)
    river_moisture_intensity = np.zeros((height, width), dtype=np.float32)
    
    for river_data in consolidated_rivers:
        path = river_data['path']
        path_length = len(path)
        
        if path_length <= 1:
            continue
        
        for i, (py, px) in enumerate(path):
            if not is_land[py, px]:
                continue
            
            # Position along river: 0.0 at source, 1.0 at end
            position = i / (path_length - 1)
            
            # Intensity: 1.0 at source (mountains), 0.2 at end (plains/ocean)
            # This ensures rivers in plains emit much less moisture
            intensity = 1.0 - (position * 0.8)  # Range: 1.0 to 0.2
            
            # Keep the maximum intensity if tile is part of multiple rivers
            if river_moisture_intensity[py, px] < intensity:
                river_moisture_intensity[py, px] = intensity
    
    return is_river, river_map, river_paths, river_hierarchy, headwater_types, river_moisture_intensity


def compute_river_width(flow_accum, is_river, width, height):
    """
    STEP 6: Compute river width based on flow accumulation.
    
    width = base_width + log(flow_accum) * scale
    Clamped to [1, max_width].
    """
    river_width = np.zeros((height, width), dtype=np.float32)
    
    river_tiles = is_river & (flow_accum > 0)
    
    river_width[river_tiles] = (
        RIVER_BASE_WIDTH + 
        np.log1p(flow_accum[river_tiles]) * RIVER_WIDTH_LOG_SCALE
    )
    
    river_width = np.clip(river_width, 0, RIVER_MAX_WIDTH)
    
    return river_width


def assign_river_ids(is_river, flow_dir, flow_accum, is_land, width, height):
    """
    Assign unique IDs to river segments.
    
    Traces from headwaters downstream, assigning IDs to connected segments.
    Returns:
    - river_ids: array with river ID for each tile (-1 for non-river)
    - num_rivers: count of distinct river systems
    """
    river_ids = np.full((height, width), -1, dtype=np.int32)
    
    # Find river headwater points (river tiles with no upstream river)
    headwaters = []
    
    for y in range(height):
        for x in range(width):
            if not is_river[y, x]:
                continue
            
            # Check if any upstream neighbor is also a river
            has_upstream_river = False
            for dir_idx, (dy, dx) in enumerate(D8_DIRECTIONS):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if is_river[ny, nx]:
                        # Check if that neighbor flows into this tile
                        neighbor_dir = flow_dir[ny, nx]
                        if neighbor_dir >= 0:
                            ndy, ndx = D8_DIRECTIONS[neighbor_dir]
                            if ny + ndy == y and nx + ndx == x:
                                has_upstream_river = True
                                break
            
            if not has_upstream_river:
                headwaters.append((flow_accum[y, x], y, x))
    
    # Sort headwaters by flow (larger first for main rivers)
    headwaters.sort(reverse=True)
    
    river_id = 0
    
    for _, hy, hx in headwaters:
        if river_ids[hy, hx] >= 0:
            continue  # Already assigned
        
        # Trace downstream from this headwater
        y, x = hy, hx
        while True:
            if not (0 <= y < height and 0 <= x < width):
                break
            if not is_river[y, x]:
                break
            if river_ids[y, x] >= 0:
                break  # Merged into existing river
            
            river_ids[y, x] = river_id
            
            dir_idx = flow_dir[y, x]
            if dir_idx < 0:
                break
            
            dy, dx = D8_DIRECTIONS[dir_idx]
            y, x = y + dy, x + dx
        
        river_id += 1
    
    return river_ids, river_id


def validate_river_network(is_river, river_ids, flow_dir, flow_accum, is_land, width, height):
    """
    STEP 11: Validate river network properties.
    
    Checks:
    - No loops
    - Accumulation increases downstream
    - Rivers reach ocean or larger rivers
    
    Returns statistics dictionary.
    """
    stats = {
        'total_tiles': int(is_river.sum()),
        'num_rivers': int(river_ids.max() + 1) if river_ids.max() >= 0 else 0,
        'loops_detected': 0,
        'decreasing_flow': 0,
        'rivers_to_ocean': 0,
        'rivers_to_inland': 0,
        'max_flow': float(flow_accum[is_river].max()) if is_river.any() else 0,
    }
    
    # Find river endpoints
    endpoints = []
    for y in range(height):
        for x in range(width):
            if not is_river[y, x]:
                continue
            
            dir_idx = flow_dir[y, x]
            if dir_idx < 0:
                endpoints.append((y, x))
                continue
            
            dy, dx = D8_DIRECTIONS[dir_idx]
            ny, nx = y + dy, x + dx
            
            # Check if flows to ocean
            if not (0 <= ny < height and 0 <= nx < width):
                endpoints.append((y, x))
            elif not is_land[ny, nx]:
                endpoints.append((y, x))
            elif not is_river[ny, nx]:
                endpoints.append((y, x))
    
    # Check each endpoint
    for ey, ex in endpoints:
        dir_idx = flow_dir[ey, ex]
        if dir_idx >= 0:
            dy, dx = D8_DIRECTIONS[dir_idx]
            ny, nx = ey + dy, ex + dx
            if 0 <= ny < height and 0 <= nx < width:
                if not is_land[ny, nx]:
                    stats['rivers_to_ocean'] += 1
                    continue
        
        stats['rivers_to_inland'] += 1
    
    # Check for flow decreasing downstream (should not happen)
    for y in range(height):
        for x in range(width):
            if not is_river[y, x]:
                continue
            
            dir_idx = flow_dir[y, x]
            if dir_idx < 0:
                continue
            
            dy, dx = D8_DIRECTIONS[dir_idx]
            ny, nx = y + dy, x + dx
            
            if 0 <= ny < height and 0 <= nx < width:
                if is_river[ny, nx] and flow_accum[ny, nx] < flow_accum[y, x]:
                    stats['decreasing_flow'] += 1
    
    return stats


def count_river_segments(river_ids, is_river):
    """Count distinct river segments by length."""
    segment_lengths = []
    
    if river_ids.max() < 0:
        return segment_lengths
    
    for rid in range(river_ids.max() + 1):
        length = int((river_ids == rid).sum())
        if length >= RIVER_MIN_LENGTH:
            segment_lengths.append(length)
    
    return sorted(segment_lengths, reverse=True)


def apply_watershed_climate_effects(river_map, is_river, moisture, temperature, is_land, elevation, width, height, river_hierarchy=None, river_moisture_intensity=None):
    """
    STEP 9: Apply river-based climate effects.
    
    Each river tile has a moisture_intensity (1.0 at source, 0.2 near ocean).
    Moisture boost from rivers is scaled by the intensity of the nearest river tile.
    This allows rivers to flow through plains without converting them to forest.
    
    Lakes still provide full moisture (handled separately).
    
    Args:
        river_hierarchy: Optional dict mapping (y,x) to 'stream', 'river', or 'major'
        river_moisture_intensity: Array with moisture emission intensity per river tile
    """
    if not is_river.any():
        return moisture, temperature
    
    # Calculate distance from rivers and find nearest river for each tile
    dist_from_river = ndimage.distance_transform_edt(~is_river)
    
    # For each tile, find the nearest river tile using indices
    # This lets us look up the moisture intensity of the nearest river
    _, nearest_river_indices = ndimage.distance_transform_edt(~is_river, return_indices=True)
    
    # Base moisture boost by distance
    moisture_boost = np.zeros_like(moisture)
    
    # Distance 0-1: strong boost
    mask_0_1 = dist_from_river < 1.5
    moisture_boost[mask_0_1] = RIVER_MOISTURE_DIST_0_1
    
    # Distance 2-3: moderate boost
    mask_2_3 = (dist_from_river >= 1.5) & (dist_from_river < 3.5)
    moisture_boost[mask_2_3] = RIVER_MOISTURE_DIST_2_3
    
    # Distance 4-6: weak boost
    mask_4_6 = (dist_from_river >= 3.5) & (dist_from_river < 6.5)
    moisture_boost[mask_4_6] = RIVER_MOISTURE_DIST_4_6
    
    # Distance 7-8: minimal boost
    mask_7_8 = (dist_from_river >= 6.5) & (dist_from_river < 8.5)
    moisture_boost[mask_7_8] = RIVER_MOISTURE_DIST_7_8
    
    # Scale moisture boost by the intensity of the nearest river tile
    # Rivers near source (mountains) emit full moisture → forests grow
    # Rivers near end (plains) emit minimal moisture → plains stay plains
    if river_moisture_intensity is not None:
        # Get intensity from nearest river tile for each position
        nearest_y = nearest_river_indices[0]
        nearest_x = nearest_river_indices[1]
        intensity_at_nearest = river_moisture_intensity[nearest_y, nearest_x]
        
        # Scale moisture boost by intensity (only where there's some boost)
        has_boost = moisture_boost > 0
        moisture_boost[has_boost] *= intensity_at_nearest[has_boost]
    
    # Enhanced effects for major rivers (if hierarchy provided)
    # Major rivers still get extra boost but also scaled by intensity
    if river_hierarchy is not None:
        # Create major river mask
        is_major = np.zeros((height, width), dtype=bool)
        for (y, x), h_type in river_hierarchy.items():
            if h_type == 'major':
                is_major[y, x] = True
        
        if is_major.any():
            # Calculate distance from major rivers specifically
            dist_from_major = ndimage.distance_transform_edt(~is_major)
            
            # Add extra moisture boost near major rivers (extended range)
            major_boost = np.zeros_like(moisture)
            major_mask = dist_from_major < MAJOR_RIVER_INFLUENCE_DIST
            # Gradual decay with distance
            major_boost[major_mask] = MAJOR_RIVER_MOISTURE_BOOST * (
                1.0 - dist_from_major[major_mask] / MAJOR_RIVER_INFLUENCE_DIST
            )
            
            # Scale major river boost by intensity too
            if river_moisture_intensity is not None:
                _, major_nearest_indices = ndimage.distance_transform_edt(~is_major, return_indices=True)
                major_nearest_y = major_nearest_indices[0]
                major_nearest_x = major_nearest_indices[1]
                major_intensity = river_moisture_intensity[major_nearest_y, major_nearest_x]
                major_has_boost = major_boost > 0
                major_boost[major_has_boost] *= major_intensity[major_has_boost]
            
            moisture_boost += major_boost
    
    # Apply to land only
    moisture[is_land] = np.clip(moisture[is_land] + moisture_boost[is_land], 0, 1)
    
    # Temperature reduction (within 4 tiles) - not affected by intensity
    temp_reduction = np.zeros_like(temperature)
    temp_mask = dist_from_river < 4.5
    temp_reduction[temp_mask] = RIVER_TEMP_REDUCTION * (1.0 - dist_from_river[temp_mask] / 4.5)
    
    temperature[is_land] = np.clip(temperature[is_land] - temp_reduction[is_land], 0, 1)
    
    return moisture, temperature


def generate_rivers(width, height, seed, elevation, is_land, is_mountain, is_lake, 
                    lake_ids, dist_ocean):
    """
    Main watershed-based river generation function.
    
    Generates rivers using flow accumulation across the heightmap.
    Rivers emerge naturally from water drainage patterns.
    
    Features:
    - Snowmelt boost from high elevations
    - Spring sources in moist areas
    - Headwater type classification (mountain, highland, spring)
    - River hierarchy (stream, river, major river)
    - System consolidation for cleaner networks
    
    Returns:
    - is_river: boolean mask of river tiles
    - river_ids: array with river ID for each tile (-1 for non-river)
    - river_map: flow accumulation values for rivers
    - river_width: width of rivers based on flow
    - river_stats: statistics dictionary
    - elevation: modified by basin carving
    - river_hierarchy: dict mapping (y,x) to hierarchy level
    - headwater_types: dict mapping headwater (y,x) to type
    """
    print("    Computing D8 flow directions...")
    flow_dir, is_basin = compute_flow_direction_field(
        elevation, is_land, is_lake, width, height
    )
    basin_count = is_basin.sum()
    print(f"    - Initial basins found: {basin_count}")
    
    print("    Resolving basins with overflow carving...")
    elevation, flow_dir, is_basin, total_carves = resolve_all_basins(
        elevation, is_land, is_lake, flow_dir, is_basin, width, height
    )
    remaining_basins = is_basin.sum()
    print(f"    - Terrain carves: {total_carves}")
    print(f"    - Remaining basins: {remaining_basins}")
    # check_flow_cycles(flow_dir, is_land, is_lake, width, height, "After basin resolution")  # DEBUG
    
    print("    Processing lake outlets...")
    flow_dir = handle_lake_outlets(
        elevation, is_land, is_lake, lake_ids, flow_dir, width, height
    )
    # check_flow_cycles(flow_dir, is_land, is_lake, width, height, "After lake outlets")  # DEBUG
    
    # Debug: Check drainage at different distances (disabled for cleaner output)
    # for dist_threshold in [15, 20, 25, 30]:
    #     deep_interior = (dist_ocean >= dist_threshold) & is_land
    #     has_flow = deep_interior & (flow_dir >= 0)
    #     total = deep_interior.sum()
    #     draining = has_flow.sum()
    #     print(f"    Tiles at distance >={dist_threshold}: {total} total, {draining} draining ({100*draining/max(1,total):.0f}%)")
    
    print("    Computing flow accumulation...")
    flow_accum = compute_flow_accumulation(flow_dir, is_land, width, height)
    base_max_flow = flow_accum.max()
    print(f"    - Base flow accumulation max: {base_max_flow:.0f}")
    
    print("    Applying snowmelt and spring boosts...")
    flow_accum = apply_snowmelt_and_springs(
        flow_accum, elevation, is_land, is_mountain, dist_ocean, width, height
    )
    max_flow = flow_accum.max()
    print(f"    - Boosted flow accumulation max: {max_flow:.0f}")
    
    print("    Tracing rivers from headwaters to ocean...")
    is_river, river_map, river_paths, river_hierarchy, headwater_types, river_moisture_intensity = create_river_from_accumulation(
        flow_accum, flow_dir, elevation, is_land, is_mountain, width, height, 
        RIVER_FLOW_THRESHOLD, dist_ocean=dist_ocean, seed=seed, is_lake=is_lake
    )
    river_tiles = is_river.sum()
    print(f"    - River tiles: {river_tiles}")
    print(f"    - River systems: {len(river_paths)}")
    
    # Count hierarchy types
    hierarchy_counts = {'stream': 0, 'river': 0, 'major': 0}
    for h in river_hierarchy.values():
        hierarchy_counts[h] += 1
    print(f"    - Hierarchy: {hierarchy_counts['stream']} streams, {hierarchy_counts['river']} rivers, {hierarchy_counts['major']} major")
    
    # Count headwater types
    hw_type_counts = {}
    for hw_type in headwater_types.values():
        hw_type_counts[hw_type] = hw_type_counts.get(hw_type, 0) + 1
    print(f"    - Headwaters: {hw_type_counts}")
    
    print("    Computing river widths...")
    river_width = compute_river_width(flow_accum, is_river, width, height)
    
    print("    Assigning river IDs...")
    river_ids, num_rivers = assign_river_ids(
        is_river, flow_dir, flow_accum, is_land, width, height
    )
    print(f"    - River segments: {num_rivers}")
    
    print("    Validating river network...")
    river_stats = validate_river_network(
        is_river, river_ids, flow_dir, flow_accum, is_land, width, height
    )
    
    # Add additional stats
    river_stats['terrain_carves'] = total_carves
    river_stats['river_lengths'] = sorted([len(p) for p in river_paths], reverse=True)
    river_stats['num_rivers'] = len(river_paths)
    river_stats['avg_width'] = float(river_width[is_river].mean()) if is_river.any() else 0
    river_stats['merged_rivers'] = 0  # Watershed model naturally handles merging
    river_stats['valid_rivers'] = river_stats['rivers_to_ocean']
    river_stats['hierarchy_counts'] = hierarchy_counts
    river_stats['headwater_types'] = hw_type_counts
    
    # Analyze river distance from ocean
    if dist_ocean is not None:
        river_dists = dist_ocean[is_river & is_land]
        if len(river_dists) > 0:
            print(f"    River distance from ocean: max={river_dists.max():.0f}, median={np.median(river_dists):.0f}")
            d10 = (river_dists < 10).sum()
            d20 = ((river_dists >= 10) & (river_dists < 20)).sum()
            d30 = ((river_dists >= 20) & (river_dists < 30)).sum()
            d40 = (river_dists >= 30).sum()
            print(f"    River tiles by distance: <10:{d10}, 10-20:{d20}, 20-30:{d30}, 30+:{d40}")
    
    return is_river, river_ids, river_map, river_width, river_stats, elevation, river_hierarchy, headwater_types, river_moisture_intensity


# =============================================================================
# LAKE GENERATION - STRICT PROBABILISTIC CONSTRAINTS
# =============================================================================
def detect_basins_for_lakes(elevation, is_land, is_mountain, width, height, seed=0):
    """
    Detect elevation basins suitable for lake formation.
    A basin is a tile lower than its surrounding area.
    Prefers MODERATE basins over extreme ones to avoid competing with rivers.
    Returns scored basin candidates.
    """
    basin_candidates = []
    rng = np.random.default_rng(seed + 7777)
    
    for y in range(2, height - 2):
        for x in range(2, width - 2):
            if not is_land[y, x] or is_mountain[y, x]:
                continue
            
            elev = elevation[y, x]
            neighbors_elev = []
            
            # Check larger neighborhood for basin detection
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if is_land[ny, nx]:
                            neighbors_elev.append(elevation[ny, nx])
            
            if neighbors_elev:
                avg_neighbor = np.mean(neighbors_elev)
                min_neighbor = min(neighbors_elev)
                max_neighbor = max(neighbors_elev)
                
                # Basin score: how much lower than average neighbors
                basin_depth = avg_neighbor - elev
                
                # CHANGE: Prefer moderate basins over extreme ones
                # Deep basins (depth > 0.08) are where rivers preferentially flow
                # We want lakes in gentler depressions
                if basin_depth > 0.08:
                    # Penalize very deep basins - rivers will take these
                    basin_score = 0.08 - (basin_depth - 0.08) * 0.5
                else:
                    basin_score = basin_depth
                
                # Small bonus if this is a local minimum
                if elev <= min_neighbor:
                    basin_score += 0.02
                
                # Bonus for flatter terrain (low elevation variance = good lake spot)
                elev_range = max_neighbor - min_neighbor
                if elev_range < 0.05:  # Relatively flat area
                    basin_score += 0.03
                
                # Add randomness to break up clustering (±20%)
                basin_score *= (0.8 + rng.random() * 0.4)
                
                if basin_score > 0:  # Must be lower than surroundings
                    basin_candidates.append((basin_score, elev, y, x))
    
    basin_candidates.sort(key=lambda t: (-t[0], t[1]))
    return basin_candidates


def enforce_coast_buffer_for_lakes(candidates, distance_from_coast, buffer_distance):
    """
    Filter lake candidates to enforce coast buffer.
    Lakes cannot be within buffer_distance tiles of coastline.
    """
    filtered = []
    for score, elev, y, x in candidates:
        if distance_from_coast[y, x] >= buffer_distance:
            filtered.append((score, elev, y, x))
    return filtered


def cluster_lake_tiles(lake_tiles, lake_id_map, lake_id, min_size):
    """
    Ensure lake tiles form contiguous clusters.
    Remove any isolated tiles and ensure minimum size.
    """
    if len(lake_tiles) < min_size:
        return [], lake_id_map
    
    # Create temporary mask
    height, width = lake_id_map.shape
    temp_mask = np.zeros((height, width), dtype=bool)
    for x, y in lake_tiles:
        temp_mask[y, x] = True
    
    # Label connected components
    labeled, num_features = ndimage.label(temp_mask)
    
    if num_features == 0:
        return [], lake_id_map
    
    # Find largest component
    component_sizes = ndimage.sum(temp_mask, labeled, range(1, num_features + 1))
    largest_component = np.argmax(component_sizes) + 1
    largest_size = component_sizes[largest_component - 1]
    
    if largest_size < min_size:
        return [], lake_id_map
    
    # Keep only tiles in largest component
    valid_tiles = []
    for x, y in lake_tiles:
        if labeled[y, x] == largest_component:
            valid_tiles.append((x, y))
            lake_id_map[y, x] = lake_id
    
    return valid_tiles, lake_id_map


def determine_desert_lake_count(seed):
    """
    Determine max number of lakes allowed IN DESERT using probabilistic system:
    - 70% chance: 0 lakes in desert
    - 20% chance: 1 lake in desert
    - 10% chance: 2 lakes in desert
    
    Returns deterministic count based on seed.
    """
    rng = np.random.default_rng(seed + 4500)
    roll = rng.random()
    
    if roll < 0.70:
        return 0  # 70% chance
    elif roll < 0.90:
        return 1  # 20% chance (0.70 to 0.90)
    else:
        return 2  # 10% chance (0.90 to 1.00)


def generate_lakes(width, height, seed, elevation, is_land, is_mountain, distance_from_ocean):
    """
    Generate lakes targeting 5-9% of land coverage.
    - Creates multiple lakes across the map
    - Lakes must be 15 tiles from coastline
    - No single-tile lakes (minimum 4 tiles)
    - Rivers flow through lakes via spill outlet
    """
    rng = np.random.default_rng(seed + 4000)
    
    land_count = is_land.sum()
    min_lake_tiles = int(land_count * LAKE_PERCENT_MIN)  # Min 5% of land
    max_lake_tiles = int(land_count * LAKE_PERCENT_MAX)  # Max 9% of land
    target_lake_tiles = rng.integers(min_lake_tiles, max_lake_tiles + 1)
    
    is_lake = np.zeros((height, width), dtype=bool)
    lake_ids = np.full((height, width), -1, dtype=np.int32)
    
    # Detect basins - now with randomness to spread lakes away from river bottoms
    basin_candidates = detect_basins_for_lakes(elevation, is_land, is_mountain, width, height, seed)
    
    # Enforce coast buffer (15 tiles)
    basin_candidates = enforce_coast_buffer_for_lakes(basin_candidates, distance_from_ocean, LAKE_COAST_BUFFER)
    
    if not basin_candidates:
        # Fallback: use any valid land tiles far from coast
        for y in range(height):
            for x in range(width):
                if is_land[y, x] and not is_mountain[y, x]:
                    if distance_from_ocean[y, x] >= LAKE_COAST_BUFFER:
                        basin_candidates.append((0.0, elevation[y, x], y, x))
    
    if not basin_candidates:
        # No valid locations for lakes
        return is_lake, lake_ids, []
    
    # Sort basins by depth (best basins first)
    basin_candidates.sort(key=lambda x: (x[0], x[1]))
    
    # Pre-filter to only valid starting points
    valid_starts = [(sy, sx) for _, _, sy, sx in basin_candidates 
                    if distance_from_ocean[sy, sx] >= LAKE_COAST_BUFFER]
    rng.shuffle(valid_starts)
    
    lake_id = 0
    lake_tiles_placed = 0
    lake_sizes = []
    lake_centers = []  # Track lake center positions for spacing
    max_lakes = 45  # Reduced cap to produce fewer lakes overall
    start_idx = 0
    
    def too_close_to_existing_lake(y, x):
        """Check if position is too close to existing lake centers."""
        for ly, lx in lake_centers:
            dist = np.sqrt((y - ly)**2 + (x - lx)**2)
            if dist < MIN_LAKE_SPACING:
                return True
        return False
    
    while lake_tiles_placed < target_lake_tiles and lake_id < max_lakes and start_idx < len(valid_starts):
        # Find next available starting point
        start_y, start_x = None, None
        
        while start_idx < len(valid_starts):
            sy, sx = valid_starts[start_idx]
            start_idx += 1
            if not is_lake[sy, sx] and not too_close_to_existing_lake(sy, sx):
                start_y, start_x = sy, sx
                break
        
        if start_y is None:
            break
        
        # Determine lake size - smaller individual lakes
        remaining = target_lake_tiles - lake_tiles_placed
        avg_size = max(20, remaining // max(1, max_lakes - lake_id))  # Smaller average
        lake_size = min(
            rng.integers(MIN_LAKE_SIZE, min(MAX_LAKE_SIZE, avg_size * 2) + 1),
            remaining
        )
        
        if lake_size < MIN_LAKE_SIZE:
            break
        
        # Grow the lake
        raw_tiles = grow_lake_with_constraints(start_x, start_y, lake_size, elevation,
                                                is_land, is_mountain, is_lake,
                                                distance_from_ocean, width, height)
        
        # Ensure contiguous cluster and minimum size
        valid_tiles, lake_ids = cluster_lake_tiles(raw_tiles, lake_ids, lake_id, MIN_LAKE_SIZE)
        
        if len(valid_tiles) >= MIN_LAKE_SIZE:
            for lx, ly in valid_tiles:
                is_lake[ly, lx] = True
                lake_tiles_placed += 1
            lake_sizes.append(len(valid_tiles))
            lake_centers.append((start_y, start_x))  # Track this lake's center
            lake_id += 1
    
    return is_lake, lake_ids, lake_sizes


def shrink_river_connected_lakes(is_lake, lake_ids, is_river, width, height, max_river_lake_width=2):
    """
    Shrink lakes that are connected to rivers to create widened river sections.
    
    Lakes that don't touch any rivers are left unchanged.
    Lakes that touch rivers are shrunk to only tiles within max_river_lake_width 
    of the river, creating a "thick river" effect instead of a full lake.
    
    Args:
        is_lake: Boolean array of lake tiles
        lake_ids: Array of lake IDs
        is_river: Boolean array of river tiles
        width, height: Map dimensions
        max_river_lake_width: Max distance from river for shrunk lake tiles (default 2)
    
    Returns:
        Updated is_lake, lake_ids arrays
    """
    # Find unique lakes
    unique_lakes = set(lake_ids[is_lake])
    unique_lakes.discard(-1)
    
    # Calculate distance from river for all tiles
    dist_from_river = ndimage.distance_transform_edt(~is_river)
    
    lakes_shrunk = 0
    tiles_removed = 0
    
    for lid in unique_lakes:
        # Get all tiles of this lake
        lake_mask = lake_ids == lid
        lake_tiles = list(zip(*np.where(lake_mask)))
        
        if not lake_tiles:
            continue
        
        # Check if this lake touches any river
        touches_river = False
        for y, x in lake_tiles:
            # Check if any adjacent tile is a river
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if is_river[ny, nx]:
                            touches_river = True
                            break
                if touches_river:
                    break
            if touches_river:
                break
        
        # If lake doesn't touch river, leave it unchanged
        if not touches_river:
            continue
        
        # Lake touches river - shrink it to only tiles close to river
        lakes_shrunk += 1
        
        for y, x in lake_tiles:
            # Keep tile only if within max_river_lake_width of river
            if dist_from_river[y, x] > max_river_lake_width:
                is_lake[y, x] = False
                lake_ids[y, x] = -1
                tiles_removed += 1
    
    return is_lake, lake_ids, lakes_shrunk, tiles_removed


def grow_lake_with_constraints(start_x, start_y, target_size, elevation, is_land, 
                                is_mountain, is_lake, distance_from_ocean, width, height):
    """Grow a lake respecting constraints."""
    lake_tiles = [(start_x, start_y)]
    frontier = [(start_x, start_y)]
    visited = {(start_x, start_y)}
    
    while len(lake_tiles) < target_size and frontier:
        frontier_with_elev = [(elevation[y, x], x, y) for x, y in frontier]
        frontier_with_elev.sort()
        
        _, cx, cy = frontier_with_elev[0]
        frontier.remove((cx, cy))
        
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < width and 0 <= ny < height:
                if (nx, ny) not in visited:
                    if (is_land[ny, nx] and not is_mountain[ny, nx] and 
                        not is_lake[ny, nx] and distance_from_ocean[ny, nx] >= LAKE_COAST_BUFFER):
                        visited.add((nx, ny))
                        lake_tiles.append((nx, ny))
                        frontier.append((nx, ny))
                        if len(lake_tiles) >= target_size:
                            break
    
    return lake_tiles


# =============================================================================
# ISLAND GENERATION
# =============================================================================
def generate_islands(width, height, seed, is_land, elevation):
    """Generate islands in ocean."""
    rng = np.random.default_rng(seed + 5000)
    py_random = random.Random(seed + 5000)
    
    ocean_count = (~is_land).sum()
    is_island = np.zeros((height, width), dtype=bool)
    
    roll = py_random.random()
    if roll < 0.40:
        return is_island
    elif roll < 0.80:
        target_ratio = rng.uniform(0.01, 0.03)
        target_islands = int(ocean_count * target_ratio)
        
        ocean_tiles = list(zip(*np.where(~is_land)))
        rng.shuffle(ocean_tiles)
        
        placed = 0
        i = 0
        while placed < target_islands and i < len(ocean_tiles):
            y, x = ocean_tiles[i]
            i += 1
            
            if is_island[y, x]:
                continue
            
            island_size = rng.integers(6, 16)  # Avoid tiny artifact-like islands
            island_tiles = grow_island(x, y, island_size, is_land, is_island, width, height)
            
            if len(island_tiles) >= 5:  # Keep only meaningful island blobs
                for ix, iy in island_tiles:
                    is_island[iy, ix] = True
                    placed += 1
    else:
        target_ratio = rng.uniform(0.03, 0.05)
        target_islands = int(ocean_count * target_ratio)
        
        start_x = rng.integers(width // 4, 3 * width // 4)
        start_y = rng.integers(height // 4, 3 * height // 4)
        
        ocean_tiles = np.array(list(zip(*np.where(~is_land))))
        if len(ocean_tiles) > 0:
            distances = np.sqrt((ocean_tiles[:, 0] - start_y)**2 + (ocean_tiles[:, 1] - start_x)**2)
            nearest_idx = np.argmin(distances)
            start_y, start_x = ocean_tiles[nearest_idx]
        
        chain_angle = rng.uniform(0, 2 * np.pi)
        chain_length = rng.integers(50, 100)
        
        placed = 0
        for step in range(chain_length):
            if placed >= target_islands:
                break
            
            cx = int(start_x + step * np.cos(chain_angle) * 2 + rng.normal(0, 3))
            cy = int(start_y + step * np.sin(chain_angle) * 2 + rng.normal(0, 3))
            
            if not (0 <= cx < width and 0 <= cy < height):
                continue
            if is_land[cy, cx] or is_island[cy, cx]:
                continue
            
            island_size = rng.integers(8, 22)
            island_tiles = grow_island(cx, cy, island_size, is_land, is_island, width, height)
            
            if len(island_tiles) >= 5:
                for ix, iy in island_tiles:
                    is_island[iy, ix] = True
                    placed += 1
    
    return is_island


def grow_island(start_x, start_y, target_size, is_land, is_island, width, height):
    """Grow an island as an organic contiguous blob (not plus/two-tile artifacts)."""
    if target_size <= 0:
        return []

    if not (0 <= start_x < width and 0 <= start_y < height):
        return []
    if is_land[start_y, start_x] or is_island[start_y, start_x]:
        return []

    rng_seed = (start_x * 92821 + start_y * 68917 + target_size * 1237) & 0xFFFFFFFF
    rng = np.random.default_rng(rng_seed)

    # Use 8-neighborhood for smoother coastlines
    neighbors8 = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    island_set = {(start_x, start_y)}
    frontier = set()

    for dx, dy in neighbors8:
        nx, ny = start_x + dx, start_y + dy
        if 0 <= nx < width and 0 <= ny < height and not is_land[ny, nx] and not is_island[ny, nx]:
            frontier.add((nx, ny))

    # Radius cap keeps shapes compact and avoids long tentacles
    max_radius = max(2.8, np.sqrt(target_size) * 1.35)

    while len(island_set) < target_size and frontier:
        best_tile = None
        best_score = -1e9

        # Evaluate a subset for performance and variability
        candidates = list(frontier)
        if len(candidates) > 80:
            sample_idx = rng.choice(len(candidates), size=80, replace=False)
            candidates = [candidates[i] for i in sample_idx]

        for nx, ny in candidates:
            # Skip points too far from center to prevent stringy islands
            dist_center = np.hypot(nx - start_x, ny - start_y)
            if dist_center > max_radius:
                continue

            # Count adjacent island neighbors (prefer cohesive growth)
            adj_island = 0
            for dx, dy in neighbors8:
                tx, ty = nx + dx, ny + dy
                if (tx, ty) in island_set:
                    adj_island += 1

            if adj_island == 0:
                continue

            # Higher score => more compact, rounded, organic shape
            compactness = adj_island * 1.6
            center_bias = 1.2 * (1.0 - dist_center / (max_radius + 1e-6))
            coastline_noise = rng.normal(0.0, 0.18)
            score = compactness + center_bias + coastline_noise

            if score > best_score:
                best_score = score
                best_tile = (nx, ny)

        if best_tile is None:
            break

        bx, by = best_tile
        island_set.add((bx, by))
        frontier.discard((bx, by))

        for dx, dy in neighbors8:
            nx, ny = bx + dx, by + dy
            if 0 <= nx < width and 0 <= ny < height:
                if not is_land[ny, nx] and not is_island[ny, nx] and (nx, ny) not in island_set:
                    frontier.add((nx, ny))

    if len(island_set) <= 2:
        return list(island_set)

    # Cleanup pass: trim thin spikes and fill tiny holes for more natural shapes
    min_x = max(0, min(x for x, _ in island_set) - 1)
    max_x = min(width - 1, max(x for x, _ in island_set) + 1)
    min_y = max(0, min(y for _, y in island_set) - 1)
    max_y = min(height - 1, max(y for _, y in island_set) + 1)

    for _ in range(2):
        to_remove = set()
        for x, y in island_set:
            ncount = 0
            for dx, dy in neighbors8:
                if (x + dx, y + dy) in island_set:
                    ncount += 1
            if ncount <= 1 and len(island_set) > 5:
                to_remove.add((x, y))
        island_set -= to_remove

        to_add = set()
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if (x, y) in island_set:
                    continue
                if is_land[y, x] or is_island[y, x]:
                    continue
                ncount = 0
                for dx, dy in neighbors8:
                    if (x + dx, y + dy) in island_set:
                        ncount += 1
                if ncount >= 5 and len(island_set) < int(target_size * 1.25):
                    to_add.add((x, y))
        island_set |= to_add

    return list(island_set)


# =============================================================================
# BIOME ASSIGNMENT WITH SCORE-BASED COMPETITION
# =============================================================================

def compute_region_averages(width, height, temperature, moisture, elevation, is_land, is_island):
    """
    Compute average moisture, temperature, and elevation for each 8x8 region.
    Returns region_avg arrays where each tile has the average of its region.
    """
    land_mask = is_land | is_island
    
    region_avg_moisture = np.zeros((height, width))
    region_avg_temperature = np.zeros((height, width))
    region_avg_elevation = np.zeros((height, width))
    
    num_regions_x = width // REGION_SIZE
    num_regions_y = height // REGION_SIZE
    
    for ry in range(num_regions_y):
        for rx in range(num_regions_x):
            # Region bounds
            y_start = ry * REGION_SIZE
            y_end = min((ry + 1) * REGION_SIZE, height)
            x_start = rx * REGION_SIZE
            x_end = min((rx + 1) * REGION_SIZE, width)
            
            # Get land tiles in this region
            region_land = land_mask[y_start:y_end, x_start:x_end]
            
            if region_land.sum() > 0:
                region_moisture = moisture[y_start:y_end, x_start:x_end]
                region_temperature = temperature[y_start:y_end, x_start:x_end]
                region_elevation = elevation[y_start:y_end, x_start:x_end]
                
                avg_moist = region_moisture[region_land].mean()
                avg_temp = region_temperature[region_land].mean()
                avg_elev = region_elevation[region_land].mean()
            else:
                avg_moist = 0.5
                avg_temp = 0.5
                avg_elev = 0.5
            
            # Assign to all tiles in region
            region_avg_moisture[y_start:y_end, x_start:x_end] = avg_moist
            region_avg_temperature[y_start:y_end, x_start:x_end] = avg_temp
            region_avg_elevation[y_start:y_end, x_start:x_end] = avg_elev
    
    return region_avg_moisture, region_avg_temperature, region_avg_elevation


def compute_biome_scores(temp, moist, elev, dist_ocean, dist_river, dist_lake,
                         region_avg_moist, region_avg_temp, region_avg_elev):
    """
    Compute climate-dependent scores for each biome.
    
    Climate factors:
    - Temperature: 0 (cold) to 1 (hot), derived from latitude + elevation
    - Moisture: 0 (dry) to 1 (wet), derived from distance to water bodies
    - Elevation: 0 (sea level) to 1 (peak)
    - Water proximity: distances to rivers, lakes, and ocean
    
    Biome logic:
    - FOREST: High moisture, moderate temp, near rivers/lakes, mid elevations
    - PLAINS: Less moisture, hot OR cold extremes, near rivers but not lakes, low elevation
    - HILLS: Mid-high elevation key, lakes common, rivers flow through, moderate climate
    - DESERT: Far from all water, hot temp, very low moisture, latitude zones
    - SNOW_TUNDRA: Cold temperatures (high latitude or elevation)
    """
    scores = {}
    
    # Normalize distance factors (closer = higher score, farther = lower)
    river_proximity = np.exp(-dist_river / 10)  # Strong decay - rivers are narrow influence
    lake_proximity = np.exp(-dist_lake / 20)    # Moderate decay - lakes have wider influence
    ocean_proximity = np.exp(-dist_ocean / 30)  # Gradual decay - ocean has broad influence
    
    # Inverse - farther from water
    far_from_river = 1.0 - river_proximity
    far_from_lake = 1.0 - lake_proximity
    far_from_ocean = 1.0 - ocean_proximity
    far_from_all_water = far_from_river * far_from_lake * far_from_ocean
    
    # ==========================================================================
    # FOREST: High moisture, moderate temp, thrives near rivers and lakes
    # ==========================================================================
    # Moisture is king for forests
    forest_moisture = moist ** 0.7  # Slight boost to high moisture
    
    # Temperature: forests like moderate temps (0.3-0.7 range)
    forest_temp = 1.0 - 2.0 * abs(temp - 0.5)  # Peak at 0.5, drops at extremes
    forest_temp = max(0.0, forest_temp)
    
    # Water proximity bonus - forests love rivers and lakes
    forest_water_bonus = 0.5 * river_proximity + 0.4 * lake_proximity + 0.1 * ocean_proximity
    
    # Elevation: forests prefer low-to-mid elevations
    forest_elev = 1.0 - elev * 0.8  # Slight penalty for very high
    
    forest_score = (
        0.35 * forest_moisture +      # High moisture is essential
        0.20 * forest_temp +          # Moderate temperature
        0.30 * forest_water_bonus +   # Near rivers/lakes strongly preferred
        0.15 * forest_elev            # Lower elevations preferred
    )
    scores['forest'] = forest_score
    
    # ==========================================================================
    # PLAINS: Moderate moisture, moderate temp, near rivers, low elevation
    # ==========================================================================
    # Plains have moderate moisture (not too wet, not too dry)
    plains_moisture = 1.0 - abs(moist - 0.45)  # Peak around 0.45 moisture
    # Heavily penalize very dry conditions (those should be desert)
    if moist < 0.25:
        plains_moisture *= 0.3  # Strong penalty for very dry
    
    # Temperature: plains prefer moderate temperatures
    # NOT hot (those should be desert) and NOT cold (those should be tundra)
    plains_temp = 1.0 - abs(temp - 0.40)  # Peak around 0.40 temp
    if temp > 0.50:
        plains_temp *= 0.6  # Penalty for hot areas (desert territory)
    
    # Water: near rivers (grass needs some water), but NOT near lakes
    plains_water = 0.6 * river_proximity + 0.4 * (1.0 - lake_proximity * 0.7)
    
    # Elevation: plains prefer low, flat areas
    plains_elev = (1.0 - elev) ** 0.8
    
    plains_score = (
        0.30 * plains_moisture +      # Moderate moisture
        0.20 * plains_temp +          # Moderate temperature
        0.25 * plains_water +         # Near rivers, away from lakes
        0.25 * plains_elev            # Low elevation
    )
    scores['plains'] = plains_score
    
    # ==========================================================================
    # HILLS: Elevation is key, split into subtypes by moisture
    # - rocky_hills: Low moisture, barren/rocky terrain
    # - grassy_hills: Medium moisture, grass-covered rolling hills
    # - forest_hills: High moisture, tree-covered hills
    # ==========================================================================
    # Elevation is the primary factor - mid to high
    hills_elev = 0.0
    if elev > 0.25:
        hills_elev = (elev - 0.25) / 0.75  # Scale 0.25-1.0 -> 0-1
        hills_elev = min(1.0, hills_elev * 1.3)  # Boost mid-range
    
    # Hills often have lakes (highland lakes)
    hills_lake = lake_proximity * 0.8
    
    # Rivers flow through hills
    hills_river = river_proximity * 0.6
    
    # Base hills score from elevation
    hills_base = (
        0.60 * hills_elev +           # Elevation is KEY
        0.20 * hills_lake +           # Lakes are common in hills
        0.20 * hills_river            # Rivers flow through
    )
    
    # Suppress at very low elevations
    if elev < 0.20:
        hills_base *= 0.3
    
    # --- ROCKY HILLS: Low moisture (< 0.35), can be warm or cold ---
    rocky_moisture_score = 0.0
    if moist < 0.35:
        rocky_moisture_score = (0.35 - moist) / 0.35  # Higher when dryer
    rocky_hills_score = hills_base * (0.5 + 0.5 * rocky_moisture_score)
    # Rocky hills prefer dry conditions but not extreme cold
    if temp < 0.20:
        rocky_hills_score *= 0.7  # Slight penalty for very cold (becomes tundra)
    scores['rocky_hills'] = rocky_hills_score
    
    # --- GRASSY HILLS: Medium moisture (0.30-0.55), moderate temp ---
    grassy_moisture_score = 1.0 - abs(moist - 0.42) / 0.25  # Peak at 0.42
    grassy_moisture_score = max(0.0, grassy_moisture_score)
    grassy_temp_score = 1.0 - abs(temp - 0.45) / 0.35  # Peak at moderate temp
    grassy_temp_score = max(0.0, grassy_temp_score)
    grassy_hills_score = hills_base * (0.4 + 0.35 * grassy_moisture_score + 0.25 * grassy_temp_score)
    # Grassy hills don't like extremes
    if moist > 0.60 or moist < 0.20:
        grassy_hills_score *= 0.5
    scores['grassy_hills'] = grassy_hills_score
    
    # --- FOREST HILLS: High moisture (> 0.45), moderate temp ---
    forest_moisture_score = 0.0
    if moist > 0.45:
        forest_moisture_score = (moist - 0.45) / 0.55  # Higher when wetter
    forest_temp_score = 1.0 - abs(temp - 0.50) / 0.40  # Moderate temp preferred
    forest_temp_score = max(0.0, forest_temp_score)
    forest_hills_score = hills_base * (0.3 + 0.45 * forest_moisture_score + 0.25 * forest_temp_score)
    # Forest hills need moisture and moderate temp
    if moist < 0.35:
        forest_hills_score *= 0.3  # Need moisture for trees
    if temp < 0.25 or temp > 0.70:
        forest_hills_score *= 0.6  # Trees don't like extremes
    # Strong bonus for water proximity (forest hills love rivers and lakes)
    forest_hills_score += 0.15 * river_proximity  # Rivers! Forest hills along rivers
    forest_hills_score += 0.10 * lake_proximity    # Lakes too
    scores['forest_hills'] = forest_hills_score
    
    # ==========================================================================
    # DESERT: Far from ALL water, hot, very low moisture, latitude dependent
    # ==========================================================================
    # Moisture must be very low - this is the DEFINING characteristic
    desert_dry = (1.0 - moist) ** 1.2  # Rewards dryness
    if moist < 0.30:
        desert_dry = 1.0 + 0.3 * (0.30 - moist) / 0.30  # Bonus for very dry
    elif moist > 0.40:
        desert_dry *= 0.3  # Penalty for moist areas
    
    # Temperature: deserts are HOT - must be warm to hot
    desert_hot = 0.0
    if temp > 0.30:
        desert_hot = (temp - 0.30) / 0.70  # Scale 0.30-1.0 -> 0-1
        desert_hot = min(1.0, desert_hot * 1.3)  # Boost hot areas
    # Cold areas get ZERO desert heat score
    
    # Distance from water - deserts are FAR from all water sources
    desert_isolation = (
        0.40 * far_from_lake +        # Far from lakes (most important)
        0.35 * far_from_river +       # Far from rivers
        0.25 * far_from_ocean         # Far from coast
    )
    
    # Strong penalty for being close to any water
    if dist_lake < 12 or dist_river < 6:
        desert_isolation *= 0.3  # Strong penalty near water
    
    desert_score = (
        0.40 * desert_dry +           # Low moisture essential (increased)
        0.35 * desert_hot +           # High temperature (increased)
        0.25 * desert_isolation       # Far from water
    )
    
    # Hard cutoff: deserts MUST be warm
    if temp < 0.20:
        desert_score *= 0.05  # Nearly zero for cold areas
    
    scores['desert'] = desert_score
    
    # ==========================================================================
    # SNOW/TUNDRA: Cold temperatures from latitude or elevation
    # ==========================================================================
    # Temperature is primary - must be cold
    tundra_cold = 0.0
    if temp < 0.35:
        tundra_cold = (0.35 - temp) / 0.35  # Scale 0-0.35 -> 1-0
        tundra_cold = tundra_cold ** 0.8    # Slightly favor very cold
    
    # Elevation can contribute to coldness (alpine tundra)
    tundra_alpine = 0.0
    if elev > 0.6:
        tundra_alpine = (elev - 0.6) / 0.4
    
    # Moisture: tundra can be wet or dry (permafrost limits absorption)
    # Slight preference for moderate-low moisture
    tundra_moist = 1.0 - abs(moist - 0.4)
    
    tundra_score = (
        0.60 * tundra_cold +          # Cold temperature essential
        0.25 * tundra_alpine +        # High elevation helps
        0.15 * tundra_moist           # Moderate moisture
    )
    
    # Hard cutoff - suppress if warm
    if temp > 0.40:
        tundra_score *= 0.15
    
    scores['snow_tundra'] = tundra_score
    
    # ==========================================================================
    # REGION INFLUENCE: Smooth biome transitions with neighbor averaging
    # ==========================================================================
    # Regional averages encourage coherent biome zones
    
    # Moist regions favor forests
    if region_avg_moist > 0.50:
        scores['forest'] += 0.08 * (region_avg_moist - 0.50) / 0.50
    
    # Dry regions favor deserts (if also hot)
    if region_avg_moist < 0.35 and region_avg_temp > 0.50:
        scores['desert'] += 0.06
    
    # Cold regions favor tundra
    if region_avg_temp < 0.30:
        scores['snow_tundra'] += 0.08
    
    # High elevation regions favor hills subtypes based on moisture
    if region_avg_elev > 0.40:
        if region_avg_moist < 0.35:
            scores['rocky_hills'] += 0.06
        elif region_avg_moist > 0.55:
            scores['forest_hills'] += 0.06
        else:
            scores['grassy_hills'] += 0.06
    
    # Low elevation moderate regions favor plains
    if region_avg_elev < 0.35 and 0.30 <= region_avg_moist <= 0.55:
        scores['plains'] += 0.05
    
    return scores


def assign_biomes(width, height, seed, is_land, is_mountain, is_lake, is_river, is_island,
                  temperature, moisture, elevation, distance_from_ocean,
                  dist_river=None, dist_lake=None):
    """
    Assign biomes using score-based competition:
    - Compute scores for each biome at each tile
    - Select biome with highest score
    - Apply region influence (8x8 averaging biases)
    - Enforce minimum cluster size
    - Validate global percentages and adjust if needed
    """
    rng = np.random.default_rng(seed + 6000)
    
    biomes = np.empty((height, width), dtype=object)
    biomes.fill('ocean')
    
    land_mask = is_land | is_island
    lake_mask = is_lake
    effective_land = land_mask & ~lake_mask
    land_count = effective_land.sum()
    
    target_mountain_count = int(land_count * 0.10)
    
    # If distance fields not provided, compute placeholder values
    if dist_river is None:
        dist_river = np.full((height, width), 50.0)  # Default far from river
    if dist_lake is None:
        dist_lake = np.full((height, width), 50.0)   # Default far from lake
    
    # Compute region averages for influence
    region_avg_moist, region_avg_temp, region_avg_elev = compute_region_averages(
        width, height, temperature, moisture, elevation, is_land, is_island
    )
    
    # First pass: assign biomes using score-based competition
    for y in range(height):
        for x in range(width):
            if not effective_land[y, x]:
                if lake_mask[y, x]:
                    biomes[y, x] = 'lake'
                elif not land_mask[y, x]:
                    biomes[y, x] = 'ocean'
                continue
            
            # Mountains are pre-determined
            if is_mountain[y, x]:
                biomes[y, x] = 'mountain'
                continue
            
            # Get tile values
            temp = temperature[y, x]
            moist = moisture[y, x]
            elev = elevation[y, x]
            d_ocean = distance_from_ocean[y, x]
            d_river = dist_river[y, x]
            d_lake = dist_lake[y, x]
            
            # Get region averages for this tile
            reg_moist = region_avg_moist[y, x]
            reg_temp = region_avg_temp[y, x]
            reg_elev = region_avg_elev[y, x]
            
            # Compute scores for each biome
            scores = compute_biome_scores(temp, moist, elev, d_ocean, d_river, d_lake,
                                         reg_moist, reg_temp, reg_elev)
            
            # Select biome with highest score
            best_biome = max(scores, key=scores.get)
            biomes[y, x] = best_biome
    
    # Adjust mountains to exactly 10%
    mountain_tiles = [(y, x) for y in range(height) for x in range(width) 
                      if effective_land[y, x] and biomes[y, x] == 'mountain']
    current_mountains = len(mountain_tiles)
    
    if current_mountains > target_mountain_count:
        candidates = [(elevation[y, x], y, x) for y, x in mountain_tiles if not is_mountain[y, x]]
        if len(candidates) < current_mountains - target_mountain_count:
            candidates.extend([(elevation[y, x], y, x) for y, x in mountain_tiles if is_mountain[y, x]])
        candidates.sort(reverse=True)
        
        for i in range(current_mountains - target_mountain_count):
            if i < len(candidates):
                _, y, x = candidates[i]
                # Use score-based assignment for replacement
                scores = compute_biome_scores(
                    temperature[y, x], moisture[y, x], elevation[y, x],
                    distance_from_ocean[y, x], dist_river[y, x], dist_lake[y, x],
                    region_avg_moist[y, x], region_avg_temp[y, x], region_avg_elev[y, x]
                )
                biomes[y, x] = max(scores, key=scores.get)
                
    elif current_mountains < target_mountain_count:
        non_mountain_land = [(elevation[y, x], y, x) for y in range(height) for x in range(width)
                            if effective_land[y, x] and biomes[y, x] != 'mountain']
        non_mountain_land.sort(reverse=True)
        
        for i in range(target_mountain_count - current_mountains):
            if i < len(non_mountain_land):
                _, y, x = non_mountain_land[i]
                biomes[y, x] = 'mountain'
    
    # Apply spatial coherence smoothing with neighbor influence
    biomes = smooth_biomes(biomes, effective_land, temperature, moisture, elevation, 
                          dist_river, dist_lake, distance_from_ocean,
                          region_avg_moist, region_avg_temp, region_avg_elev, seed, num_passes=3)
    
    # Enforce minimum cluster size
    biomes = enforce_min_cluster_size(biomes, effective_land, temperature, moisture, elevation, 
                                       dist_river, dist_lake, distance_from_ocean,
                                       region_avg_moist, region_avg_temp, region_avg_elev,
                                       MIN_BIOME_CLUSTER_SIZE)
    
    # Calculate targets for non-mountain biomes
    non_mountain_land_count = land_count - target_mountain_count
    
    targets = {
        'plains': (0.25, 0.35),
        'forest': (0.18, 0.30),
        'desert': (0.05, 0.15),  # Climate constrained, variable
        'rocky_hills': (0.03, 0.07),
        'grassy_hills': (0.03, 0.07),
        'forest_hills': (0.03, 0.07),
        'snow_tundra': (0.15, 0.28),
    }
    
    target_counts = {}
    for biome, (min_pct, max_pct) in targets.items():
        target_counts[biome] = (int(non_mountain_land_count * min_pct), 
                                int(non_mountain_land_count * max_pct))
    
    # Enforce percentage constraints
    biomes = enforce_biome_percentages_v2(biomes, effective_land, non_mountain_land_count,
                                          target_counts, temperature, moisture, elevation,
                                          dist_river, dist_lake, distance_from_ocean,
                                          region_avg_moist, region_avg_temp, region_avg_elev, seed)
    
    # Final smoothing pass to fix any isolated tiles created by percentage enforcement
    biomes = enforce_min_cluster_size(biomes, effective_land, temperature, moisture, elevation,
                                       dist_river, dist_lake, distance_from_ocean,
                                       region_avg_moist, region_avg_temp, region_avg_elev,
                                       MIN_BIOME_CLUSTER_SIZE, debug=False)
    
    return biomes


def enforce_desert_lake_constraint(biomes, is_lake, lake_ids, seed):
    """
    Enforce the desert lake constraint: 0-2 lakes in desert.
    - 70% chance: 0 lakes in desert
    - 20% chance: 1 lake in desert
    - 10% chance: 2 lakes in desert
    
    Lakes exceeding the limit are converted to the most common surrounding biome.
    Returns updated biomes map and a list of removed lake IDs.
    """
    max_desert_lakes = determine_desert_lake_count(seed)
    height, width = biomes.shape
    
    # Find all lakes that are in desert biome
    desert_lakes = {}  # lake_id -> list of tiles
    
    for y in range(height):
        for x in range(width):
            if is_lake[y, x] and biomes[y, x] == 'desert':
                lid = lake_ids[y, x]
                if lid >= 0:
                    if lid not in desert_lakes:
                        desert_lakes[lid] = []
                    desert_lakes[lid].append((y, x))
    
    # Check how many lakes have ANY tiles in desert
    lakes_with_desert = set()
    for y in range(height):
        for x in range(width):
            if is_lake[y, x]:
                lid = lake_ids[y, x]
                if lid >= 0:
                    # Check if any adjacent or same-lake tile is desert
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < height and 0 <= nx < width:
                                if biomes[ny, nx] == 'desert' and not is_lake[ny, nx]:
                                    lakes_with_desert.add(lid)
    
    # If we have more lakes in/touching desert than allowed, we need to either:
    # 1. Convert the lake tiles to land with appropriate biome, or
    # 2. Convert desert tiles near those lakes to a different biome
    # We'll use option 2: push desert away from excess lakes
    
    removed_desert_lake_ids = []
    
    if len(lakes_with_desert) > max_desert_lakes:
        # Sort lake IDs and pick which ones to "remove from desert"
        rng = np.random.default_rng(seed + 4600)
        lake_list = list(lakes_with_desert)
        rng.shuffle(lake_list)
        
        lakes_to_keep_in_desert = set(lake_list[:max_desert_lakes])
        lakes_to_remove_from_desert = set(lake_list[max_desert_lakes:])
        removed_desert_lake_ids = list(lakes_to_remove_from_desert)
        
        # For lakes to remove from desert, change surrounding desert to plains/hills
        for lid in lakes_to_remove_from_desert:
            # Find all tiles of this lake
            lake_tiles = []
            for y in range(height):
                for x in range(width):
                    if lake_ids[y, x] == lid:
                        lake_tiles.append((y, x))
            
            # Change desert tiles near this lake to plains or hills
            for ly, lx in lake_tiles:
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        ny, nx = ly + dy, lx + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if biomes[ny, nx] == 'desert' and not is_lake[ny, nx]:
                                # Change to plains (more water = less desert)
                                biomes[ny, nx] = 'plains'
    
    return biomes, max_desert_lakes, len(lakes_with_desert), removed_desert_lake_ids


def smooth_biomes(biomes, effective_land, temperature, moisture, elevation,
                  dist_river, dist_lake, distance_from_ocean,
                  region_avg_moist, region_avg_temp, region_avg_elev, seed, num_passes=3):
    """
    Apply cellular automata-style smoothing to biomes.
    Tiles consider their neighbors when determining biome.
    Uses score-based competition for replacement decisions.
    """
    height, width = biomes.shape
    rng = np.random.default_rng(seed + 8000)
    
    for pass_num in range(num_passes):
        new_biomes = biomes.copy()
        
        for y in range(height):
            for x in range(width):
                if not effective_land[y, x]:
                    continue
                
                if biomes[y, x] == 'mountain':
                    continue  # Don't smooth mountains
                
                current_biome = biomes[y, x]
                
                # Count neighbor biomes
                neighbor_counts = {}
                total_neighbors = 0
                
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if effective_land[ny, nx] and biomes[ny, nx] not in ['ocean', 'lake']:
                                nb = biomes[ny, nx]
                                neighbor_counts[nb] = neighbor_counts.get(nb, 0) + 1
                                total_neighbors += 1
                
                if total_neighbors == 0:
                    continue
                
                # Calculate weighted scores for each biome using score-based competition
                biome_scores = {}
                for biome, count in neighbor_counts.items():
                    if biome == 'mountain':
                        continue  # Don't convert to mountain via smoothing
                    
                    # Neighbor influence (higher count = higher score)
                    neighbor_weight = count / total_neighbors
                    
                    # Use new score-based suitability
                    tile_scores = compute_biome_scores(
                        temperature[y, x], moisture[y, x], elevation[y, x],
                        distance_from_ocean[y, x], dist_river[y, x], dist_lake[y, x],
                        region_avg_moist[y, x], region_avg_temp[y, x], region_avg_elev[y, x]
                    )
                    climate_weight = tile_scores.get(biome, 0.5)
                    
                    # Combined score with neighbor bias
                    biome_scores[biome] = neighbor_weight * 0.6 + climate_weight * 0.4
                
                # Add current biome with slight bonus for stability
                if current_biome != 'mountain':
                    tile_scores = compute_biome_scores(
                        temperature[y, x], moisture[y, x], elevation[y, x],
                        distance_from_ocean[y, x], dist_river[y, x], dist_lake[y, x],
                        region_avg_moist[y, x], region_avg_temp[y, x], region_avg_elev[y, x]
                    )
                    current_score = tile_scores.get(current_biome, 0.5)
                    biome_scores[current_biome] = biome_scores.get(current_biome, 0) + current_score * 0.3
                
                if biome_scores:
                    # Deterministic selection based on position
                    best_biome = max(biome_scores, key=biome_scores.get)
                    
                    # Only change if neighbor influence is strong enough
                    if best_biome != current_biome:
                        neighbor_consensus = neighbor_counts.get(best_biome, 0) / total_neighbors
                        if neighbor_consensus >= 0.4:  # At least 40% neighbors agree
                            new_biomes[y, x] = best_biome
        
        biomes = new_biomes
    
    return biomes


def enforce_min_cluster_size(biomes, effective_land, temperature, moisture, elevation,
                             dist_river, dist_lake, distance_from_ocean,
                             region_avg_moist, region_avg_temp, region_avg_elev, min_size, debug=False):
    """
    Ensure no isolated single-tile or small biome patches.
    Tiles in clusters smaller than min_size are reassigned using score-based competition.
    Runs iteratively until no more small clusters exist.
    """
    height, width = biomes.shape
    max_iterations = 50  # Increased iterations for thorough enforcement
    locked_tiles = set()  # Tiles that have stabilized - don't change them again
    tile_change_count = {}  # Track how many times each tile has been changed
    
    for iteration in range(max_iterations):
        changes_made = False
        small_clusters_found = {b: 0 for b in ['plains', 'forest', 'desert', 'rocky_hills', 'grassy_hills', 'forest_hills', 'snow_tundra', 'mountain']}
        changed_this_iteration = set()  # Track tiles changed to prevent oscillation within iteration
        
        # Define 8-connectivity structure (includes diagonals)
        connectivity_struct = np.array([[1, 1, 1],
                                        [1, 1, 1],
                                        [1, 1, 1]])
        
        # Process each biome type
        for biome_type in ['plains', 'forest', 'desert', 'rocky_hills', 'grassy_hills', 'forest_hills', 'snow_tundra', 'mountain']:
            biome_mask = (biomes == biome_type) & effective_land
            
            if biome_mask.sum() == 0:
                continue
            
            # Label connected components with 8-connectivity
            labeled, num_features = ndimage.label(biome_mask, structure=connectivity_struct)
            
            for comp_id in range(1, num_features + 1):
                comp_mask = labeled == comp_id
                comp_size = comp_mask.sum()
                
                if comp_size < min_size:
                    small_clusters_found[biome_type] += 1
                    # Find tiles in this small cluster
                    cluster_tiles = np.argwhere(comp_mask)
                    
                    for ty, tx in cluster_tiles:
                        # Skip tiles that are locked (already stabilized or oscillating)
                        if (ty, tx) in locked_tiles:
                            continue
                        # Skip tiles already changed this iteration to prevent oscillation
                        if (ty, tx) in changed_this_iteration:
                            continue
                            
                        # Find dominant neighbor biome - search wider area for edge cases
                        neighbor_counts = {}
                        search_radius = 3  # Increased for better edge handling
                        
                        for dy in range(-search_radius, search_radius + 1):
                            for dx in range(-search_radius, search_radius + 1):
                                if dy == 0 and dx == 0:
                                    continue
                                ny, nx = ty + dy, tx + dx
                                if 0 <= ny < height and 0 <= nx < width:
                                    if effective_land[ny, nx]:
                                        nb = biomes[ny, nx]
                                        if nb not in ['ocean', 'lake', biome_type]:
                                            # Weight by distance (closer = more important)
                                            dist = max(abs(dy), abs(dx))
                                            weight = 1.0 / dist
                                            neighbor_counts[nb] = neighbor_counts.get(nb, 0) + weight
                        
                        if neighbor_counts:
                            # Use score-based selection with neighbor count as tiebreaker
                            tile_scores = compute_biome_scores(
                                temperature[ty, tx], moisture[ty, tx], elevation[ty, tx],
                                distance_from_ocean[ty, tx], dist_river[ty, tx], dist_lake[ty, tx],
                                region_avg_moist[ty, tx], region_avg_temp[ty, tx], region_avg_elev[ty, tx]
                            )
                            best_neighbor = max(neighbor_counts, key=lambda b: (
                                neighbor_counts[b], tile_scores.get(b, 0.5)
                            ))
                            if biomes[ty, tx] != best_neighbor:
                                if debug and iteration >= 10:
                                    print(f"    Changing ({ty},{tx}) from {biome_type} to {best_neighbor}")
                                biomes[ty, tx] = best_neighbor
                                changed_this_iteration.add((ty, tx))
                                changes_made = True
                                
                                # Track change count - lock tiles that oscillate too many times
                                tile_change_count[(ty, tx)] = tile_change_count.get((ty, tx), 0) + 1
                                if tile_change_count[(ty, tx)] >= 3:
                                    locked_tiles.add((ty, tx))
                        else:
                            # No valid neighbors of different biome found 
                            # This tile is isolated - count it for debug
                            pass  # Skip debug output for isolated tiles
        
        if debug:
            total_small = sum(small_clusters_found.values())
            print(f"  Iteration {iteration+1}: {total_small} small clusters found, changes_made={changes_made}")
            if total_small > 0:
                for b, cnt in small_clusters_found.items():
                    if cnt > 0:
                        print(f"    {b}: {cnt} small clusters")
        
        # If no changes were made this iteration, we're done
        if not changes_made:
            break
    
    return biomes


def count_biomes(biomes, effective_land):
    """Count biome occurrences on effective land."""
    land_biomes = biomes[effective_land]
    if land_biomes.size == 0:
        return {}
    unique, counts_arr = np.unique(land_biomes, return_counts=True)
    counts = {str(b): int(c) for b, c in zip(unique, counts_arr)}

    return counts


def enforce_biome_percentages_v2(biomes, effective_land, non_mountain_count, target_counts,
                                  temperature, moisture, elevation,
                                  dist_river, dist_lake, distance_from_ocean,
                                  region_avg_moist, region_avg_temp, region_avg_elev, seed):
    """
    Enforce biome percentages while respecting climate constraints.
    Uses vectorized climate suitability masks for efficiency.
    """
    rng = np.random.default_rng(seed + 7000)
    height, width = biomes.shape
    
    # Pre-compute climate suitability masks (vectorized)
    # Desert: dry (low moisture), warm/hot, far from water - RELAXED criteria for enforcement
    desert_suitable = (moisture < 0.55) & (temperature > 0.18) & (dist_lake > 4) & (dist_river > 2)
    
    # Forest: needs moisture, moderate temperature
    forest_suitable = (moisture > 0.35) | ((dist_river < 6) | (dist_lake < 10))
    
    # Snow_tundra: needs cold
    tundra_suitable = (temperature < 0.40) | (elevation > 0.60)
    
    # Plains: flexible, prefer low elevation
    plains_suitable = (elevation < 0.75) & (temperature > 0.03)
    
    # Hills: needs elevation - all subtypes share this
    hills_suitable = (elevation > 0.30) | (dist_lake < 18)
    rocky_hills_suitable = hills_suitable & (moisture < 0.40)
    grassy_hills_suitable = hills_suitable & (moisture >= 0.30) & (moisture <= 0.60)
    forest_hills_suitable = hills_suitable & (moisture > 0.45)
    
    suitability_masks = {
        'desert': desert_suitable,
        'forest': forest_suitable,
        'snow_tundra': tundra_suitable,
        'plains': plains_suitable,
        'rocky_hills': rocky_hills_suitable,
        'grassy_hills': grassy_hills_suitable,
        'forest_hills': forest_hills_suitable,
    }
    
    # Also pre-compute simple suitability scores (vectorized)
    # Desert score: lower moisture = higher score, REQUIRES warmth
    desert_heat = np.clip((temperature - 0.25) / 0.75, 0, 1)  # 0 below 0.25, scales up
    desert_score = ((1.0 - moisture) * 0.4 + desert_heat * 0.35 + np.clip(dist_lake / 30, 0, 1) * 0.25)
    desert_score = np.where(temperature < 0.20, desert_score * 0.1, desert_score)  # Heavy penalty for cold
    
    forest_score = moisture * 0.5 + np.clip(1.0 - dist_river / 15, 0, 1) * 0.3 + (1.0 - elevation) * 0.2
    tundra_score = (1.0 - temperature) * 0.6 + elevation * 0.3 + moisture * 0.1
    plains_score = (1.0 - elevation) * 0.4 + np.clip(1.0 - dist_river / 20, 0, 1) * 0.3 + 0.3
    hills_base_score = elevation * 0.5 + np.clip(1.0 - dist_lake / 25, 0, 1) * 0.3 + 0.2
    rocky_hills_score = hills_base_score * np.clip((0.40 - moisture) / 0.30, 0.3, 1.0)
    grassy_hills_score = hills_base_score * (1.0 - np.abs(moisture - 0.45) / 0.25)
    # Forest hills get strong river proximity bonus (rivers flow through forested hills)
    forest_hills_score = hills_base_score * np.clip((moisture - 0.35) / 0.35, 0.3, 1.0) + np.clip(1.0 - dist_river / 12, 0, 1) * 0.15
    
    score_arrays = {
        'desert': desert_score,
        'forest': forest_score,
        'snow_tundra': tundra_score,
        'plains': plains_score,
        'rocky_hills': rocky_hills_score,
        'grassy_hills': grassy_hills_score,
        'forest_hills': forest_hills_score,
    }
    
    max_iterations = 2000
    
    for iteration in range(max_iterations):
        counts = count_biomes(biomes, effective_land)
        
        all_satisfied = True
        deficits = {}
        surpluses = {}
        in_range = {}
        
        for biome, (min_count, max_count) in target_counts.items():
            current = counts.get(biome, 0)
            
            if current < min_count:
                deficits[biome] = min_count - current
                all_satisfied = False
            elif current > max_count:
                surpluses[biome] = current - max_count
                all_satisfied = False
            else:
                in_range[biome] = max_count - current
        
        if all_satisfied:
            break
        
        if surpluses and not deficits:
            can_accept = {b: room for b, room in in_range.items() if room > 0}
            if not can_accept:
                break
            deficits = can_accept
        
        surplus_items = sorted(surpluses.items(), key=lambda x: -x[1])
        deficit_items = sorted(deficits.items(), key=lambda x: -x[1])
        
        for surplus_biome, surplus_count in surplus_items:
            if surplus_count <= 0:
                continue
                
            for deficit_biome, deficit_count in deficit_items:
                if surplus_count <= 0 or deficit_count <= 0:
                    continue
                
                convert_count = min(surplus_count, deficit_count, 500)
                
                # For desert: search across ALL biomes for suitable hot/dry tiles
                # (not just the surplus biome, which might be cold snow_tundra)
                if deficit_biome == 'desert':
                    # Look for desert-suitable tiles in ANY non-desert biome
                    candidate_mask = (
                        effective_land & 
                        (biomes != 'desert') &
                        (biomes != 'mountain') &
                        suitability_masks.get('desert', np.ones_like(effective_land, dtype=bool))
                    )
                else:
                    # Standard: convert from surplus biome
                    candidate_mask = (
                        effective_land & 
                        (biomes == surplus_biome) & 
                        suitability_masks.get(deficit_biome, np.ones_like(effective_land, dtype=bool))
                    )
                
                score_arr = score_arrays.get(deficit_biome, np.ones((height, width)) * 0.5)
                
                # Get candidate positions and scores
                ys, xs = np.where(candidate_mask)
                if len(ys) == 0:
                    # For climate-critical biomes (desert, snow_tundra), NEVER fallback to unsuitable tiles
                    if deficit_biome in ['desert', 'snow_tundra']:
                        continue  # Skip - don't force these into wrong climates
                    
                    # For other biomes, try without climate filter but with penalty
                    candidate_mask = effective_land & (biomes == surplus_biome)
                    ys, xs = np.where(candidate_mask)
                    if len(ys) == 0:
                        continue
                    scores = score_arr[ys, xs] * 0.5  # Penalty for unsuitable climate
                else:
                    scores = score_arr[ys, xs]
                
                # Add adjacency bonus
                for i in range(len(ys)):
                    y, x = ys[i], xs[i]
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if biomes[ny, nx] == deficit_biome:
                                scores[i] += 0.3
                                break
                
                # Sort by score and convert top candidates
                sorted_idx = np.argsort(-scores)
                num_convert = min(convert_count, len(sorted_idx))
                
                for i in range(num_convert):
                    idx = sorted_idx[i]
                    biomes[ys[idx], xs[idx]] = deficit_biome
                
                surplus_count -= num_convert
                deficits[deficit_biome] = deficit_count - num_convert
        
        deficit_items = [(b, deficits.get(b, 0)) for b in deficits if deficits.get(b, 0) > 0]
    
    return biomes


def biome_suitability(biome, temp, moist, elev, dist_ocean=50, dist_river=50, dist_lake=50):
    """
    Legacy suitability function for backwards compatibility.
    Uses simplified scoring based on the new competition model.
    """
    scores = compute_biome_scores(temp, moist, elev, dist_ocean, dist_river, dist_lake, 
                                  moist, temp, elev)  # Use tile values as region averages
    return scores.get(biome, 0.5)


# =============================================================================
# BIOME CLUSTER STATISTICS
# =============================================================================
def calculate_biome_cluster_stats(biomes, effective_land):
    """Calculate statistics about biome clusters."""
    height, width = biomes.shape
    stats = {}
    
    for biome_type in ['plains', 'forest', 'desert', 'rocky_hills', 'grassy_hills', 'forest_hills', 'snow_tundra', 'mountain']:
        biome_mask = (biomes == biome_type) & effective_land
        
        if biome_mask.sum() == 0:
            stats[biome_type] = {'count': 0, 'clusters': 0, 'avg_size': 0, 'min_size': 0, 'max_size': 0}
            continue
        
        labeled, num_features = ndimage.label(biome_mask)
        
        if num_features == 0:
            stats[biome_type] = {'count': 0, 'clusters': 0, 'avg_size': 0, 'min_size': 0, 'max_size': 0}
            continue
        
        component_sizes = ndimage.sum(biome_mask, labeled, range(1, num_features + 1))
        
        stats[biome_type] = {
            'count': int(biome_mask.sum()),
            'clusters': num_features,
            'avg_size': float(np.mean(component_sizes)),
            'min_size': int(np.min(component_sizes)),
            'max_size': int(np.max(component_sizes))
        }
    
    return stats


# =============================================================================
# REGION ASSIGNMENT
# =============================================================================
def assign_regions(width, height):
    """Assign region IDs to each tile (8x8 regions, 25x25 total)."""
    x_ids = (np.arange(width, dtype=np.int32) // REGION_SIZE)
    y_ids = (np.arange(height, dtype=np.int32) // REGION_SIZE)
    region_x = np.broadcast_to(x_ids, (height, width)).copy()
    region_y = np.broadcast_to(y_ids[:, None], (height, width)).copy()

    return region_x, region_y


# =============================================================================
# TILE ENVIRONMENT RESOURCE SYSTEM
# =============================================================================
"""
Environmental resource system that computes tile-level capacities and potentials.
This system reads from existing terrain data and produces environmental profiles
for use by AI simulation layers.

Does NOT modify: terrain, rivers, temperature, moisture, or biome generation.
"""

# Biome vegetation density lookup (for fertility and biomass calculations)
BIOME_VEGETATION_DENSITY = {
    # Forests - high vegetation
    'rainforest': 1.0,
    'tropical_forest': 0.95,
    'temperate_forest': 0.85,
    'woodland': 0.7,
    'forest_mountains': 0.65,
    'forest_hills': 0.75,
    'snow_forest': 0.5,
    
    # Wetlands - moderate to high
    'swamp': 0.8,
    'marsh': 0.7,
    'mangrove': 0.85,
    
    # Grasslands - moderate
    'grassland': 0.5,
    'meadow': 0.55,
    'savanna': 0.4,
    'steppe': 0.3,
    'grassy_hills': 0.45,
    'alpine_meadows': 0.35,
    
    # Sparse vegetation
    'rocky_mountains': 0.1,
    'rocky_hills': 0.15,
    'snow_mountains': 0.05,
    'snow_hills': 0.1,
    'snow_plains': 0.15,
    'glacier': 0.0,
    'badlands': 0.1,
    
    # Deserts - minimal
    'sand_desert': 0.05,
    'rock_desert': 0.08,
    'oasis': 0.6,
    
    # Water
    'ocean': 0.0,
    'lake': 0.0,
}

# Biome sedimentary potential (for fossil fuels and coal)
BIOME_SEDIMENTARY = {
    'steppe': 0.7, 'grassland': 0.6, 'savanna': 0.5, 'meadow': 0.55,
    'swamp': 0.9, 'marsh': 0.85, 'mangrove': 0.8,
    'sand_desert': 0.4, 'rock_desert': 0.3,
    'snow_plains': 0.5, 'grassy_hills': 0.4,
}


def generate_resource_noise(width, height, seed, scale=20, octaves=3):
    """Generate noise for resource clustering."""
    rng = np.random.default_rng(seed)
    noise = np.zeros((height, width), dtype=np.float32)
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    
    for octave in range(octaves):
        freq = 2 ** octave
        amp = 0.5 ** octave
        
        # Generate random phase shifts
        phase_x = rng.uniform(0, 100)
        phase_y = rng.uniform(0, 100)

        nx = (x_grid / scale * freq + phase_x)
        ny = (y_grid / scale * freq + phase_y)
        noise += (amp * (np.sin(nx) * np.cos(ny) + 1) / 2).astype(np.float32)
    
    # Normalize to 0-1
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
    return noise


def compute_geological_resources(elevation, slope, roughness, is_land, is_mountain, 
                                  dist_river, biomes, width, height, seed):
    """
    Compute geological resource potentials for each tile.
    
    Returns dict of 2D arrays:
    - iron_potential: Iron ore likelihood (mountains, volcanic)
    - copper_potential: Copper ore likelihood (mountains, hills)
    - tin_potential: Tin ore likelihood (granite regions)
    - gold_potential: Gold likelihood (mountains + placer in rivers)
    - coal_potential: Coal likelihood (sedimentary basins)
    - mineral_potential: General rare minerals
    - rock_quantity: Available rock for construction
    """
    # Generate clustered noise for ore veins
    iron_noise = generate_resource_noise(width, height, seed + 1001, scale=25)
    copper_noise = generate_resource_noise(width, height, seed + 1002, scale=22)
    tin_noise = generate_resource_noise(width, height, seed + 1003, scale=30)
    gold_noise = generate_resource_noise(width, height, seed + 1004, scale=35)
    coal_noise = generate_resource_noise(width, height, seed + 1005, scale=40)
    mineral_noise = generate_resource_noise(width, height, seed + 1006, scale=28)
    
    # Initialize arrays
    iron = np.zeros((height, width), dtype=np.float32)
    copper = np.zeros((height, width), dtype=np.float32)
    tin = np.zeros((height, width), dtype=np.float32)
    gold = np.zeros((height, width), dtype=np.float32)
    coal = np.zeros((height, width), dtype=np.float32)
    minerals = np.zeros((height, width), dtype=np.float32)
    rock = np.zeros((height, width), dtype=np.float32)
    
    for y in range(height):
        for x in range(width):
            if not is_land[y, x]:
                continue
            
            elev = elevation[y, x]
            slp = slope[y, x]
            rough = roughness[y, x]
            is_mtn = is_mountain[y, x]
            biome = biomes[y, x]
            river_dist = dist_river[y, x]
            
            # Base factors
            mountain_factor = 1.0 if is_mtn else (0.3 + 0.4 * elev)
            hill_factor = 0.5 + 0.5 * slp
            rocky_biome = 1.0 if biome in ('rocky_mountains', 'rocky_hills', 'rock_desert', 'badlands') else 0.5
            
            # Iron: common in mountains, volcanic regions
            iron_base = mountain_factor * 0.6 + rough * 0.2 + rocky_biome * 0.2
            iron[y, x] = np.clip(iron_base * iron_noise[y, x] * 1.5, 0, 1)
            
            # Copper: mountains and hills
            copper_base = mountain_factor * 0.5 + hill_factor * 0.3 + elev * 0.2
            copper[y, x] = np.clip(copper_base * copper_noise[y, x] * 1.4, 0, 1)
            
            # Tin: granite regions (high elevation, rough terrain)
            tin_base = elev * 0.4 + rough * 0.4 + mountain_factor * 0.2
            tin[y, x] = np.clip(tin_base * tin_noise[y, x] * 1.3, 0, 1)
            
            # Gold: mountains + placer deposits near rivers downstream of mountains
            gold_mountain = mountain_factor * 0.7
            # Placer gold: near rivers, higher if downstream of mountains
            placer_factor = max(0, 1 - river_dist / 5) * (1 - elev) * 0.5
            gold_base = gold_mountain + placer_factor
            gold[y, x] = np.clip(gold_base * gold_noise[y, x] * 1.2, 0, 1)
            
            # Coal: sedimentary basins (low elevation plains, former swamps)
            sedimentary = BIOME_SEDIMENTARY.get(biome, 0.2)
            lowland_factor = max(0, 1 - elev) * 0.6
            coal_base = sedimentary * 0.5 + lowland_factor * 0.3 + (1 - slp) * 0.2
            coal[y, x] = np.clip(coal_base * coal_noise[y, x] * 1.5, 0, 1)
            
            # General minerals: everywhere with terrain variation
            mineral_base = rough * 0.4 + elev * 0.3 + slp * 0.3
            minerals[y, x] = np.clip(mineral_base * mineral_noise[y, x] * 1.3, 0, 1)
            
            # Rock quantity: slope and roughness increase rock availability
            rock[y, x] = np.clip(slp * 0.4 + rough * 0.4 + rocky_biome * 0.2, 0, 1)
    
    return {
        'iron_potential': iron,
        'copper_potential': copper,
        'tin_potential': tin,
        'gold_potential': gold,
        'coal_potential': coal,
        'mineral_potential': minerals,
        'rock_quantity': rock
    }


def compute_soil_properties(elevation, slope, roughness, moisture, temperature,
                            dist_river, biomes, is_land, width, height, seed):
    """
    Compute soil-related properties for each tile.
    
    Returns dict of 2D arrays:
    - soil_quantity: Amount of soil (low in mountains, high in plains/valleys)
    - soil_fertility: Agricultural potential
    - sediment_accumulation: Sediment deposit potential
    """
    # Noise for soil variation
    soil_noise = generate_resource_noise(width, height, seed + 2001, scale=15)
    
    soil_qty = np.zeros((height, width), dtype=np.float32)
    soil_fert = np.zeros((height, width), dtype=np.float32)
    sediment = np.zeros((height, width), dtype=np.float32)
    
    for y in range(height):
        for x in range(width):
            if not is_land[y, x]:
                continue
            
            elev = elevation[y, x]
            slp = slope[y, x]
            moist = moisture[y, x]
            temp = temperature[y, x]
            river_dist = dist_river[y, x]
            biome = biomes[y, x]
            
            veg_density = BIOME_VEGETATION_DENSITY.get(biome, 0.3)
            
            # Soil quantity: high in plains/valleys, low in mountains/steep slopes
            # River valleys accumulate soil
            river_bonus = max(0, 1 - river_dist / 8) * 0.3
            slope_penalty = slp * 0.6
            elevation_penalty = max(0, elev - 0.5) * 0.4
            
            soil_base = 0.7 - slope_penalty - elevation_penalty + river_bonus
            soil_qty[y, x] = np.clip(soil_base * (0.7 + 0.3 * soil_noise[y, x]), 0, 1)
            
            # Soil fertility: depends on moisture, temperature, vegetation, organic matter
            # Forests and wetlands have high organic matter
            organic_matter = veg_density * 0.4
            
            # Optimal temperature for fertility (not too hot, not too cold)
            temp_factor = 1 - abs(temp - 0.5) * 1.2
            temp_factor = max(0, temp_factor)
            
            # Moisture helps but waterlogged is less fertile
            moist_factor = moist * 0.8 if moist < 0.85 else 0.85 - (moist - 0.85) * 2
            moist_factor = max(0, moist_factor)
            
            # River proximity improves fertility (floodplains)
            river_fertility = max(0, 1 - river_dist / 6) * 0.2
            
            fert_base = organic_matter + temp_factor * 0.25 + moist_factor * 0.25 + river_fertility
            
            # Desert and rocky biomes have very low fertility
            if biome in ('sand_desert', 'rock_desert', 'rocky_mountains', 'rocky_hills', 'glacier', 'badlands'):
                fert_base *= 0.2
            
            soil_fert[y, x] = np.clip(fert_base * soil_qty[y, x], 0, 1)
            
            # Sediment accumulation: river deltas, valleys, low slopes
            sediment_base = river_bonus * 2 + (1 - slp) * 0.3 + (1 - elev) * 0.2
            sediment[y, x] = np.clip(sediment_base, 0, 1)
    
    return {
        'soil_quantity': soil_qty,
        'soil_fertility': soil_fert,
        'sediment_accumulation': sediment
    }


def compute_water_resources(moisture, dist_river, dist_lake, dist_ocean, 
                            is_land, is_river, is_lake, elevation, biomes,
                            width, height, seed):
    """
    Compute water availability values for each tile.
    
    Returns dict of 2D arrays:
    - fresh_water: Fresh water availability
    - salt_water: Salt water availability (ocean/coastal)
    - wild_water: Natural unmanaged water (wetlands, seasonal)
    - groundwater_potential: Groundwater availability
    """
    fresh = np.zeros((height, width), dtype=np.float32)
    salt = np.zeros((height, width), dtype=np.float32)
    wild = np.zeros((height, width), dtype=np.float32)
    ground = np.zeros((height, width), dtype=np.float32)
    
    # Noise for groundwater variation
    ground_noise = generate_resource_noise(width, height, seed + 3001, scale=18)
    
    for y in range(height):
        for x in range(width):
            biome = biomes[y, x]
            moist = moisture[y, x]
            river_dist = dist_river[y, x]
            lake_dist = dist_lake[y, x]
            ocean_dist = dist_ocean[y, x]
            elev = elevation[y, x]
            
            if not is_land[y, x]:
                # Ocean tiles
                salt[y, x] = 1.0
                continue
            
            # Fresh water: rivers, lakes, high rainfall
            river_water = max(0, 1 - river_dist / 10) * 0.5
            lake_water = max(0, 1 - lake_dist / 8) * 0.4
            rain_water = moist * 0.3
            
            if is_river[y, x]:
                river_water = 1.0
            if is_lake[y, x]:
                lake_water = 1.0
            
            fresh[y, x] = np.clip(river_water + lake_water + rain_water, 0, 1)
            
            # Salt water: ocean and coastal tiles
            coastal_salt = max(0, 1 - ocean_dist / 5) * 0.3
            salt[y, x] = coastal_salt
            
            # Wild water: wetlands, seasonal streams, unmanaged
            if biome in ('swamp', 'marsh', 'mangrove'):
                wild[y, x] = 0.8 + moist * 0.2
            elif moist > 0.7:
                wild[y, x] = (moist - 0.7) * 2
            else:
                wild[y, x] = moist * 0.2
            
            # Groundwater: depends on soil depth (inverse of slope), rainfall, basins
            soil_depth_factor = max(0, 1 - elev) * 0.4 + (1 - elevation[y, x]) * 0.2
            rainfall_factor = moist * 0.4
            basin_factor = (1 - elev) * 0.2  # Lower areas collect groundwater
            
            ground[y, x] = np.clip(
                (soil_depth_factor + rainfall_factor + basin_factor) * ground_noise[y, x] * 1.3,
                0, 1
            )
    
    return {
        'fresh_water': fresh,
        'salt_water': salt,
        'wild_water': wild,
        'groundwater_potential': ground
    }


def compute_biological_capacity(biomes, soil_fertility, moisture, temperature,
                                 fresh_water, is_land, width, height, seed):
    """
    Compute biological capacity values for each tile.
    
    Returns dict of 2D arrays:
    - wood_biomass_capacity: Forest biomass potential
    - animal_habitat_capacity: Wildlife support capacity
    - plant_diversity_capacity: Plant species diversity potential
    - insect_population_capacity: Insect population support
    """
    wood = np.zeros((height, width), dtype=np.float32)
    animal = np.zeros((height, width), dtype=np.float32)
    plant = np.zeros((height, width), dtype=np.float32)
    insect = np.zeros((height, width), dtype=np.float32)
    
    for y in range(height):
        for x in range(width):
            if not is_land[y, x]:
                continue
            
            biome = biomes[y, x]
            fert = soil_fertility[y, x]
            moist = moisture[y, x]
            temp = temperature[y, x]
            water = fresh_water[y, x]
            
            veg_density = BIOME_VEGETATION_DENSITY.get(biome, 0.3)
            
            # Wood biomass: forests have high capacity
            if biome in ('rainforest', 'tropical_forest', 'temperate_forest', 'woodland', 
                        'forest_mountains', 'forest_hills', 'snow_forest'):
                wood[y, x] = veg_density * 0.7 + fert * 0.2 + moist * 0.1
            elif biome in ('swamp', 'mangrove'):
                wood[y, x] = veg_density * 0.5  # Wetland trees
            else:
                wood[y, x] = veg_density * 0.2  # Scattered trees
            
            # Animal habitat: depends on vegetation, water, and climate diversity
            habitat_base = veg_density * 0.4 + water * 0.3 + fert * 0.2
            # Moderate temperatures support more wildlife
            temp_habitat = 1 - abs(temp - 0.55) * 1.5
            temp_habitat = max(0, temp_habitat)
            animal[y, x] = np.clip(habitat_base + temp_habitat * 0.1, 0, 1)
            
            # Plant diversity: tropical and temperate regions with moisture
            diversity_base = moist * 0.4 + fert * 0.3 + veg_density * 0.2
            # Tropics have highest diversity
            if temp > 0.6 and moist > 0.5:
                diversity_base += 0.2
            plant[y, x] = np.clip(diversity_base, 0, 1)
            
            # Insect population: wetlands and warm moist regions
            insect_base = moist * 0.4 + temp * 0.3 + veg_density * 0.2
            if biome in ('swamp', 'marsh', 'mangrove', 'tropical_forest', 'rainforest'):
                insect_base += 0.3
            # Cold reduces insects
            if temp < 0.3:
                insect_base *= temp / 0.3
            insect[y, x] = np.clip(insect_base, 0, 1)
    
    return {
        'wood_biomass_capacity': wood,
        'animal_habitat_capacity': animal,
        'plant_diversity_capacity': plant,
        'insect_population_capacity': insect
    }


def compute_climate_exposure(moisture, temperature, slope, elevation, 
                              is_land, biomes, width, height, seed):
    """
    Compute climate exposure values for each tile.
    
    Returns dict of 2D arrays:
    - rain_exposure: Rainfall exposure factor
    - sunlight_exposure: Sunlight availability factor
    """
    rain = np.zeros((height, width), dtype=np.float32)
    sun = np.zeros((height, width), dtype=np.float32)
    
    for y in range(height):
        for x in range(width):
            if not is_land[y, x]:
                continue
            
            moist = moisture[y, x]
            temp = temperature[y, x]
            slp = slope[y, x]
            elev = elevation[y, x]
            biome = biomes[y, x]
            veg_density = BIOME_VEGETATION_DENSITY.get(biome, 0.3)
            
            # Rain exposure: moisture + terrain effects
            # Slopes facing prevailing wind get more rain
            rain_base = moist * 0.7 + (1 - elev) * 0.2
            # Mountains can block or enhance rain
            terrain_effect = slp * 0.1
            rain[y, x] = np.clip(rain_base + terrain_effect, 0, 1)
            
            # Sunlight exposure: latitude (via temp), slope orientation, vegetation shade
            # Use y position as proxy for latitude
            lat_factor = 1 - abs(y / height - 0.5) * 0.4  # Equator gets more sun
            
            # Elevation increases sun exposure (above clouds)
            elev_factor = elev * 0.15
            
            # Dense vegetation reduces ground-level sunlight
            shade_factor = veg_density * 0.25
            
            # Steep slopes may reduce effective sunlight
            slope_factor = slp * 0.1
            
            sun[y, x] = np.clip(lat_factor + elev_factor - shade_factor - slope_factor, 0, 1)
    
    return {
        'rain_exposure': rain,
        'sunlight_exposure': sun
    }


def compute_fuel_resources(wood_biomass, biomes, moisture, soil_fertility,
                            coal_potential, is_land, width, height, seed):
    """
    Compute natural fuel resource potentials.
    
    Returns dict of 2D arrays:
    - biomass_fuel_potential: Wood and plant fuel availability
    - peat_potential: Peat deposits in wetlands
    - fossil_fuel_potential: Oil and gas potential
    """
    biomass = np.zeros((height, width), dtype=np.float32)
    peat = np.zeros((height, width), dtype=np.float32)
    fossil = np.zeros((height, width), dtype=np.float32)
    
    # Noise for fossil fuel deposits
    fossil_noise = generate_resource_noise(width, height, seed + 4001, scale=45)
    
    for y in range(height):
        for x in range(width):
            if not is_land[y, x]:
                continue
            
            biome = biomes[y, x]
            moist = moisture[y, x]
            fert = soil_fertility[y, x]
            wood = wood_biomass[y, x]
            coal = coal_potential[y, x]
            
            # Biomass fuel: correlates with wood biomass
            biomass[y, x] = wood * 0.8 + fert * 0.2
            
            # Peat: wetlands with high organic accumulation
            if biome in ('swamp', 'marsh'):
                peat_base = 0.6 + moist * 0.3 + fert * 0.1
            elif biome == 'mangrove':
                peat_base = 0.4 + moist * 0.2
            elif moist > 0.75:
                peat_base = (moist - 0.75) * 2
            else:
                peat_base = 0
            peat[y, x] = np.clip(peat_base, 0, 1)
            
            # Fossil fuel: sedimentary basins, correlated with coal but different distribution
            sedimentary = BIOME_SEDIMENTARY.get(biome, 0.2)
            fossil_base = sedimentary * 0.4 + coal * 0.3
            fossil[y, x] = np.clip(fossil_base * fossil_noise[y, x] * 1.5, 0, 1)
    
    return {
        'biomass_fuel_potential': biomass,
        'peat_potential': peat,
        'fossil_fuel_potential': fossil
    }


def compute_resource_accessibility(slope, roughness, elevation, biomes, 
                                    dist_river, is_land, width, height):
    """
    Compute accessibility factors for resource extraction.
    
    Returns dict of 2D arrays:
    - mineral_accessibility: Ease of mining operations
    - water_accessibility: Ease of water access
    """
    mineral_acc = np.zeros((height, width), dtype=np.float32)
    water_acc = np.zeros((height, width), dtype=np.float32)
    
    for y in range(height):
        for x in range(width):
            if not is_land[y, x]:
                continue
            
            slp = slope[y, x]
            rough = roughness[y, x]
            elev = elevation[y, x]
            biome = biomes[y, x]
            river_dist = dist_river[y, x]
            
            # Mineral accessibility: reduced by steep slopes, high roughness
            # River valleys improve accessibility
            slope_penalty = slp * 0.4
            roughness_penalty = rough * 0.3
            river_bonus = max(0, 1 - river_dist / 10) * 0.2
            
            # Dense forests reduce accessibility
            if biome in ('rainforest', 'tropical_forest', 'swamp', 'mangrove'):
                vegetation_penalty = 0.2
            else:
                vegetation_penalty = 0
            
            mineral_acc[y, x] = np.clip(
                0.8 - slope_penalty - roughness_penalty - vegetation_penalty + river_bonus,
                0.1, 1
            )
            
            # Water accessibility: proximity to rivers/lakes, terrain
            river_access = max(0, 1 - river_dist / 8)
            terrain_factor = 1 - slp * 0.3
            
            water_acc[y, x] = np.clip(river_access * 0.6 + terrain_factor * 0.4, 0, 1)
    
    return {
        'mineral_accessibility': mineral_acc,
        'water_accessibility': water_acc
    }


def compute_tile_environment(elevation, slope, roughness, moisture, temperature,
                              dist_river, dist_lake, dist_ocean, dist_coast,
                              biomes, is_land, is_mountain, is_river, is_lake,
                              width, height, seed):
    """
    Master function to compute complete tile environment profiles.
    
    Returns a dict containing all environmental resource maps:
    - geological_resources: iron, copper, tin, gold, coal, minerals, rock
    - soil_properties: quantity, fertility, sediment
    - water_resources: fresh, salt, wild, groundwater
    - biological_capacity: wood, animal, plant, insect
    - climate_exposure: rain, sunlight
    - fuel_resources: biomass, peat, fossil
    - accessibility: mineral, water
    """
    print("  Computing geological resources...")
    geo = compute_geological_resources(
        elevation, slope, roughness, is_land, is_mountain,
        dist_river, biomes, width, height, seed
    )
    
    print("  Computing soil properties...")
    soil = compute_soil_properties(
        elevation, slope, roughness, moisture, temperature,
        dist_river, biomes, is_land, width, height, seed
    )
    
    print("  Computing water resources...")
    water = compute_water_resources(
        moisture, dist_river, dist_lake, dist_ocean,
        is_land, is_river, is_lake, elevation, biomes,
        width, height, seed
    )
    
    print("  Computing biological capacity...")
    bio = compute_biological_capacity(
        biomes, soil['soil_fertility'], moisture, temperature,
        water['fresh_water'], is_land, width, height, seed
    )
    
    print("  Computing climate exposure...")
    climate = compute_climate_exposure(
        moisture, temperature, slope, elevation,
        is_land, biomes, width, height, seed
    )
    
    print("  Computing fuel resources...")
    fuel = compute_fuel_resources(
        bio['wood_biomass_capacity'], biomes, moisture, soil['soil_fertility'],
        geo['coal_potential'], is_land, width, height, seed
    )
    
    print("  Computing resource accessibility...")
    access = compute_resource_accessibility(
        slope, roughness, elevation, biomes,
        dist_river, is_land, width, height
    )
    
    # Combine all into unified structure
    return {
        'geological': geo,
        'soil': soil,
        'water': water,
        'biological': bio,
        'climate': climate,
        'fuel': fuel,
        'accessibility': access
    }


def get_tile_environment_data(tile_env, y, x):
    """
    Extract environment data for a single tile as a flat dictionary.
    For DataFrame integration.
    """
    return {
        # Geological resources
        'iron_potential': round(tile_env['geological']['iron_potential'][y, x], 4),
        'copper_potential': round(tile_env['geological']['copper_potential'][y, x], 4),
        'tin_potential': round(tile_env['geological']['tin_potential'][y, x], 4),
        'gold_potential': round(tile_env['geological']['gold_potential'][y, x], 4),
        'coal_potential': round(tile_env['geological']['coal_potential'][y, x], 4),
        'mineral_potential': round(tile_env['geological']['mineral_potential'][y, x], 4),
        'rock_quantity': round(tile_env['geological']['rock_quantity'][y, x], 4),
        
        # Soil properties
        'soil_quantity': round(tile_env['soil']['soil_quantity'][y, x], 4),
        'soil_fertility': round(tile_env['soil']['soil_fertility'][y, x], 4),
        'sediment_accumulation': round(tile_env['soil']['sediment_accumulation'][y, x], 4),
        
        # Water resources
        'fresh_water': round(tile_env['water']['fresh_water'][y, x], 4),
        'salt_water': round(tile_env['water']['salt_water'][y, x], 4),
        'wild_water': round(tile_env['water']['wild_water'][y, x], 4),
        'groundwater_potential': round(tile_env['water']['groundwater_potential'][y, x], 4),
        
        # Biological capacity
        'wood_biomass_capacity': round(tile_env['biological']['wood_biomass_capacity'][y, x], 4),
        'animal_habitat_capacity': round(tile_env['biological']['animal_habitat_capacity'][y, x], 4),
        'plant_diversity_capacity': round(tile_env['biological']['plant_diversity_capacity'][y, x], 4),
        'insect_population_capacity': round(tile_env['biological']['insect_population_capacity'][y, x], 4),
        
        # Climate exposure
        'rain_exposure': round(tile_env['climate']['rain_exposure'][y, x], 4),
        'sunlight_exposure': round(tile_env['climate']['sunlight_exposure'][y, x], 4),
        
        # Fuel resources
        'biomass_fuel_potential': round(tile_env['fuel']['biomass_fuel_potential'][y, x], 4),
        'peat_potential': round(tile_env['fuel']['peat_potential'][y, x], 4),
        'fossil_fuel_potential': round(tile_env['fuel']['fossil_fuel_potential'][y, x], 4),
        
        # Accessibility
        'mineral_accessibility': round(tile_env['accessibility']['mineral_accessibility'][y, x], 4),
        'water_accessibility': round(tile_env['accessibility']['water_accessibility'][y, x], 4),
    }


# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================
def generate_world(seed):
    """
    Generate a complete world from a seed.
    Returns DataFrame and saves to CSV.
    """
    print(f"="*60)
    print(f"GENERATING WORLD WITH SEED: {seed}")
    print(f"="*60)
    
    width, height = WORLD_WIDTH, WORLD_HEIGHT
    total_steps = 16
    
    def step_header(step_num, label):
        print(f"\n[{step_num:02d}/{total_steps}] {label}")
    
    # Initialize seeded generators
    rng = np.random.default_rng(seed)
    random.seed(seed)
    
    step_header(1, "Generating tectonic plates...")
    plate_map, plate_centers, plate_types, plate_vectors, num_plates = generate_tectonic_plates(width, height, seed)
    boundaries, convergent = determine_boundary_type(plate_map, plate_vectors, width, height)
    print(f"  - Created {num_plates} tectonic plates")
    print(f"  - Plate types: {plate_types}")
    
    step_header(2, "Generating elevation...")
    elevation = generate_elevation(width, height, seed, plate_map, convergent, plate_types, num_plates)
    
    step_header(3, "Enforcing 50% land / 50% ocean...")
    is_land, sea_level = enforce_land_ocean_ratio(elevation, LAND_PERCENT)
    
    step_header(4, "Validating ocean connectivity (no inland oceans)...")
    is_land, inland_ocean_count = flood_fill_ocean_validation(is_land, width, height)
    land_count = is_land.sum()
    ocean_count = (~is_land).sum()
    print(f"  - Converted {inland_ocean_count} inland ocean tiles to land")
    print(f"  - Land tiles: {land_count} ({land_count/TOTAL_TILES*100:.2f}%)")
    print(f"  - Ocean tiles: {ocean_count} ({ocean_count/TOTAL_TILES*100:.2f}%)")
    
    step_header(5, "Identifying mountains (10% of land)...")
    is_mountain = identify_mountains(elevation, is_land, convergent, MOUNTAIN_PERCENT_OF_LAND)
    mountain_count = is_mountain.sum()
    print(f"  - Mountain tiles: {mountain_count} ({mountain_count/land_count*100:.2f}% of land)")
    
    step_header(6, "Computing distance from ocean...")
    dist_ocean = compute_distance_from_ocean(is_land, width, height)
    print(f"  - Max distance from ocean: {dist_ocean.max():.1f} tiles")
    
    # Generate river system FIRST (before lakes) for complete river networks
    step_header(7, "Generating realistic river network...")
    is_river, river_ids, river_map, river_width, river_stats, elevation, river_hierarchy, headwater_types, river_moisture_intensity = generate_rivers(
        width, height, seed, elevation, is_land, is_mountain, None, None, dist_ocean
    )
    print(f"  - Total river tiles: {river_stats['total_tiles']}")
    print(f"  - Number of rivers: {river_stats['num_rivers']}")
    print(f"  - Rivers reaching ocean: {river_stats['rivers_to_ocean']}")
    print(f"  - Rivers merged with others: {river_stats['merged_rivers']}")
    print(f"  - Rivers ending inland: {river_stats['rivers_to_inland']}")
    print(f"  - Terrain carves: {river_stats['terrain_carves']}")
    if river_stats['river_lengths']:
        print(f"  - River lengths (top 5): {river_stats['river_lengths'][:5]}")
    if river_stats.get('hierarchy_counts'):
        hc = river_stats['hierarchy_counts']
        print(f"  - River hierarchy: {hc.get('stream', 0)} streams, {hc.get('river', 0)} rivers, {hc.get('major', 0)} major rivers")
    if river_stats.get('headwater_types'):
        print(f"  - Headwater types: {river_stats['headwater_types']}")
    
    # Generate lakes AFTER rivers - lakes can overlap rivers (as reservoirs)
    step_header(8, "Generating lakes (targeting 2-5% of land)...")
    is_lake, lake_ids, lake_sizes = generate_lakes(width, height, seed, elevation, is_land, is_mountain, dist_ocean)
    
    # Shrink lakes that touch rivers to create widened river sections
    is_lake, lake_ids, lakes_shrunk, tiles_removed = shrink_river_connected_lakes(
        is_lake, lake_ids, is_river, width, height, max_river_lake_width=2
    )
    
    lake_count = is_lake.sum()
    num_lakes = len(lake_sizes)
    # Count lakes that overlap rivers (now should be thinner "river widenings")
    reservoir_tiles = (is_lake & is_river).sum()
    print(f"  - Lake tiles: {lake_count} ({lake_count/land_count*100:.2f}% of land, target: {LAKE_PERCENT_MIN*100:.0f}-{LAKE_PERCENT_MAX*100:.0f}%)")
    print(f"  - Number of lakes: {num_lakes}")
    print(f"  - River-connected lakes shrunk: {lakes_shrunk} (removed {tiles_removed} tiles)")
    print(f"  - Widened river tiles (lake on river): {reservoir_tiles}")
    print(f"  - Coast buffer: {LAKE_COAST_BUFFER} tiles")
    print(f"  - Min lake size: {MIN_LAKE_SIZE} tiles")
    if lake_sizes:
        print(f"  - Original lake sizes: min={min(lake_sizes)}, max={max(lake_sizes)}, avg={np.mean(lake_sizes):.1f}")
    
    step_header(9, "Computing distance fields for rivers and lakes...")
    dist_river = compute_distance_from_rivers(is_river, width, height)
    dist_lake = compute_distance_from_lakes(is_lake, width, height)
    print(f"  - Max distance from river: {dist_river.max():.1f} tiles")
    print(f"  - Max distance from lake: {dist_lake.max():.1f} tiles")
    
    step_header(10, "Computing derived terrain maps (slope, roughness, coast distance)...")
    slope = compute_slope_map(elevation, width, height)
    roughness = compute_roughness_map(elevation, width, height)
    dist_coast = compute_distance_to_coast(is_land, width, height)
    aspect, grad_x, grad_y = compute_slope_direction(elevation, width, height)
    print(f"  - Slope range: {slope.min():.3f} - {slope.max():.3f}")
    print(f"  - Roughness range: {roughness.min():.3f} - {roughness.max():.3f}")
    print(f"  - Max distance from coast: {dist_coast.max():.1f} tiles")
    
    step_header(11, "Generating natural temperature simulation...")
    temperature = generate_natural_temperature(
        width, height, seed, elevation, is_land, is_mountain,
        dist_ocean, dist_river, dist_lake, slope, aspect
    )
    temp_land = temperature[is_land]
    print(f"  - Temperature range on land: {temp_land.min():.3f} - {temp_land.max():.3f}")
    
    step_header(12, "Generating natural moisture simulation (with wind & rain shadow)...")
    moisture, wind_dir, rain_shadow = generate_natural_moisture(
        width, height, seed, elevation, is_land, is_mountain,
        dist_ocean, dist_river, dist_lake, temperature, slope
    )
    moist_land = moisture[is_land]
    print(f"  - Moisture range on land: {moist_land.min():.3f} - {moist_land.max():.3f}")
    print(f"  - Prevailing wind direction: ({wind_dir[0]:.2f}, {wind_dir[1]:.2f})")
    
    # Report rain shadow coverage
    rain_shadow_land = rain_shadow[is_land]
    shadow_tiles = (rain_shadow_land > 0.3).sum()
    shadow_pct = shadow_tiles / is_land.sum() * 100 if is_land.sum() > 0 else 0
    print(f"  - Rain shadow affected tiles: {shadow_tiles} ({shadow_pct:.1f}% of land)")
    
    step_header(13, "Applying river climate effects...")
    moisture, temperature = apply_watershed_climate_effects(
        river_map, is_river, moisture, temperature, is_land, elevation, width, height, 
        river_hierarchy=river_hierarchy, river_moisture_intensity=river_moisture_intensity
    )
    moist_land = moisture[is_land]
    temp_land = temperature[is_land]
    print(f"  - Final moisture range on land: {moist_land.min():.3f} - {moist_land.max():.3f}")
    print(f"  - Final temperature range on land: {temp_land.min():.3f} - {temp_land.max():.3f}")
    
    step_header(14, "Generating islands...")
    is_island = generate_islands(width, height, seed, is_land, elevation)
    island_count = is_island.sum()
    print(f"  - Island tiles: {island_count} ({island_count/ocean_count*100:.2f}% of ocean)")
    
    is_land_with_islands = is_land | is_island
    
    step_header(15, "Assigning biomes with expanded classification system...")
    # Use the new expanded biome system
    biomes = assign_biomes_expanded(
        width, height, seed, is_land, is_mountain, is_lake, is_river,
        temperature, moisture, elevation, slope, roughness,
        dist_river, dist_coast, is_island
    )
    
    # Count biome distribution
    land_mask = is_land | is_island
    biome_counts = {}
    for y in range(height):
        for x in range(width):
            if land_mask[y, x] and not is_lake[y, x]:
                b = biomes[y, x]
                biome_counts[b] = biome_counts.get(b, 0) + 1
    
    total_land_biomes = sum(biome_counts.values())
    print(f"  - Expanded biome types used: {len(biome_counts)}")
    
    # Show top biomes
    sorted_biomes = sorted(biome_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    for biome, count in sorted_biomes:
        pct = count / total_land_biomes * 100 if total_land_biomes > 0 else 0
        print(f"    {biome}: {count} ({pct:.1f}%)")
    
    step_header(16, "Computing tile environment resources...")
    tile_env = compute_tile_environment(
        elevation, slope, roughness, moisture, temperature,
        dist_river, dist_lake, dist_ocean, dist_coast,
        biomes, is_land, is_mountain, is_river, is_lake,
        width, height, seed
    )
    
    print("\nAssigning region IDs...")
    region_x, region_y = assign_regions(width, height)
    
    # Build DataFrame (vectorized)
    print("\nBuilding DataFrame...")
    yy, xx = np.indices((height, width))
    is_headwater = np.zeros((height, width), dtype=bool)
    for hy, hx in headwater_types:
        if 0 <= hy < height and 0 <= hx < width:
            is_headwater[hy, hx] = True

    df_dict = {
        'x': xx.ravel(),
        'y': yy.ravel(),
        'region_id_x': region_x.ravel(),
        'region_id_y': region_y.ravel(),
        'plate_id': plate_map.ravel(),
        'elevation': np.round(elevation, 6).ravel(),
        'temperature': np.round(temperature, 6).ravel(),
        'moisture': np.round(moisture, 6).ravel(),
        'dist_ocean': np.round(dist_ocean, 2).ravel(),
        'dist_river': np.round(dist_river, 2).ravel(),
        'dist_lake': np.round(dist_lake, 2).ravel(),
        'biome': biomes.ravel(),
        'is_land': is_land_with_islands.ravel(),
        'is_mountain': is_mountain.ravel(),
        'is_river': is_river.ravel(),
        'is_lake': is_lake.ravel(),
        'is_headwater': is_headwater.ravel(),
        'river_id': river_ids.ravel(),
        'lake_id': lake_ids.ravel(),
    }

    env_cols = {
        'iron_potential': tile_env['geological']['iron_potential'],
        'copper_potential': tile_env['geological']['copper_potential'],
        'tin_potential': tile_env['geological']['tin_potential'],
        'gold_potential': tile_env['geological']['gold_potential'],
        'coal_potential': tile_env['geological']['coal_potential'],
        'mineral_potential': tile_env['geological']['mineral_potential'],
        'rock_quantity': tile_env['geological']['rock_quantity'],
        'soil_quantity': tile_env['soil']['soil_quantity'],
        'soil_fertility': tile_env['soil']['soil_fertility'],
        'sediment_accumulation': tile_env['soil']['sediment_accumulation'],
        'fresh_water': tile_env['water']['fresh_water'],
        'salt_water': tile_env['water']['salt_water'],
        'wild_water': tile_env['water']['wild_water'],
        'groundwater_potential': tile_env['water']['groundwater_potential'],
        'wood_biomass_capacity': tile_env['biological']['wood_biomass_capacity'],
        'animal_habitat_capacity': tile_env['biological']['animal_habitat_capacity'],
        'plant_diversity_capacity': tile_env['biological']['plant_diversity_capacity'],
        'insect_population_capacity': tile_env['biological']['insect_population_capacity'],
        'rain_exposure': tile_env['climate']['rain_exposure'],
        'sunlight_exposure': tile_env['climate']['sunlight_exposure'],
        'biomass_fuel_potential': tile_env['fuel']['biomass_fuel_potential'],
        'peat_potential': tile_env['fuel']['peat_potential'],
        'fossil_fuel_potential': tile_env['fuel']['fossil_fuel_potential'],
        'mineral_accessibility': tile_env['accessibility']['mineral_accessibility'],
        'water_accessibility': tile_env['accessibility']['water_accessibility'],
    }
    for col, arr in env_cols.items():
        df_dict[col] = np.round(arr, 4).ravel()

    df = pd.DataFrame(df_dict)
    
    # Validation
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    
    total_tiles = len(df)
    land_tiles = df['is_land'].sum()
    ocean_tiles = total_tiles - land_tiles
    
    print(f"\nWorld Size: {width}x{height} = {total_tiles} tiles")
    print(f"\nLand/Ocean Distribution:")
    print(f"  - Total Land: {land_tiles} ({land_tiles/total_tiles*100:.2f}%)")
    print(f"  - Total Ocean: {ocean_tiles} ({ocean_tiles/total_tiles*100:.2f}%)")
    
    # Ocean connectivity validation
    print(f"\nOcean Connectivity: VALIDATED (no inland ocean patches)")
    
    # Expanded Biome breakdown
    print(f"\nExpanded Biome Distribution (on effective land - excludes lakes):")
    land_df = df[df['is_land'] & (df['biome'] != 'lake')]
    effective_land_count = len(land_df)
    
    biome_counts = land_df['biome'].value_counts()
    
    # Group biomes by category
    biome_groups = {
        'Mountains': ['rocky_mountains', 'snow_mountains', 'forest_mountains', 'alpine_meadows', 'glacier'],
        'Hills': ['grassy_hills', 'forest_hills', 'rocky_hills', 'snow_hills'],
        'Plains': ['grassland', 'meadow', 'steppe', 'savanna'],
        'Forests': ['temperate_forest', 'woodland', 'tropical_forest', 'rainforest'],
        'Deserts': ['sand_desert', 'rock_desert', 'badlands', 'oasis'],
        'Snow': ['snow_plains', 'snow_forest'],
        'Wetlands': ['swamp', 'marsh', 'mangrove'],
    }
    
    for group_name, group_biomes in biome_groups.items():
        group_total = sum(biome_counts.get(b, 0) for b in group_biomes)
        group_pct = group_total / effective_land_count * 100 if effective_land_count > 0 else 0
        print(f"\n  {group_name}: {group_total} tiles ({group_pct:.1f}%)")
        for biome in group_biomes:
            count = biome_counts.get(biome, 0)
            if count > 0:
                pct = count / effective_land_count * 100
                print(f"    - {biome:20s}: {count:5d} ({pct:5.2f}%)")
    
    # Biome cluster statistics
    print(f"\nBiome Cluster Statistics (top 10 by count):")
    effective_land_mask = (is_land | is_island) & ~is_lake
    cluster_stats = calculate_biome_cluster_stats(biomes, effective_land_mask)
    sorted_stats = sorted(cluster_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
    for biome, stats in sorted_stats:
        if stats['count'] > 0:
            print(f"  - {biome:20s}: {stats['clusters']:3d} clusters, "
                  f"sizes: min={stats['min_size']}, max={stats['max_size']}, avg={stats['avg_size']:.1f}")
    
    # Lakes
    lake_tiles_count = df['is_lake'].sum()
    lake_pct = lake_tiles_count / land_tiles * 100 if land_tiles > 0 else 0
    lake_status = "OK" if LAKE_PERCENT_MIN*100 <= lake_pct <= LAKE_PERCENT_MAX*100 else "OUTSIDE TARGET"
    print(f"\nLake Coverage: {lake_tiles_count} tiles ({lake_pct:.2f}% of land) [target: {LAKE_PERCENT_MIN*100:.0f}-{LAKE_PERCENT_MAX*100:.0f}%] {lake_status}")
    print(f"Lake Count: {num_lakes} lakes")
    if lake_sizes:
        print(f"Lake Sizes: min={min(lake_sizes)}, max={max(lake_sizes)}, avg={np.mean(lake_sizes):.1f}")
    
    # Mountains
    mountain_tiles_count = df['is_mountain'].sum()
    original_land = is_land.sum()
    mountain_pct = mountain_tiles_count / original_land * 100 if original_land > 0 else 0
    print(f"\nMountain Coverage: {mountain_tiles_count} tiles ({mountain_pct:.2f}% of original land) [target: 10%]")
    
    # River System Summary
    print(f"\nRIVER NETWORK SUMMARY:")
    print(f"  - Total river tiles: {river_stats['total_tiles']}")
    print(f"  - River segments: {river_stats['num_rivers']}")
    print(f"  - Rivers reaching ocean: {river_stats['rivers_to_ocean']}")
    print(f"  - Rivers ending inland: {river_stats['rivers_to_inland']}")
    print(f"  - Max flow accumulation: {river_stats['max_flow']:.1f}")
    print(f"  - Average river width: {river_stats['avg_width']:.2f}")
    print(f"  - Terrain carves: {river_stats['terrain_carves']}")
    if river_stats['river_lengths']:
        longest = max(river_stats['river_lengths']) if river_stats['river_lengths'] else 0
        print(f"  - Longest river segment: {longest} tiles")
        print(f"  - Top segment lengths: {river_stats['river_lengths'][:5]}")
    
    # Islands
    island_tiles_count = is_island.sum()
    print(f"Islands: {island_tiles_count} tiles ({island_tiles_count/ocean_count*100:.2f}% of ocean) [target: 0-5%]")
    
    # Environment Resources Summary
    print(f"\nENVIRONMENT RESOURCES SUMMARY:")
    land_df = df[df['is_land']]
    if len(land_df) > 0:
        # Geological resources overview
        high_iron = (land_df['iron_potential'] > 0.5).sum()
        high_copper = (land_df['copper_potential'] > 0.5).sum()
        high_gold = (land_df['gold_potential'] > 0.5).sum()
        high_coal = (land_df['coal_potential'] > 0.5).sum()
        print(f"  Geological (high potential tiles):")
        print(f"    - Iron: {high_iron}, Copper: {high_copper}, Gold: {high_gold}, Coal: {high_coal}")
        
        # Soil & fertility
        fertile_tiles = (land_df['soil_fertility'] > 0.5).sum()
        print(f"  Soil:")
        print(f"    - Fertile land (>50%): {fertile_tiles} tiles ({fertile_tiles/len(land_df)*100:.1f}%)")
        
        # Water resources
        fresh_water_tiles = (land_df['fresh_water'] > 0.5).sum()
        groundwater_tiles = (land_df['groundwater_potential'] > 0.5).sum()
        print(f"  Water:")
        print(f"    - Fresh water access: {fresh_water_tiles}, Groundwater: {groundwater_tiles}")
        
        # Biological capacity
        high_wood = (land_df['wood_biomass_capacity'] > 0.5).sum()
        high_habitat = (land_df['animal_habitat_capacity'] > 0.5).sum()
        print(f"  Biological:")
        print(f"    - High wood biomass: {high_wood}, High habitat: {high_habitat}")
        
        # Fuel resources
        high_peat = (land_df['peat_potential'] > 0.3).sum()
        high_fossil = (land_df['fossil_fuel_potential'] > 0.3).sum()
        print(f"  Fuel:")
        print(f"    - Peat deposits: {high_peat}, Fossil fuel: {high_fossil}")
    
    # Save to CSV
    filename = f"world_seed_{seed}.csv"
    df.to_csv(filename, index=False)
    print(f"\nWorld saved to: {filename}")
    
    return df


# =============================================================================
# VISUALIZATION (Optional)
# =============================================================================
def add_scale_bar(ax, width, height, tiles_per_unit=10, unit_name='tiles'):
    """Add a scale bar to the map."""
    from matplotlib.patches import Rectangle
    from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox
    
    # Scale bar size (in tiles)
    scale_length = 20  # 20 tiles
    
    # Position at bottom-left
    bar_x = width * 0.05
    bar_y = height * 0.95
    bar_height = height * 0.015
    
    # Draw scale bar background
    ax.add_patch(Rectangle((bar_x - 2, bar_y - bar_height - 2), scale_length + 4, bar_height + 10,
                           facecolor='white', edgecolor='black', alpha=0.8, zorder=10))
    
    # Draw scale bar
    ax.add_patch(Rectangle((bar_x, bar_y), scale_length, bar_height,
                           facecolor='black', edgecolor='black', zorder=11))
    ax.add_patch(Rectangle((bar_x, bar_y), scale_length/2, bar_height,
                           facecolor='white', edgecolor='black', zorder=11))
    
    # Add scale text
    ax.text(bar_x + scale_length/2, bar_y + bar_height + 3, f'{scale_length} {unit_name}',
            ha='center', va='bottom', fontsize=8, fontweight='bold', zorder=12)


def visualize_world(df, seed, save_image=True):
    """Create a visualization of the generated world with rivers in RED."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for visualization.")
        return
    
    width, height = WORLD_WIDTH, WORLD_HEIGHT
    
    # Use expanded biome colors (with fallback to legacy)
    biome_colors = EXPANDED_BIOME_COLORS.copy()
    
    x_vals = df['x'].to_numpy(dtype=np.int32)
    y_vals = df['y'].to_numpy(dtype=np.int32)
    biome_vals = df['biome'].to_numpy()
    elev_vals = df['elevation'].to_numpy(dtype=np.float32)
    river_vals = df['is_river'].to_numpy(dtype=bool)

    # Create biome/attribute image
    img = np.zeros((height, width, 3), dtype=np.float32)
    base_colors = np.array([mcolors.to_rgb(biome_colors.get(b, '#000000')) for b in biome_vals], dtype=np.float32)
    brightness = (0.7 + 0.3 * elev_vals)[:, None]
    shaded_colors = np.clip(base_colors * brightness, 0.0, 1.0)
    shaded_colors[river_vals] = mcolors.to_rgb('#1e90ff')  # Dodger blue
    img[y_vals, x_vals] = shaded_colors
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Biome/Attribute map with scale
    axes[0, 0].imshow(img)
    axes[0, 0].set_title(f'World Map (Seed: {seed})')
    axes[0, 0].axis('off')
    add_scale_bar(axes[0, 0], width, height)
    
    # Biome Atlas (pure biomes without elevation adjustment)
    atlas_img = np.zeros((height, width, 3), dtype=np.float32)
    atlas_colors = np.array([mcolors.to_rgb(biome_colors.get(b, '#000000')) for b in biome_vals], dtype=np.float32)
    atlas_colors[river_vals] = mcolors.to_rgb('#1e90ff')  # Dodger blue rivers
    atlas_img[y_vals, x_vals] = atlas_colors
    
    axes[0, 1].imshow(atlas_img)
    axes[0, 1].set_title('Biome Atlas Map')
    axes[0, 1].axis('off')
    add_scale_bar(axes[0, 1], width, height)
    
    # Add color legend for expanded biomes (grouped)
    from matplotlib.patches import Patch
    legend_elements = [
        # Water
        Patch(facecolor='#1a5276', label='Ocean'),
        Patch(facecolor='#3498db', label='Lake'),
        Patch(facecolor='#1e90ff', label='River'),
        # Plains
        Patch(facecolor='#90c965', label='Grassland'),
        Patch(facecolor='#7dcea0', label='Meadow'),
        Patch(facecolor='#c4b896', label='Steppe'),
        Patch(facecolor='#d4b86a', label='Savanna'),
        # Forests
        Patch(facecolor='#228b22', label='Temperate Forest'),
        Patch(facecolor='#6b8e23', label='Woodland'),
        Patch(facecolor='#006400', label='Tropical Forest'),
        Patch(facecolor='#004d00', label='Rainforest'),
        # Deserts
        Patch(facecolor='#f4d03f', label='Sand Desert'),
        Patch(facecolor='#a0522d', label='Rock Desert'),
        Patch(facecolor='#cd853f', label='Badlands'),
        Patch(facecolor='#32cd32', label='Oasis'),
        # Snow
        Patch(facecolor='#e8f4f8', label='Snow Plains'),
        Patch(facecolor='#a8d8ea', label='Snow Forest'),
        Patch(facecolor='#b8c9d4', label='Snow Hills'),
        Patch(facecolor='#e0ffff', label='Glacier'),
        # Hills
        Patch(facecolor='#5c8a4d', label='Grassy Hills'),
        Patch(facecolor='#2d5a3f', label='Forest Hills'),
        Patch(facecolor='#6b5344', label='Rocky Hills'),
        # Mountains
        Patch(facecolor='#5d6d7e', label='Rocky Mountains'),
        Patch(facecolor='#d5d8dc', label='Snow Mountains'),
        Patch(facecolor='#3d5c4f', label='Forest Mountains'),
        Patch(facecolor='#8fbc8f', label='Alpine Meadows'),
        # Wetlands
        Patch(facecolor='#4a5d23', label='Swamp'),
        Patch(facecolor='#6b7b3a', label='Marsh'),
        Patch(facecolor='#3d5e3a', label='Mangrove'),
    ]
    # Only show biomes that exist in this world
    unique_biomes = df['biome'].unique()
    legend_elements = [p for p in legend_elements if p.get_label().lower().replace(' ', '_') in 
                      [b for b in unique_biomes] or p.get_label() in ['Ocean', 'Lake', 'River']]
    axes[0, 1].legend(handles=legend_elements, loc='upper left', fontsize=6, 
                       framealpha=0.9, ncol=3, bbox_to_anchor=(0.0, 1.0))
    
    # Temperature map with scale
    temp_img = np.zeros((height, width), dtype=np.float32)
    temp_img[y_vals, x_vals] = df['temperature'].to_numpy(dtype=np.float32)
    
    im = axes[1, 0].imshow(temp_img, cmap='RdYlBu_r')
    axes[1, 0].set_title('Temperature')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    add_scale_bar(axes[1, 0], width, height)
    
    # Moisture map with scale
    moisture_img = np.zeros((height, width), dtype=np.float32)
    moisture_img[y_vals, x_vals] = df['moisture'].to_numpy(dtype=np.float32)
    
    im = axes[1, 1].imshow(moisture_img, cmap='YlGnBu')
    axes[1, 1].set_title('Moisture')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
    add_scale_bar(axes[1, 1], width, height)
    
    plt.tight_layout()
    
    if save_image:
        plt.savefig(f'world_seed_{seed}.png', dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: world_seed_{seed}.png")
    
    plt.close(fig)
    
    # Generate Atlas Map with Grid (separate image)
    generate_atlas_map(df, seed, biome_colors, save_image)


def generate_atlas_map(df, seed, biome_colors, save_image=True):
    """
    Generate a separate atlas map image with grid lines.
    Only shows the map with colors and grids, no scale or other annotations.
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    width, height = WORLD_WIDTH, WORLD_HEIGHT
    
    # Create atlas image
    x_vals = df['x'].to_numpy(dtype=np.int32)
    y_vals = df['y'].to_numpy(dtype=np.int32)
    biome_vals = df['biome'].to_numpy()
    river_vals = df['is_river'].to_numpy(dtype=bool)

    atlas_img = np.zeros((height, width, 3), dtype=np.float32)
    atlas_colors = np.array([mcolors.to_rgb(biome_colors.get(b, '#000000')) for b in biome_vals], dtype=np.float32)
    atlas_colors[river_vals] = mcolors.to_rgb('#1e90ff')
    atlas_img[y_vals, x_vals] = atlas_colors
    
    # Create figure with no margins
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Display the map
    ax.imshow(atlas_img)
    
    # Add grid lines for all 200x200 tiles
    for x in range(0, width + 1):
        ax.axvline(x - 0.5, color='black', linewidth=0.2, alpha=0.6)
    
    for y in range(0, height + 1):
        ax.axhline(y - 0.5, color='black', linewidth=0.2, alpha=0.6)
    
    # Remove axes and all annotations
    ax.axis('off')
    
    # Set tight layout with no padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    if save_image:
        plt.savefig(f'world_seed_{seed}_atlas.png', dpi=200, bbox_inches='tight', pad_inches=0)
        print(f"Atlas map saved to: world_seed_{seed}_atlas.png")
    
    plt.close(fig)


def visualize_resources(df, seed, save_image=True):
    """
    Create a resource map visualization showing distribution of key resources.
    Generates a 3x2 grid showing:
    - Metal Ores (iron, copper, tin, gold combined)
    - Coal & Fossil Fuels
    - Wood Biomass
    - Water Resources (fresh + groundwater)
    - Soil Fertility
    - Overall Resource Richness
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for visualization.")
        return
    
    width, height = WORLD_WIDTH, WORLD_HEIGHT
    
    # Initialize arrays for each resource category
    metal_ores = np.zeros((height, width))
    coal_fossil = np.zeros((height, width))
    wood_biomass = np.zeros((height, width))
    water_resources = np.zeros((height, width))
    soil_fertility = np.zeros((height, width))
    overall_richness = np.zeros((height, width))
    is_water = np.zeros((height, width), dtype=bool)
    
    # Populate arrays from dataframe (vectorized)
    x_vals = df['x'].to_numpy(dtype=np.int32)
    y_vals = df['y'].to_numpy(dtype=np.int32)
    biome_vals = df['biome'].to_numpy()
    river_vals = df['is_river'].to_numpy(dtype=bool)

    is_water_vals = np.isin(biome_vals, ['ocean', 'lake']) | river_vals
    is_water[y_vals, x_vals] = is_water_vals

    iron_vals = df['iron_potential'].to_numpy(dtype=np.float32)
    copper_vals = df['copper_potential'].to_numpy(dtype=np.float32)
    tin_vals = df['tin_potential'].to_numpy(dtype=np.float32)
    gold_vals = df['gold_potential'].to_numpy(dtype=np.float32)
    coal_vals = df['coal_potential'].to_numpy(dtype=np.float32)
    fossil_vals = df['fossil_fuel_potential'].to_numpy(dtype=np.float32)
    peat_vals = df['peat_potential'].to_numpy(dtype=np.float32)
    wood_vals = df['wood_biomass_capacity'].to_numpy(dtype=np.float32)
    fresh_vals = df['fresh_water'].to_numpy(dtype=np.float32)
    ground_vals = df['groundwater_potential'].to_numpy(dtype=np.float32)
    soil_vals = df['soil_fertility'].to_numpy(dtype=np.float32)
    mineral_vals = df['mineral_potential'].to_numpy(dtype=np.float32)
    habitat_vals = df['animal_habitat_capacity'].to_numpy(dtype=np.float32)

    metal_vals = iron_vals * 0.3 + copper_vals * 0.3 + tin_vals * 0.2 + gold_vals * 0.2
    coal_fossil_vals = coal_vals * 0.5 + fossil_vals * 0.3 + peat_vals * 0.2
    water_vals = fresh_vals * 0.6 + ground_vals * 0.4
    overall_vals = (
        metal_vals * 0.2 +
        coal_fossil_vals * 0.15 +
        wood_vals * 0.2 +
        fresh_vals * 0.15 +
        soil_vals * 0.15 +
        mineral_vals * 0.1 +
        habitat_vals * 0.05
    )

    metal_ores[y_vals, x_vals] = metal_vals
    coal_fossil[y_vals, x_vals] = coal_fossil_vals
    wood_biomass[y_vals, x_vals] = wood_vals
    water_resources[y_vals, x_vals] = np.where(is_water_vals, 0.0, water_vals)
    soil_fertility[y_vals, x_vals] = soil_vals
    overall_richness[y_vals, x_vals] = overall_vals
    
    # Create masked arrays for water (to show water areas differently)
    water_mask = is_water
    
    # Custom colormaps
    from matplotlib.colors import LinearSegmentedColormap
    
    # Metal ores: dark gray to orange to gold
    metal_cmap = LinearSegmentedColormap.from_list('metal_ores', 
        ['#2c2c2c', '#4a4a4a', '#8b4513', '#cd853f', '#daa520', '#ffd700'])
    
    # Coal: dark to red-orange
    coal_cmap = LinearSegmentedColormap.from_list('coal', 
        ['#1a1a1a', '#333333', '#4d3319', '#8b4513', '#d2691e', '#ff4500'])
    
    # Wood: brown to dark green
    wood_cmap = LinearSegmentedColormap.from_list('wood', 
        ['#f5deb3', '#deb887', '#8fbc8f', '#228b22', '#006400', '#004d00'])
    
    # Water: light blue to deep blue
    water_cmap = LinearSegmentedColormap.from_list('water', 
        ['#f0f8ff', '#b0e0e6', '#87ceeb', '#4682b4', '#1e90ff', '#0066cc'])
    
    # Soil: tan to rich brown
    soil_cmap = LinearSegmentedColormap.from_list('soil', 
        ['#f5f5dc', '#d2b48c', '#bc8f8f', '#8b4513', '#654321', '#3d2314'])
    
    # Overall: purple gradient (low) to green (high)
    rich_cmap = LinearSegmentedColormap.from_list('richness', 
        ['#2c1810', '#4a3728', '#6b5b47', '#8fbc8f', '#32cd32', '#00ff00'])
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Resource Map (Seed: {seed})', fontsize=16, fontweight='bold')
    
    # Helper function to plot with water overlay
    def plot_resource(ax, data, cmap, title, mask_water=True):
        # Create display data
        display_data = data.copy()
        
        if mask_water:
            # Set water areas to NaN for masking
            display_data = np.where(is_water, np.nan, display_data)
        
        im = ax.imshow(display_data, cmap=cmap, vmin=0, vmax=1)
        
        # Overlay water areas in blue
        if mask_water:
            water_overlay = np.zeros((height, width, 4))
            water_overlay[is_water] = [0.1, 0.4, 0.7, 1.0]  # Semi-transparent blue
            ax.imshow(water_overlay)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        add_scale_bar(ax, width, height)
    
    # Plot each resource type
    plot_resource(axes[0, 0], metal_ores, metal_cmap, 'Metal Ore Deposits\n(Iron, Copper, Tin, Gold)')
    plot_resource(axes[0, 1], coal_fossil, coal_cmap, 'Coal & Fossil Fuels\n(Coal, Peat, Fossil)')
    plot_resource(axes[0, 2], wood_biomass, wood_cmap, 'Wood Biomass\n(Forest Resources)')
    plot_resource(axes[1, 0], water_resources, water_cmap, 'Fresh Water Access\n(Rivers, Groundwater)', mask_water=False)
    plot_resource(axes[1, 1], soil_fertility, soil_cmap, 'Soil Fertility\n(Agricultural Potential)')
    plot_resource(axes[1, 2], overall_richness, rich_cmap, 'Overall Resource Richness\n(Combined Score)')
    
    plt.tight_layout()
    
    if save_image:
        plt.savefig(f'world_seed_{seed}_resources.png', dpi=150, bbox_inches='tight')
        print(f"Resource map saved to: world_seed_{seed}_resources.png")
    
    plt.close(fig)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deterministic fantasy world generator")
    parser.add_argument("seed", nargs="?", type=int, default=42, help="World seed (default: 42)")
    parser.add_argument("--verbose-logs", action="store_true", help="Show all detailed generation logs")
    args = parser.parse_args()

    seed = args.seed
    configure_terminal_logging(verbose=args.verbose_logs)

    try:
        # Generate world
        df = generate_world(seed)

        # Auto-generate visualizations
        visualize_world(df, seed)
        visualize_resources(df, seed)

        print("\nWorld generation complete!")
    finally:
        restore_terminal_logging(show_summary=not args.verbose_logs)

