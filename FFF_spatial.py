"""
Flood Adaptation ABM - Spatial Module (v11)

Generates agent positions (x, y) and elevations (z).
Modes: 0=CSV, 1=random, 2=grid/connected grids.
Mode 2 with n_connectors=0 gives a plain grid.
"""

import numpy as np
import pandas as pd


# ============================================================================
# LAYOUT CONSTANTS FOR UNIT-SQUARE PLACEMENT
# ============================================================================
# Agents are placed in a [0,1] x [0,1] unit square.
# These constants control spacing and margins for grid layouts (Mode 2).

# On a regular grid with spacing s, diagonal distance = s * sqrt(2).
# To guarantee diagonal neighbors are within distance_threshold:
#   s * sqrt(2) < distance_threshold  =>  s < distance_threshold / sqrt(2)
# The safety factor keeps spacing just below this theoretical limit.
_DIAGONAL_SAFETY = 0.999

# Minimum margin between the layout boundary and the unit-square edge.
# Prevents agents from being placed at the exact domain boundary.
_MIN_MARGIN = 0.02

# Maximum usable extent of the unit square (width or height).
# Derived from margin: 1.0 - 2 * _MIN_MARGIN = 0.96.
_MAX_EXTENT = 1.0 - 2 * _MIN_MARGIN

# Final coordinate clipping bounds to keep all agents strictly
# inside the unit square after layout computation.
_COORD_MIN = 0.01
_COORD_MAX = 0.99


# ============================================================================
# LOCATION GENERATION FUNCTIONS
# ============================================================================

def generate_random_locations(n_agents, rng):
    """Generate agents at random positions (Mode 1)."""
    x = np.round(rng.uniform(0, 1, n_agents), 3)
    y = np.round(rng.uniform(0, 1, n_agents), 3)
    return x, y


def generate_connected_grid_neighborhoods(n_agents, distance_threshold,
                                          grid_rows, grid_cols, n_connectors):
    """
    Generate agents in grid neighborhoods (Mode 2).

    Agents are placed on sub-grids spaced so that within-neighborhood
    distances are less than distance_threshold (enabling network edges).

    If n_connectors=0, produces a single plain grid.
    If n_connectors>0, creates separate neighborhood grids linked by
    bridge agents between adjacent neighborhoods.
    """
    n_neighborhoods = grid_rows * grid_cols

    # Count connector agents between adjacent neighborhoods
    n_horizontal = grid_rows * (grid_cols - 1)
    n_vertical = (grid_rows - 1) * grid_cols
    total_connectors = (n_horizontal + n_vertical) * n_connectors

    # Distribute remaining agents across neighborhoods
    agents_in_grids = n_agents - total_connectors
    base_per_neighborhood = agents_in_grids // n_neighborhoods
    remainder = agents_in_grids % n_neighborhoods

    def get_grid_dims(n):
        nr = int(np.ceil(np.sqrt(n)))
        if nr % 2 == 0:
            nr += 1
        nc = int(np.ceil(n / nr))
        return nr, nc

    max_per_neighborhood = base_per_neighborhood + (1 if remainder > 0 else 0)
    nh_rows, nh_cols = get_grid_dims(max_per_neighborhood)

    # Grid spacing: ensures diagonal distance < distance_threshold
    spacing = _DIAGONAL_SAFETY * distance_threshold / np.sqrt(2)
    nh_width = (nh_cols - 1) * spacing
    nh_height = (nh_rows - 1) * spacing
    gap = (n_connectors + 1) * spacing

    # Total layout dimensions
    total_width = grid_cols * nh_width + (grid_cols - 1) * gap
    total_height = grid_rows * nh_height + (grid_rows - 1) * gap

    # Scale down if layout exceeds usable extent of unit square
    if total_width > _MAX_EXTENT or total_height > _MAX_EXTENT:
        scale = min(_MAX_EXTENT / total_width, _MAX_EXTENT / total_height)
        spacing *= scale
        nh_width = (nh_cols - 1) * spacing
        nh_height = (nh_rows - 1) * spacing
        gap = (n_connectors + 1) * spacing
        total_width = grid_cols * nh_width + (grid_cols - 1) * gap
        total_height = grid_rows * nh_height + (grid_rows - 1) * gap

    # Left-align horizontally so agents start near x=0 (low elevation);
    # center vertically within the unit square.
    margin_x = _MIN_MARGIN
    margin_y = max(_MIN_MARGIN, (1.0 - total_height) / 2)

    all_coords = []

    # Place agents in neighborhood sub-grids
    for gr in range(grid_rows):
        for gc in range(grid_cols):
            nh_idx = gr * grid_cols + gc
            n_in_this = base_per_neighborhood + (1 if nh_idx < remainder else 0)

            origin_x = margin_x + gc * (nh_width + gap)
            origin_y = margin_y + gr * (nh_height + gap)

            count = 0
            for row in range(nh_rows):
                for col in range(nh_cols):
                    if count >= n_in_this:
                        break
                    all_coords.append([origin_x + col * spacing,
                                       origin_y + row * spacing])
                    count += 1
                if count >= n_in_this:
                    break

    # Place horizontal connectors (between left-right adjacent neighborhoods)
    for gr in range(grid_rows):
        for gc in range(grid_cols - 1):
            left_origin_x = margin_x + gc * (nh_width + gap)
            left_origin_y = margin_y + gr * (nh_height + gap)
            right_edge_x = left_origin_x + nh_width
            middle_y = left_origin_y + nh_height / 2

            for c in range(n_connectors):
                cx = right_edge_x + (c + 1) * spacing
                all_coords.append([cx, middle_y])

    # Place vertical connectors (between top-bottom adjacent neighborhoods)
    for gr in range(grid_rows - 1):
        for gc in range(grid_cols):
            bottom_origin_x = margin_x + gc * (nh_width + gap)
            bottom_origin_y = margin_y + gr * (nh_height + gap)
            top_edge_y = bottom_origin_y + nh_height
            middle_x = bottom_origin_x + nh_width / 2

            for c in range(n_connectors):
                cy = top_edge_y + (c + 1) * spacing
                all_coords.append([middle_x, cy])

    # Clip coordinates to stay within unit-square bounds
    coords = np.clip(np.array(all_coords), _COORD_MIN, _COORD_MAX)
    return coords[:, 0], coords[:, 1]


def generate_elevation(x, slope, noise_factor, rng):
    """Generate elevation with linear gradient and noise."""
    z_base = slope * x
    noise_magnitude = noise_factor * slope
    noise = rng.uniform(-noise_magnitude, noise_magnitude, len(x))
    z = np.clip(z_base + noise, 0, slope)
    return z


# ============================================================================
# SPATIAL GENERATOR CLASS
# ============================================================================

class SpatialGenerator:
    """Generate positions and elevations for agents."""

    def __init__(self, n_agents, mode, distance_threshold,
                 grid_rows, grid_cols, n_connectors,
                 slope=1.0, noise_factor=0.05, csv_path=None, rng=None):
        self.n_agents = n_agents
        self.mode = mode
        self.distance_threshold = distance_threshold
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.n_connectors = n_connectors
        self.slope = slope
        self.noise_factor = noise_factor
        self.csv_path = csv_path
        self.rng = rng if rng is not None else np.random.default_rng()

    def generate(self):
        """
        Generate positions and elevations.

        Returns: (positions [n,2], elevations [n,])
        """
        if self.mode == 0:
            df = pd.read_csv(self.csv_path)
            x, y, z = df["x"].values, df["y"].values, df["z"].values
        elif self.mode == 1:
            x, y = generate_random_locations(self.n_agents, self.rng)
            z = generate_elevation(x, self.slope, self.noise_factor, self.rng)
        else:
            x, y = generate_connected_grid_neighborhoods(
                self.n_agents, self.distance_threshold,
                self.grid_rows, self.grid_cols, self.n_connectors
            )
            z = generate_elevation(x, self.slope, self.noise_factor, self.rng)

        positions = np.column_stack([x, y])
        return positions, z


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("Spatial Module Test (v11)")
    print("=" * 40)

    rng = np.random.default_rng(42)
    gen = SpatialGenerator(n_agents=50, mode=2, distance_threshold=0.09,
                           grid_rows=3, grid_cols=4, n_connectors=2, rng=rng)
    positions, elevations = gen.generate()

    print(f"Agents: {len(positions)}")
    print(f"Position range: x=[{positions[:, 0].min():.2f}, "
          f"{positions[:, 0].max():.2f}]")
    print(f"Elevation range: [{elevations.min():.2f}, "
          f"{elevations.max():.2f}]")
