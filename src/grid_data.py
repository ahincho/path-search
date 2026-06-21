"""Grid scenario definitions and generators for benchmarking.

This module defines:
- Terrain types with associated traversal costs
- Grid scenario configurations (4-dir, 8-dir, obstacles, weighted, etc.)
- Generator functions for reproducible grid creation
- Predefined scenarios for academic comparison

All grids are generated deterministically using explicit seeds,
ensuring experiment reproducibility across runs.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum

from search_library import Grid

# Position type alias
Position = tuple[int, int]


# ---------------------------------------------------------------------------
# Terrain System
# ---------------------------------------------------------------------------


class Terrain(Enum):
    """Terrain types with associated traversal costs.

    Each terrain has a base cost that multiplies the movement cost.
    BLOCKED terrain is impassable (represented as obstacles in the grid).

    Costs are designed for academic demonstration:
    - FREE: normal traversal (cost = 1.0)
    - DIFFICULT: rough terrain, 2x cost (mud, sand)
    - WATER: very expensive traversal, 5x cost (swimming/wading)
    - BLOCKED: impassable (∞ cost, treated as obstacle)
    """

    FREE = 1.0
    DIFFICULT = 2.0
    WATER = 5.0
    BLOCKED = math.inf

    @property
    def cost(self) -> float:
        """Return the traversal cost for this terrain type."""
        return self.value

    @property
    def is_passable(self) -> bool:
        """Return whether this terrain can be traversed."""
        return self.value != math.inf

    @property
    def symbol(self) -> str:
        """Return the display symbol for this terrain."""
        symbols = {
            Terrain.FREE: " ",
            Terrain.DIFFICULT: "~",
            Terrain.WATER: "≈",
            Terrain.BLOCKED: "█",
        }
        return symbols[self]


# ---------------------------------------------------------------------------
# Grid Scenario Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GridScenario:
    """Configuration for a grid benchmark scenario.

    Encapsulates all parameters needed to create a reproducible grid
    experiment: dimensions, movement model, obstacle density, terrain
    distribution, and start/goal positions.

    Attributes:
        name: Human-readable scenario name.
        description: Brief description of what this scenario demonstrates.
        rows: Grid height.
        cols: Grid width.
        allow_diagonal: Whether 8-directional movement is enabled.
        obstacle_density: Fraction of cells that are obstacles (0.0 to 1.0).
        start: Starting position (row, col).
        goal: Goal position (row, col).
        seed: Random seed for reproducible generation.
        terrain_map: Optional explicit terrain assignment for cells.
        weighted: Whether the grid uses variable traversal costs.
    """

    name: str
    description: str
    rows: int
    cols: int
    allow_diagonal: bool
    obstacle_density: float
    start: Position
    goal: Position
    seed: int = 42
    terrain_map: dict[Position, Terrain] = field(default_factory=dict)
    weighted: bool = False


# ---------------------------------------------------------------------------
# Grid Generator Functions
# ---------------------------------------------------------------------------


def generate_grid(scenario: GridScenario) -> Grid:
    """Generate a Grid instance from a scenario configuration.

    Creates a grid with obstacles placed according to the scenario's
    density parameter, using the seed for reproducible placement.
    Also applies terrain costs if the scenario is weighted.

    The generator guarantees that start and goal positions are never
    placed on obstacles.

    Args:
        scenario: The GridScenario configuration.

    Returns:
        A configured Grid instance ready for pathfinding.
    """
    grid = Grid(
        scenario.rows,
        scenario.cols,
        allow_diagonal=scenario.allow_diagonal,
    )

    # Apply explicit terrain map if provided
    if scenario.terrain_map:
        for pos, terrain in scenario.terrain_map.items():
            row, col = pos
            if not grid.in_bounds(row, col):
                continue
            if terrain == Terrain.BLOCKED:
                grid.set_obstacle(row, col)
            elif terrain != Terrain.FREE:
                grid.set_cost(row, col, terrain.cost)

    # Generate random obstacles based on density
    if scenario.obstacle_density > 0 and not scenario.terrain_map:
        rng = random.Random(scenario.seed)
        total_cells = scenario.rows * scenario.cols
        num_obstacles = int(total_cells * scenario.obstacle_density)

        # Build list of candidate positions (exclude start and goal)
        candidates = [
            (r, c)
            for r in range(scenario.rows)
            for c in range(scenario.cols)
            if (r, c) != scenario.start and (r, c) != scenario.goal
        ]

        rng.shuffle(candidates)

        for row, col in candidates[:num_obstacles]:
            grid.set_obstacle(row, col)

    return grid


def generate_weighted_grid(scenario: GridScenario) -> Grid:
    """Generate a grid with variable terrain costs.

    Places terrain types randomly according to a distribution:
    - 60% FREE (cost 1.0)
    - 20% DIFFICULT (cost 2.0)
    - 10% WATER (cost 5.0)
    - 10% BLOCKED (obstacle)

    The obstacle_density parameter in the scenario is ignored here;
    terrain distribution controls the blocking percentage.

    Args:
        scenario: The GridScenario configuration.

    Returns:
        A Grid with variable cell costs and obstacles.
    """
    grid = Grid(
        scenario.rows,
        scenario.cols,
        allow_diagonal=scenario.allow_diagonal,
    )

    rng = random.Random(scenario.seed)

    for r in range(scenario.rows):
        for c in range(scenario.cols):
            if (r, c) == scenario.start or (r, c) == scenario.goal:
                continue

            roll = rng.random()
            if roll < 0.60:
                pass  # FREE: default cost 1.0
            elif roll < 0.80:
                grid.set_cost(r, c, Terrain.DIFFICULT.cost)
            elif roll < 0.90:
                grid.set_cost(r, c, Terrain.WATER.cost)
            else:
                grid.set_obstacle(r, c)

    return grid


def generate_bug_trap_grid(rows: int, cols: int, *, allow_diagonal: bool = False) -> Grid:
    """Generate a grid with a 'bug trap' — a concave obstacle pocket.

    A bug trap is a U-shaped or concave obstacle formation that traps
    greedy algorithms (like GBFS) because the heuristic guides the search
    into the pocket. Only algorithms that explore exhaustively (BFS, Dijkstra)
    or use proper f-cost (A*) escape efficiently.

    Layout example (20x20 grid):
        Start at (1, 1), Goal at (rows-2, cols-2)
        U-shaped wall in the center creates the trap

    Args:
        rows: Grid height (minimum 15).
        cols: Grid width (minimum 15).
        allow_diagonal: Whether diagonal movement is allowed.

    Returns:
        A Grid with a bug trap obstacle formation.
    """
    rows = max(rows, 15)
    cols = max(cols, 15)
    grid = Grid(rows, cols, allow_diagonal=allow_diagonal)

    # Create U-shaped trap in the center
    trap_top = rows // 3
    trap_bottom = 2 * rows // 3
    trap_left = cols // 3
    trap_right = 2 * cols // 3

    # Left wall of U
    for r in range(trap_top, trap_bottom + 1):
        grid.set_obstacle(r, trap_left)

    # Bottom wall of U
    for c in range(trap_left, trap_right + 1):
        grid.set_obstacle(trap_bottom, c)

    # Right wall of U
    for r in range(trap_top, trap_bottom + 1):
        grid.set_obstacle(r, trap_right)

    return grid


# ---------------------------------------------------------------------------
# Predefined Benchmark Scenarios
# ---------------------------------------------------------------------------


def build_scenario_4dir_basic() -> GridScenario:
    """Scenario 1: Basic 4-directional grid with 10% obstacles.

    Demonstrates baseline behavior with cardinal movement only.
    Manhattan heuristic is optimal here.
    """
    return GridScenario(
        name="4-Dir Basic (20x20, 10% obstacles)",
        description="Cardinal movement only, uniform cost, moderate obstacles",
        rows=20,
        cols=20,
        allow_diagonal=False,
        obstacle_density=0.10,
        start=(0, 0),
        goal=(19, 19),
        seed=42,
    )


def build_scenario_8dir_basic() -> GridScenario:
    """Scenario 2: Basic 8-directional grid with 10% obstacles.

    Demonstrates diagonal movement with sqrt(2) cost.
    Octile heuristic is tightest here.
    """
    return GridScenario(
        name="8-Dir Basic (20x20, 10% obstacles)",
        description="Diagonal movement allowed, sqrt(2) diagonal cost, moderate obstacles",
        rows=20,
        cols=20,
        allow_diagonal=True,
        obstacle_density=0.10,
        start=(0, 0),
        goal=(19, 19),
        seed=42,
    )


def build_scenario_dense_obstacles() -> GridScenario:
    """Scenario 3: Dense obstacle field (30% density).

    Tests algorithm behavior when paths are heavily constrained.
    Many nodes are explored before finding a path.
    """
    return GridScenario(
        name="Dense Obstacles (20x20, 30%)",
        description="Heavy obstacle density forcing long detours",
        rows=20,
        cols=20,
        allow_diagonal=False,
        obstacle_density=0.30,
        start=(0, 0),
        goal=(19, 19),
        seed=123,
    )


def build_scenario_weighted_terrain() -> GridScenario:
    """Scenario 4: Weighted terrain grid with variable costs.

    Demonstrates how cost-aware algorithms (Dijkstra, A*) outperform
    hop-count algorithms (BFS) when cell costs vary.
    """
    return GridScenario(
        name="Weighted Terrain (20x20)",
        description="Variable terrain costs (free/difficult/water/blocked)",
        rows=20,
        cols=20,
        allow_diagonal=True,
        obstacle_density=0.0,  # Handled by weighted generator
        start=(0, 0),
        goal=(19, 19),
        seed=77,
        weighted=True,
    )


def build_scenario_bug_trap() -> GridScenario:
    """Scenario 5: Bug trap — concave obstacle formation.

    Tests how algorithms handle deceptive heuristic guidance.
    A* must explore nodes outside the trap to escape.
    """
    return GridScenario(
        name="Bug Trap (20x20)",
        description="U-shaped obstacle trapping greedy heuristic-guided search",
        rows=20,
        cols=20,
        allow_diagonal=False,
        obstacle_density=0.0,
        start=(1, 1),
        goal=(18, 18),
        seed=42,
    )


def build_scenario_large_sparse() -> GridScenario:
    """Scenario 6: Large sparse grid (50x50, 5% obstacles).

    Tests scalability and shows efficiency differences at scale.
    """
    return GridScenario(
        name="Large Sparse (50x50, 5%)",
        description="Large grid with few obstacles — tests scalability",
        rows=50,
        cols=50,
        allow_diagonal=True,
        obstacle_density=0.05,
        start=(0, 0),
        goal=(49, 49),
        seed=99,
    )


def build_scenario_no_path() -> GridScenario:
    """Scenario 7: No path possible — goal surrounded by obstacles.

    Edge case: algorithms must detect and report failure.
    """
    # Goal at (9,9) in a 10x10 grid, surrounded by obstacles
    terrain: dict[Position, Terrain] = {}
    # Surround goal (9,9) with blocked cells
    for r in range(8, 10):
        for c in range(8, 10):
            if (r, c) != (9, 9):
                terrain[(r, c)] = Terrain.BLOCKED
    # Also block the edge cells adjacent to goal
    terrain[(9, 8)] = Terrain.BLOCKED
    terrain[(8, 9)] = Terrain.BLOCKED

    return GridScenario(
        name="No Path (10x10, blocked goal)",
        description="Goal completely surrounded — tests failure detection",
        rows=10,
        cols=10,
        allow_diagonal=False,
        obstacle_density=0.0,
        start=(0, 0),
        goal=(9, 9),
        seed=42,
        terrain_map=terrain,
    )


def build_scenario_start_is_goal() -> GridScenario:
    """Scenario 8: Start equals goal — trivial case.

    Edge case: should return immediately with zero-cost path.
    """
    return GridScenario(
        name="Start == Goal (10x10)",
        description="Trivial case — algorithms should return instantly",
        rows=10,
        cols=10,
        allow_diagonal=False,
        obstacle_density=0.0,
        start=(5, 5),
        goal=(5, 5),
        seed=42,
    )


def get_all_scenarios() -> list[GridScenario]:
    """Return all predefined benchmark scenarios.

    Returns:
        Ordered list of GridScenario instances for full benchmark suite.
    """
    return [
        build_scenario_4dir_basic(),
        build_scenario_8dir_basic(),
        build_scenario_dense_obstacles(),
        build_scenario_weighted_terrain(),
        build_scenario_bug_trap(),
        build_scenario_large_sparse(),
        build_scenario_no_path(),
        build_scenario_start_is_goal(),
    ]
