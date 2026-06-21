"""Extended visualization utilities for grid search benchmarks.

This module provides enhanced grid visualization beyond the base library,
including terrain-aware rendering, multi-algorithm comparison displays,
and metric summaries suitable for academic output.

Rendering Symbols:
    S  — Start position
    G  — Goal position
    *  — Path cell
    .  — Explored cell (not on final path)
    █  — Obstacle / blocked
    ~  — Difficult terrain
    ≈  — Water terrain
       — Free / empty cell
"""

from __future__ import annotations

from grid_path_benchmark.comparison import ScenarioReport
from grid_path_benchmark.grid_data import GridScenario, Terrain
from search_library import Grid
from search_library.grid.grid import Position


def render_grid_with_terrain(
    grid: Grid,
    scenario: GridScenario,
    *,
    path: list[Position] | None = None,
    explored: frozenset[Position] | None = None,
) -> str:
    """Render a grid with terrain types, path, and explored nodes.

    Produces a text representation of the grid using terrain-specific
    symbols. Overlays path and explored nodes when provided.

    Priority of symbols (highest first):
        1. Start (S)
        2. Goal (G)
        3. Path (*)
        4. Explored (.)
        5. Obstacle (█)
        6. Terrain symbol (~, ≈, or space)

    Args:
        grid: The Grid instance to render.
        scenario: The scenario configuration (for terrain info).
        path: Optional solution path to display.
        explored: Optional set of explored positions.

    Returns:
        Multi-line string representation of the grid.
    """
    path_set: set[Position] = set(path) if path else set()
    explored_set: set[Position] = set(explored) if explored else set()
    start = scenario.start
    goal = scenario.goal

    lines: list[str] = []

    # Column header
    col_header = "    " + "".join(f"{c % 10}" for c in range(grid.cols))
    lines.append(col_header)

    for row in range(grid.rows):
        cells: list[str] = []
        for col in range(grid.cols):
            pos: Position = (row, col)
            if pos == start:
                cells.append("S")
            elif pos == goal:
                cells.append("G")
            elif pos in path_set and pos != start and pos != goal:
                cells.append("*")
            elif pos in explored_set:
                cells.append("·")
            elif grid.is_obstacle(row, col):
                cells.append("█")
            else:
                # Check terrain
                terrain = scenario.terrain_map.get(pos)
                if terrain == Terrain.DIFFICULT:
                    cells.append("~")
                elif terrain == Terrain.WATER:
                    cells.append("≈")
                else:
                    # Check cost for generated weighted grids
                    cost = grid.get_cost(pos)
                    if cost >= Terrain.WATER.cost:
                        cells.append("≈")
                    elif cost >= Terrain.DIFFICULT.cost:
                        cells.append("~")
                    else:
                        cells.append(" ")

        row_str = f"{row:3d} " + "".join(cells)
        lines.append(row_str)

    return "\n".join(lines)


def render_scenario_result(
    report: ScenarioReport,
    algorithm_name: str,
    *,
    show_explored: bool = False,
) -> str:
    """Render a specific algorithm's result on the scenario grid.

    Args:
        report: The scenario report containing grid and results.
        algorithm_name: Which algorithm's result to render.
        show_explored: Whether to show explored (visited) cells.

    Returns:
        Text rendering of the grid with the algorithm's path.
    """
    if report.grid is None:
        return "(no grid available)"

    result = next(
        (r for r in report.results if r.algorithm_name == algorithm_name),
        None,
    )
    if result is None:
        return f"(no result for {algorithm_name})"

    explored: frozenset[Position] | None = None
    if show_explored:
        # For visualization, we would need to re-run with track_explored
        # or store it. For now, we skip explored display.
        explored = None

    path = result.path if result.success else None
    return render_grid_with_terrain(
        report.grid,
        report.scenario,
        path=path,
        explored=explored,
    )


def render_comparison_grids(
    report: ScenarioReport,
    algorithms: list[str] | None = None,
) -> str:
    """Render multiple algorithm results side-by-side.

    For each algorithm, renders a labeled grid showing its path.
    Useful for visual comparison in terminal output.

    Args:
        report: The scenario report.
        algorithms: Which algorithms to show. Defaults to all.

    Returns:
        Multi-line string with labeled grids.
    """
    if algorithms is None:
        algorithms = report.algorithm_names

    lines: list[str] = []

    for algo_name in algorithms:
        result = next(
            (r for r in report.results if r.algorithm_name == algo_name),
            None,
        )
        if result is None:
            continue

        lines.append(f"  ┌─ {algo_name} {'─' * (40 - len(algo_name))}")
        if result.success:
            lines.append(f"  │ Cost: {result.total_cost:.2f} | Nodes: {result.nodes_explored}")
        else:
            lines.append("  │ No path found")
        lines.append("  │")

        grid_text = render_scenario_result(report, algo_name)
        for grid_line in grid_text.split("\n"):
            lines.append(f"  │ {grid_line}")

        lines.append(f"  └{'─' * 50}")
        lines.append("")

    return "\n".join(lines)


def format_metrics_summary(report: ScenarioReport) -> str:
    """Format a concise metrics summary for a scenario.

    Shows key metrics in a compact table format.

    Args:
        report: The scenario report.

    Returns:
        Formatted metrics summary string.
    """
    lines: list[str] = []
    lines.append(f"  Metrics Summary: {report.scenario.name}")
    lines.append(f"  {'─' * 50}")

    for result in report.results:
        if result.success:
            opt = "✓" if result.is_optimal else "✗"
            lines.append(
                f"  {result.algorithm_name:<15} "
                f"cost={result.total_cost:>8.2f}  "
                f"nodes={result.nodes_explored:>6}  "
                f"ms={result.execution_time_ms:>8.3f}  "
                f"opt={opt}"
            )
        else:
            lines.append(f"  {result.algorithm_name:<15} NO PATH FOUND")

    return "\n".join(lines)
