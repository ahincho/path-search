"""Extended visualization utilities for grid search benchmarks.

This module provides enhanced grid visualization beyond the base library,
including terrain-aware rendering with ANSI colors, multi-algorithm
comparison displays, and metric summaries suitable for academic output.

Rendering Symbols (colored):
    S  — Start position     (bright green)
    G  — Goal position      (bright magenta)
    *  — Path cell          (bright yellow)
    .  — Explored cell      (dim cyan)
    █  — Obstacle / blocked (dim)
    ~  — Difficult terrain  (yellow)
    ≈  — Water terrain      (bright blue)
       — Free / empty cell  (default)
"""

from __future__ import annotations

from colorama import Fore, Style
from grid_path_benchmark.comparison import ScenarioReport
from grid_path_benchmark.grid_data import GridScenario, Terrain
from search_library import Grid
from search_library.grid.grid import Position

# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------

_R = Style.RESET_ALL
_D = Style.DIM
_H = Style.BRIGHT

# Cell-type color styles — applied before the character, _R after
_CELL_STYLE: dict[str, str] = {
    "S": Fore.GREEN + Style.BRIGHT,
    "G": Fore.MAGENTA + Style.BRIGHT,
    "*": Fore.YELLOW + Style.BRIGHT,
    "·": Fore.CYAN + Style.DIM,
    "█": Style.DIM,
    "~": Fore.YELLOW,
    "≈": Fore.BLUE + Style.BRIGHT,
}


def _c(char: str) -> str:
    """Apply cell color and reset. Plain chars pass through unchanged."""
    style = _CELL_STYLE.get(char)
    if style:
        return style + char + _R
    return char


def _render_legend() -> str:
    """Return a colored single-line legend for all grid symbols."""
    items = [
        ("S", "start"),
        ("G", "goal"),
        ("*", "path"),
        ("·", "explored"),
        ("█", "obstacle"),
        ("~", "difficult"),
        ("≈", "water"),
    ]
    parts = [f"{_c(sym)} {_D}{label}{_R}" for sym, label in items]
    return f"  {_D}Legend:{_R}  " + "   ".join(parts)


# ---------------------------------------------------------------------------
# Grid rendering
# ---------------------------------------------------------------------------


def render_grid_with_terrain(
    grid: Grid,
    scenario: GridScenario,
    *,
    path: list[Position] | None = None,
    explored: frozenset[Position] | None = None,
) -> str:
    """Render a grid with terrain types, path, and explored nodes (colored).

    Symbol priority (highest first):
        1. Start (S)  2. Goal (G)  3. Path (*)  4. Explored (·)
        5. Obstacle (█)  6. Terrain (~, ≈)  7. Free ( )

    Args:
        grid: The Grid instance to render.
        scenario: Scenario config (provides start, goal, terrain_map).
        path: Optional solution path to display.
        explored: Optional set of explored positions.

    Returns:
        Multi-line colored string representation of the grid.
    """
    path_set: set[Position] = set(path) if path else set()
    explored_set: set[Position] = set(explored) if explored else set()
    start = scenario.start
    goal = scenario.goal

    lines: list[str] = []

    # Column header — dim digits
    col_digits = "".join(f"{c % 10}" for c in range(grid.cols))
    lines.append(f"  {_D}    {col_digits}{_R}")

    for row in range(grid.rows):
        cells: list[str] = []
        for col in range(grid.cols):
            pos: Position = (row, col)
            if pos == start:
                cells.append(_c("S"))
            elif pos == goal:
                cells.append(_c("G"))
            elif pos in path_set and pos != start and pos != goal:
                cells.append(_c("*"))
            elif pos in explored_set:
                cells.append(_c("·"))
            elif grid.is_obstacle(row, col):
                cells.append(_c("█"))
            else:
                terrain = scenario.terrain_map.get(pos)
                if terrain == Terrain.DIFFICULT:
                    cells.append(_c("~"))
                elif terrain == Terrain.WATER:
                    cells.append(_c("≈"))
                else:
                    cost = grid.get_cost(pos)
                    if cost >= Terrain.WATER.cost:
                        cells.append(_c("≈"))
                    elif cost >= Terrain.DIFFICULT.cost:
                        cells.append(_c("~"))
                    else:
                        cells.append(" ")

        # Row number — dim, right-aligned
        row_str = f"  {_D}{row:3d}{_R} " + "".join(cells)
        lines.append(row_str)

    lines.append("")
    lines.append(_render_legend())

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
        Colored text rendering of the grid with the algorithm's path.
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
        # explored_states requires re-running with track_explored; omitted here.
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
    """Render multiple algorithm results with labeled headers.

    Args:
        report: The scenario report.
        algorithms: Which algorithms to show. Defaults to all.

    Returns:
        Multi-line string with labeled, colored grids.
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

        bar = "─" * max(1, 40 - len(algo_name))
        lines.append(f"  {Fore.CYAN}┌─ {_H}{algo_name}{_R}{Fore.CYAN} {bar}{_R}")
        if result.success:
            opt = f"{Fore.GREEN + _H}optimal{_R}" if result.is_optimal else f"{Fore.RED}suboptimal{_R}"
            lines.append(
                f"  {Fore.CYAN}│{_R} Cost: {_H}{result.total_cost:.2f}{_R}"
                f"  Nodes: {_H}{result.nodes_explored}{_R}  {opt}"
            )
        else:
            lines.append(f"  {Fore.CYAN}│{_R} {Fore.RED}No path found{_R}")
        lines.append(f"  {Fore.CYAN}│{_R}")

        grid_text = render_scenario_result(report, algo_name)
        for grid_line in grid_text.split("\n"):
            lines.append(f"  {Fore.CYAN}│{_R} {grid_line}")

        lines.append(f"  {Fore.CYAN}└{'─' * 50}{_R}")
        lines.append("")

    return "\n".join(lines)


def format_metrics_summary(report: ScenarioReport) -> str:
    """Format a concise colored metrics summary for a scenario.

    Args:
        report: The scenario report.

    Returns:
        Formatted metrics summary string.
    """
    lines: list[str] = []
    lines.append(f"  {Fore.CYAN + _H}Metrics Summary: {report.scenario.name}{_R}")
    lines.append(f"  {_D}{'─' * 50}{_R}")

    for result in report.results:
        if result.success:
            opt = (Fore.GREEN + _H + "✓" + _R) if result.is_optimal else (Fore.RED + "✗" + _R)
            lines.append(
                f"  {result.algorithm_name:<15} "
                f"cost={_H}{result.total_cost:>8.2f}{_R}  "
                f"nodes={_H}{result.nodes_explored:>6}{_R}  "
                f"ms={result.execution_time_ms:>8.3f}  "
                f"opt={opt}"
            )
        else:
            lines.append(f"  {_D}{result.algorithm_name:<15} NO PATH FOUND{_R}")

    return "\n".join(lines)
