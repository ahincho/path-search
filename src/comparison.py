"""Algorithm comparison engine for grid pathfinding benchmarks.

This module provides structured comparison of search algorithms on grid
scenarios, collecting metrics and generating formatted output suitable
for academic analysis and reproducible experiments.

Metrics collected per algorithm per scenario:
- Path found (success/failure)
- Total cost (sum of traversal costs along the path)
- Path length (number of cells visited)
- Nodes explored (search efficiency / expansions)
- Execution time (wall-clock in milliseconds)
- Optimality (compared against Dijkstra baseline)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from grid_path_benchmark.grid_data import (
    GridScenario,
    generate_bug_trap_grid,
    generate_grid,
    generate_weighted_grid,
)
from grid_path_benchmark.heuristics import OctileHeuristic
from search_library import (
    Grid,
    ManhattanHeuristic,
    SearchResult,
    astar_search,
    bfs_search,
    bidirectional_search,
    dfs_search,
    dijkstra_search,
)
from search_library.grid.grid import Position
from search_library.grid.grid_search import GridSearchProblem
from search_library.heuristics.base import Heuristic
from search_library.heuristics.euclidean import EuclideanHeuristic

# ---------------------------------------------------------------------------
# Result Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlgorithmResult:
    """Stores the result of running a single algorithm on a grid scenario.

    Attributes:
        algorithm_name: Human-readable algorithm name.
        path: List of positions in the found path.
        total_cost: Total traversal cost of the path.
        path_length: Number of cells in the path.
        nodes_explored: Number of nodes expanded during search.
        execution_time_ms: Wall-clock time in milliseconds.
        is_optimal: Whether this path cost matches the optimal (Dijkstra) cost.
        success: Whether a path was found.
    """

    algorithm_name: str
    path: list[Position]
    total_cost: float
    path_length: int
    nodes_explored: int
    execution_time_ms: float
    is_optimal: bool
    success: bool


@dataclass
class ScenarioReport:
    """Aggregated comparison of all algorithms on a single grid scenario.

    Attributes:
        scenario: The scenario configuration.
        optimal_cost: The minimum achievable cost (from Dijkstra).
        results: Individual results per algorithm.
        grid: The generated grid used for this scenario.
    """

    scenario: GridScenario
    optimal_cost: float
    results: list[AlgorithmResult] = field(default_factory=list)
    grid: Grid | None = None

    @property
    def algorithm_names(self) -> list[str]:
        """Return list of algorithm names in comparison order."""
        return [r.algorithm_name for r in self.results]


@dataclass
class BenchmarkReport:
    """Complete benchmark report across all scenarios.

    Attributes:
        scenario_reports: Results for each scenario.
    """

    scenario_reports: list[ScenarioReport] = field(default_factory=list)

    @property
    def total_scenarios(self) -> int:
        """Return total number of scenarios benchmarked."""
        return len(self.scenario_reports)


# ---------------------------------------------------------------------------
# Heuristic Selection
# ---------------------------------------------------------------------------


def select_heuristic(scenario: GridScenario) -> Heuristic[Position]:
    """Select the optimal heuristic for a given scenario.

    Heuristic selection rules:
    - 4-directional uniform cost -> Manhattan (exact for obstacle-free)
    - 8-directional with sqrt(2) diagonal -> Octile (exact for obstacle-free)
    - Weighted grids -> Euclidean (admissible, conservative)

    Args:
        scenario: The grid scenario configuration.

    Returns:
        The most appropriate Heuristic instance.
    """
    if scenario.weighted:
        # Euclidean is admissible for weighted grids since cell costs >= 1
        return EuclideanHeuristic()
    if scenario.allow_diagonal:
        return OctileHeuristic()
    return ManhattanHeuristic()


# ---------------------------------------------------------------------------
# Comparison Engine
# ---------------------------------------------------------------------------


def run_scenario_comparison(scenario: GridScenario) -> ScenarioReport:
    """Execute all algorithms on a single grid scenario and compare results.

    Runs BFS, DFS, Dijkstra, A*, and Bidirectional on the same grid,
    collecting performance metrics for each. The optimal cost is
    determined by Dijkstra (guaranteed optimal for non-negative costs).

    Args:
        scenario: The GridScenario to benchmark.

    Returns:
        ScenarioReport with individual results and optimal cost reference.
    """
    # Generate the grid
    if scenario.name == "Bug Trap (20x20)":
        grid = generate_bug_trap_grid(
            scenario.rows, scenario.cols, allow_diagonal=scenario.allow_diagonal
        )
    elif scenario.weighted:
        grid = generate_weighted_grid(scenario)
    else:
        grid = generate_grid(scenario)

    heuristic = select_heuristic(scenario)
    start = scenario.start
    goal = scenario.goal

    results: list[AlgorithmResult] = []

    # Check if start == goal (trivial case)
    if start == goal:
        trivial = AlgorithmResult(
            algorithm_name="All (trivial)",
            path=[start],
            total_cost=0.0,
            path_length=1,
            nodes_explored=1,
            execution_time_ms=0.0,
            is_optimal=True,
            success=True,
        )
        return ScenarioReport(
            scenario=scenario,
            optimal_cost=0.0,
            results=[trivial],
            grid=grid,
        )

    # --- 1. Dijkstra (baseline: guaranteed optimal) ---
    dijkstra_result = _run_timed(
        "Dijkstra",
        grid,
        start,
        goal,
        heuristic=None,  # Dijkstra uses no heuristic
    )
    results.append(dijkstra_result)
    optimal_cost = dijkstra_result.total_cost if dijkstra_result.success else 0.0

    # --- 2. A* Search (informed: optimal with admissible heuristic) ---
    astar_result = _run_timed(
        "A*",
        grid,
        start,
        goal,
        heuristic=heuristic,
    )
    results.append(astar_result)

    # --- 3. BFS (uninformed: optimal by hop count, not by cost) ---
    bfs_result = _run_timed(
        "BFS",
        grid,
        start,
        goal,
        heuristic=None,
    )
    results.append(bfs_result)

    # --- 4. DFS (uninformed: no optimality guarantee) ---
    dfs_result = _run_timed(
        "DFS",
        grid,
        start,
        goal,
        heuristic=None,
    )
    results.append(dfs_result)

    # --- 5. Bidirectional BFS ---
    bidir_result = _run_timed_bidirectional(
        "Bidirectional",
        grid,
        start,
        goal,
    )
    results.append(bidir_result)

    # Mark optimality for each result
    final_results = [
        AlgorithmResult(
            algorithm_name=r.algorithm_name,
            path=r.path,
            total_cost=r.total_cost,
            path_length=r.path_length,
            nodes_explored=r.nodes_explored,
            execution_time_ms=r.execution_time_ms,
            is_optimal=_is_close(r.total_cost, optimal_cost) if r.success else False,
            success=r.success,
        )
        for r in results
    ]

    return ScenarioReport(
        scenario=scenario,
        optimal_cost=optimal_cost,
        results=final_results,
        grid=grid,
    )


def run_full_benchmark(scenarios: list[GridScenario]) -> BenchmarkReport:
    """Execute the full benchmark suite across all scenarios.

    Args:
        scenarios: List of GridScenario configurations.

    Returns:
        BenchmarkReport containing all scenario results.
    """
    report = BenchmarkReport()
    for scenario in scenarios:
        scenario_report = run_scenario_comparison(scenario)
        report.scenario_reports.append(scenario_report)
    return report


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_scenario_table(report: ScenarioReport) -> str:
    """Format a scenario report as a readable ASCII table.

    Args:
        report: The ScenarioReport to format.

    Returns:
        Multi-line string with formatted comparison table.
    """
    lines: list[str] = []

    lines.append(f"  Scenario: {report.scenario.name}")
    lines.append(f"  {report.scenario.description}")
    lines.append(f"  Grid: {report.scenario.rows}x{report.scenario.cols}")
    lines.append(f"  Start: {report.scenario.start} → Goal: {report.scenario.goal}")
    if report.optimal_cost > 0:
        lines.append(f"  Optimal cost (Dijkstra): {report.optimal_cost:.2f}")
    lines.append("")

    # Table header
    header = (
        f"  {'Algorithm':<15} {'Cost':<10} {'Path Len':<10} "
        f"{'Nodes':<10} {'Time (ms)':<12} {'Optimal':<8}"
    )
    lines.append(header)
    lines.append("  " + "-" * 70)

    for result in report.results:
        if result.success:
            optimal_mark = "✓" if result.is_optimal else "✗"
            row = (
                f"  {result.algorithm_name:<15} "
                f"{result.total_cost:<10.2f} "
                f"{result.path_length:<10} "
                f"{result.nodes_explored:<10} "
                f"{result.execution_time_ms:<12.4f} "
                f"{optimal_mark:<8}"
            )
        else:
            row = (
                f"  {result.algorithm_name:<15} "
                f"{'N/A':<10} {'N/A':<10} "
                f"{result.nodes_explored:<10} "
                f"{result.execution_time_ms:<12.4f} "
                f"{'—':<8}"
            )
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def format_full_report(benchmark: BenchmarkReport) -> str:
    """Format the complete benchmark report.

    Args:
        benchmark: The full BenchmarkReport.

    Returns:
        Multi-line formatted string with all scenario results.
    """
    lines: list[str] = []

    lines.append("=" * 80)
    lines.append("  GRID SEARCH BENCHMARK — Algorithm Comparison Report")
    lines.append("  search-library v1.0.0 | grid-path-benchmark v1.0.0")
    lines.append("=" * 80)
    lines.append("")

    for i, scenario_report in enumerate(benchmark.scenario_reports, 1):
        lines.append(f"{'─' * 80}")
        lines.append(f"  [{i}/{benchmark.total_scenarios}]")
        lines.append(format_scenario_table(scenario_report))

    return "\n".join(lines)


def format_academic_analysis(benchmark: BenchmarkReport) -> str:
    """Generate academic analysis explaining benchmark results.

    Args:
        benchmark: The full BenchmarkReport.

    Returns:
        Multi-line academic analysis string.
    """
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  ACADEMIC ANALYSIS")
    lines.append("=" * 80)
    lines.append("")

    # 1. A* vs Dijkstra efficiency comparison
    lines.append("  1. A* vs Dijkstra — Effect of Heuristic Guidance")
    lines.append("  " + "-" * 60)
    astar_savings: list[str] = []
    for sr in benchmark.scenario_reports:
        dijkstra = next((r for r in sr.results if r.algorithm_name == "Dijkstra"), None)
        astar = next((r for r in sr.results if r.algorithm_name == "A*"), None)
        if (
            dijkstra and astar and dijkstra.success and astar.success
            and dijkstra.nodes_explored > 0
        ):
                savings = (
                    (dijkstra.nodes_explored - astar.nodes_explored)
                    / dijkstra.nodes_explored
                    * 100
                )
                astar_savings.append(
                    f"    {sr.scenario.name}: "
                    f"A* explored {astar.nodes_explored} vs Dijkstra {dijkstra.nodes_explored} "
                    f"({savings:.0f}% fewer nodes)"
                )
    if astar_savings:
        lines.extend(astar_savings)
    lines.append("")
    lines.append("    Conclusion: A* consistently explores fewer or equal nodes than Dijkstra")
    lines.append("    while finding the same optimal cost. The heuristic prunes unpromising")
    lines.append("    branches, with greater savings on larger grids.")
    lines.append("")

    # 2. BFS limitations in weighted grids
    lines.append("  2. BFS in Weighted Grids — Cost vs Hop-Count Optimality")
    lines.append("  " + "-" * 60)
    for sr in benchmark.scenario_reports:
        if sr.scenario.weighted:
            bfs = next((r for r in sr.results if r.algorithm_name == "BFS"), None)
            if bfs and bfs.success and sr.optimal_cost > 0:
                cost_ratio = bfs.total_cost / sr.optimal_cost
                lines.append(
                    f"    {sr.scenario.name}: BFS cost = {bfs.total_cost:.2f} "
                    f"(optimal = {sr.optimal_cost:.2f}, ratio = {cost_ratio:.2f}x)"
                )
    lines.append("")
    lines.append("    BFS minimizes hop count (edge count), not total traversal cost.")
    lines.append("    In weighted grids this produces suboptimal paths since it ignores")
    lines.append("    terrain difficulty when choosing which path to take.")
    lines.append("")

    # 3. DFS non-optimality
    lines.append("  3. DFS — First Path Found vs Optimal Path")
    lines.append("  " + "-" * 60)
    for sr in benchmark.scenario_reports:
        dfs = next((r for r in sr.results if r.algorithm_name == "DFS"), None)
        if dfs and dfs.success and sr.optimal_cost > 0:
            ratio = dfs.total_cost / sr.optimal_cost
            lines.append(
                f"    {sr.scenario.name}: DFS cost = {dfs.total_cost:.2f} "
                f"(ratio = {ratio:.2f}x optimal)"
            )
    lines.append("")
    lines.append("    DFS explores depth-first and returns the first path found.")
    lines.append("    Results depend entirely on successor ordering, not path quality.")
    lines.append("")

    # 4. Heuristic comparison
    lines.append("  4. Heuristic Selection — Manhattan vs Octile vs Euclidean")
    lines.append("  " + "-" * 60)
    lines.append("    • Manhattan: exact for 4-dir uniform-cost grids")
    lines.append("    • Octile: exact for 8-dir grids with √2 diagonal cost")
    lines.append("    • Euclidean: admissible universally, but less tight than Octile")
    lines.append("    • Chebyshev: exact for 8-dir uniform-cost (all moves cost 1)")
    lines.append("")
    lines.append("    Tighter heuristics reduce node expansions. The octile heuristic")
    lines.append("    outperforms Euclidean on 8-directional grids because it accounts")
    lines.append("    for the actual movement model (diagonal costs √2, not 1).")
    lines.append("")

    # 5. Obstacle density impact
    lines.append("  5. Obstacle Density Impact on Search Effort")
    lines.append("  " + "-" * 60)
    density_data: list[tuple[str, float, int]] = []
    for sr in benchmark.scenario_reports:
        astar = next((r for r in sr.results if r.algorithm_name == "A*"), None)
        if astar and astar.success:
            density_data.append(
                (sr.scenario.name, sr.scenario.obstacle_density, astar.nodes_explored)
            )
    for name, density, nodes in density_data:
        lines.append(f"    {name}: density={density:.0%}, A* expanded {nodes} nodes")
    lines.append("")
    lines.append("    Higher obstacle density increases path complexity and search effort.")
    lines.append("    Algorithms must explore more alternatives when direct paths are blocked.")
    lines.append("")

    # Conclusion
    lines.append("  CONCLUSIONS")
    lines.append("  " + "=" * 60)
    lines.append("  For 2D grid pathfinding:")
    lines.append("  • A* with tight heuristic is the best general choice (optimal + efficient)")
    lines.append("  • Dijkstra guarantees optimality but explores more nodes")
    lines.append("  • BFS is only cost-optimal on uniform-cost grids")
    lines.append("  • DFS is unsuitable for shortest-path problems")
    lines.append("  • Bidirectional BFS halves search space on uniform-cost grids")
    lines.append("  • Heuristic choice directly impacts performance:")
    lines.append("    - Use Manhattan for 4-dir, Octile for 8-dir with √2 diagonal")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _run_timed(
    name: str,
    grid: Grid,
    start: Position,
    goal: Position,
    *,
    heuristic: Heuristic[Position] | None,
) -> AlgorithmResult:
    """Run a search algorithm with timing and return structured result."""
    problem = GridSearchProblem(grid, start, goal, heuristic)

    t0 = time.perf_counter_ns()

    if name == "Dijkstra":
        result: SearchResult[Position] = dijkstra_search(problem, track_explored=True)
    elif name == "A*":
        result = astar_search(problem, track_explored=True)
    elif name == "BFS":
        result = bfs_search(problem, track_explored=True)
    elif name == "DFS":
        result = dfs_search(problem, track_explored=True)
    else:
        msg = f"Unknown algorithm: {name}"
        raise ValueError(msg)

    elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000

    return AlgorithmResult(
        algorithm_name=name,
        path=list(result.path),
        total_cost=result.total_cost,
        path_length=result.path_length,
        nodes_explored=result.nodes_explored,
        execution_time_ms=elapsed_ms,
        is_optimal=False,  # Will be set later
        success=result.success,
    )


def _run_timed_bidirectional(
    name: str,
    grid: Grid,
    start: Position,
    goal: Position,
) -> AlgorithmResult:
    """Run bidirectional search with timing."""
    forward_problem = GridSearchProblem(grid, start, goal)
    reverse_problem = GridSearchProblem(grid, goal, start)

    t0 = time.perf_counter_ns()
    result: SearchResult[Position] = bidirectional_search(
        forward_problem,
        reverse_problem=reverse_problem,
        track_explored=True,
    )
    elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000

    return AlgorithmResult(
        algorithm_name=name,
        path=list(result.path),
        total_cost=result.total_cost,
        path_length=result.path_length,
        nodes_explored=result.nodes_explored,
        execution_time_ms=elapsed_ms,
        is_optimal=False,
        success=result.success,
    )


def _is_close(a: float, b: float, rel_tol: float = 1e-9) -> bool:
    """Check if two floats are approximately equal."""
    if b == 0.0:
        return a == 0.0
    return abs(a - b) / max(abs(a), abs(b)) <= rel_tol
