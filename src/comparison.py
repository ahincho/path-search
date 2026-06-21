"""Algorithm comparison engine for grid pathfinding benchmarks.

This module provides structured comparison of search algorithms on grid
scenarios, collecting metrics and generating colored formatted output
suitable for academic analysis and reproducible experiments.

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

from colorama import Fore, Style
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
# Color constants (module-level for reuse across all format functions)
# ---------------------------------------------------------------------------

_R = Style.RESET_ALL       # reset all
_D = Style.DIM             # dim / subtle
_H = Style.BRIGHT          # bold / bright
_CYAN = Fore.CYAN + Style.BRIGHT
_GREEN = Fore.GREEN + Style.BRIGHT
_RED = Fore.RED
_YELLOW = Fore.YELLOW + Style.BRIGHT


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
    """Complete benchmark report across all scenarios."""

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

    Rules:
    - 4-dir uniform cost  → Manhattan (exact for obstacle-free)
    - 8-dir sqrt(2) diag  → Octile (exact for obstacle-free)
    - Weighted grids      → Euclidean (admissible, conservative)

    Args:
        scenario: The grid scenario configuration.

    Returns:
        The most appropriate Heuristic instance.
    """
    if scenario.weighted:
        return EuclideanHeuristic()
    if scenario.allow_diagonal:
        return OctileHeuristic()
    return ManhattanHeuristic()


# ---------------------------------------------------------------------------
# Comparison Engine
# ---------------------------------------------------------------------------


def run_scenario_comparison(scenario: GridScenario) -> ScenarioReport:
    """Execute all algorithms on a single grid scenario and compare results."""
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
        return ScenarioReport(scenario=scenario, optimal_cost=0.0, results=[trivial], grid=grid)

    dijkstra_result = _run_timed("Dijkstra", grid, start, goal, heuristic=None)
    results.append(dijkstra_result)
    optimal_cost = dijkstra_result.total_cost if dijkstra_result.success else 0.0

    results.append(_run_timed("A*", grid, start, goal, heuristic=heuristic))
    results.append(_run_timed("BFS", grid, start, goal, heuristic=None))
    results.append(_run_timed("DFS", grid, start, goal, heuristic=None))
    results.append(_run_timed_bidirectional("Bidirectional", grid, start, goal))

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
        scenario=scenario, optimal_cost=optimal_cost, results=final_results, grid=grid
    )


def run_full_benchmark(scenarios: list[GridScenario]) -> BenchmarkReport:
    """Execute the full benchmark suite across all scenarios."""
    report = BenchmarkReport()
    for scenario in scenarios:
        report.scenario_reports.append(run_scenario_comparison(scenario))
    return report


# ---------------------------------------------------------------------------
# Formatting (colored)
# ---------------------------------------------------------------------------


def format_scenario_table(report: ScenarioReport) -> str:
    """Format a scenario report as a colored ASCII table.

    Args:
        report: The ScenarioReport to format.

    Returns:
        Multi-line colored string with the comparison table.
    """
    lines: list[str] = []

    # Scenario header
    lines.append(f"  {_CYAN}Scenario: {_H}{report.scenario.name}{_R}")
    lines.append(f"  {_D}{report.scenario.description}{_R}")
    lines.append(
        f"  Grid: {_H}{report.scenario.rows}x{report.scenario.cols}{_R}"
        f"  |  Start: {_H}{report.scenario.start}{_R}"
        f"  →  Goal: {_H}{report.scenario.goal}{_R}"
    )
    if report.optimal_cost > 0:
        lines.append(
            f"  {_D}Optimal cost (Dijkstra):{_R} {_GREEN}{report.optimal_cost:.2f}{_R}"
        )
    lines.append("")

    # Table header — format plain strings first, then wrap in bold
    h_algo  = f"{'Algorithm':<15}"
    h_cost  = f"{'Cost':<10}"
    h_plen  = f"{'Path Len':<10}"
    h_nodes = f"{'Nodes':<10}"
    h_time  = f"{'Time (ms)':<12}"
    h_opt   = "Optimal"
    lines.append(
        f"  {_H}{h_algo} {h_cost} {h_plen} {h_nodes} {h_time} {h_opt}{_R}"
    )
    lines.append(f"  {_D}{'-' * 70}{_R}")

    for result in report.results:
        # Format all numeric fields as plain strings (preserves alignment)
        algo_str  = f"{result.algorithm_name:<15}"
        if result.success:
            cost_str  = f"{result.total_cost:<10.2f}"
            plen_str  = f"{result.path_length:<10}"
            nodes_str = f"{result.nodes_explored:<10}"
            time_str  = f"{result.execution_time_ms:<12.4f}"
            opt_mark  = (
                f"{_GREEN}✓{_R}" if result.is_optimal else f"{_RED}✗{_R}"
            )
            lines.append(
                f"  {algo_str} {cost_str} {plen_str} {nodes_str} {time_str} {opt_mark}"
            )
        else:
            nodes_str = f"{result.nodes_explored:<10}"
            time_str  = f"{result.execution_time_ms:<12.4f}"
            lines.append(
                f"  {_D}{algo_str} {'N/A':<10} {'N/A':<10}"
                f" {nodes_str} {time_str} —{_R}"
            )

    lines.append("")
    return "\n".join(lines)


def format_full_report(benchmark: BenchmarkReport) -> str:
    """Format the complete benchmark report with colored headers.

    Args:
        benchmark: The full BenchmarkReport.

    Returns:
        Multi-line colored string with all scenario results.
    """
    lines: list[str] = []

    lines.append(f"{_CYAN}{'=' * 80}{_R}")
    lines.append(f"  {_CYAN}GRID SEARCH BENCHMARK{_R} {_D}— Algorithm Comparison Report{_R}")
    lines.append(f"  {_D}search-library v1.0.0 | grid-path-benchmark v1.0.0{_R}")
    lines.append(f"{_CYAN}{'=' * 80}{_R}")
    lines.append("")

    for i, scenario_report in enumerate(benchmark.scenario_reports, 1):
        lines.append(f"{_D}{'─' * 80}{_R}")
        lines.append(f"  {_D}[{i}/{benchmark.total_scenarios}]{_R}")
        lines.append(format_scenario_table(scenario_report))

    return "\n".join(lines)


def format_academic_analysis(benchmark: BenchmarkReport) -> str:
    """Generate colored academic analysis explaining benchmark results.

    Args:
        benchmark: The full BenchmarkReport.

    Returns:
        Multi-line colored academic analysis string.
    """
    lines: list[str] = []
    lines.append(f"{_CYAN}{'=' * 80}{_R}")
    lines.append(f"  {_CYAN}ACADEMIC ANALYSIS{_R}")
    lines.append(f"{_CYAN}{'=' * 80}{_R}")
    lines.append("")

    # 1. A* vs Dijkstra
    lines.append(f"  {_H}1. A* vs Dijkstra — Effect of Heuristic Guidance{_R}")
    lines.append(f"  {_D}{'-' * 60}{_R}")
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
                / dijkstra.nodes_explored * 100
            )
            color = _GREEN if savings >= 50 else _YELLOW
            astar_savings.append(
                f"    {sr.scenario.name}: "
                f"A* {_H}{astar.nodes_explored}{_R} vs Dijkstra {_H}{dijkstra.nodes_explored}{_R}"
                f" nodes  ({color}{savings:.0f}% fewer{_R})"
            )
    lines.extend(astar_savings)
    lines.append("")
    lines.append(
        f"    {_D}Conclusion: A* consistently explores fewer or equal nodes than Dijkstra{_R}"
    )
    lines.append(
        f"    {_D}while finding the same optimal cost. Greater savings on larger grids.{_R}"
    )
    lines.append("")

    # 2. BFS limitations in weighted grids
    lines.append(f"  {_H}2. BFS in Weighted Grids — Cost vs Hop-Count Optimality{_R}")
    lines.append(f"  {_D}{'-' * 60}{_R}")
    for sr in benchmark.scenario_reports:
        if sr.scenario.weighted:
            bfs = next((r for r in sr.results if r.algorithm_name == "BFS"), None)
            if bfs and bfs.success and sr.optimal_cost > 0:
                ratio = bfs.total_cost / sr.optimal_cost
                lines.append(
                    f"    {sr.scenario.name}: BFS cost = {_RED}{bfs.total_cost:.2f}{_R}"
                    f"  (optimal = {_GREEN}{sr.optimal_cost:.2f}{_R}, ratio = {_H}{ratio:.2f}x{_R})"
                )
    lines.append(f"    {_D}BFS minimizes hop count, not cost — produces suboptimal paths on weighted grids.{_R}")
    lines.append("")

    # 3. DFS non-optimality
    lines.append(f"  {_H}3. DFS — First Path Found vs Optimal Path{_R}")
    lines.append(f"  {_D}{'-' * 60}{_R}")
    for sr in benchmark.scenario_reports:
        dfs = next((r for r in sr.results if r.algorithm_name == "DFS"), None)
        if dfs and dfs.success and sr.optimal_cost > 0:
            ratio = dfs.total_cost / sr.optimal_cost
            color = _RED if ratio > 2 else _YELLOW
            lines.append(
                f"    {sr.scenario.name}: DFS cost = {_H}{dfs.total_cost:.2f}{_R}"
                f"  ({color}{ratio:.2f}x optimal{_R})"
            )
    lines.append(f"    {_D}DFS returns the first path found — result depends on successor ordering.{_R}")
    lines.append("")

    # 4. Heuristic selection
    lines.append(f"  {_H}4. Heuristic Selection Guide{_R}")
    lines.append(f"  {_D}{'-' * 60}{_R}")
    lines.append(f"    {_YELLOW}•{_R} {_H}Manhattan{_R}  — exact for 4-dir uniform-cost grids")
    lines.append(f"    {_YELLOW}•{_R} {_H}Octile{_R}     — exact for 8-dir grids with √2 diagonal cost")
    lines.append(f"    {_YELLOW}•{_R} {_H}Euclidean{_R}  — admissible universally, less tight than Octile")
    lines.append(f"    {_YELLOW}•{_R} {_H}Chebyshev{_R}  — exact for 8-dir uniform-cost (all moves cost 1)")
    lines.append("")

    # 5. Obstacle density impact
    lines.append(f"  {_H}5. Obstacle Density Impact on Search Effort{_R}")
    lines.append(f"  {_D}{'-' * 60}{_R}")
    for sr in benchmark.scenario_reports:
        astar = next((r for r in sr.results if r.algorithm_name == "A*"), None)
        if astar and astar.success:
            lines.append(
                f"    {sr.scenario.name}: "
                f"density={_H}{sr.scenario.obstacle_density:.0%}{_R}, "
                f"A* expanded {_H}{astar.nodes_explored}{_R} nodes"
            )
    lines.append(f"    {_D}Higher density → longer detours → more nodes expanded.{_R}")
    lines.append("")

    # Conclusions
    lines.append(f"  {_CYAN}{'=' * 60}{_R}")
    lines.append(f"  {_CYAN}CONCLUSIONS — For 2D grid pathfinding:{_R}")
    lines.append(f"  {_CYAN}{'=' * 60}{_R}")
    lines.append(f"  {_YELLOW}•{_R} {_GREEN}A* + tight heuristic{_R} — optimal and efficient (best general choice)")
    lines.append(f"  {_YELLOW}•{_R} Dijkstra            — guarantees optimality, explores more nodes")
    lines.append(f"  {_YELLOW}•{_R} BFS                 — cost-optimal {_H}only{_R} on uniform-cost grids")
    lines.append(f"  {_YELLOW}•{_R} DFS                 — unsuitable for shortest-path problems")
    lines.append(f"  {_YELLOW}•{_R} Bidirectional BFS   — halves search space on uniform-cost grids")
    lines.append(f"    {_D}Use Manhattan for 4-dir, Octile for 8-dir with √2 diagonal cost.{_R}")
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
        is_optimal=False,
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
