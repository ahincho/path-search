"""Main entry point for the grid pathfinding benchmark demonstration.

Orchestrates the complete benchmark:
1. Generates all grid scenarios
2. Runs all search algorithms on each scenario
3. Displays comparison tables and grid visualizations
4. Prints academic analysis and conclusions

Usage:
    uv run grid-benchmark
    # or
    uv run python -m grid_path_benchmark.main
"""

from __future__ import annotations

import sys

import colorama
from colorama import Fore, Style
from grid_path_benchmark.comparison import (
    BenchmarkReport,
    format_academic_analysis,
    format_full_report,
    run_full_benchmark,
)
from grid_path_benchmark.grid_data import get_all_scenarios
from grid_path_benchmark.visualization import render_scenario_result

# ---------------------------------------------------------------------------
# Color constants (local shortcuts)
# ---------------------------------------------------------------------------

_R = Style.RESET_ALL
_D = Style.DIM
_H = Style.BRIGHT
_CYAN = Fore.CYAN + Style.BRIGHT
_GREEN = Fore.GREEN + Style.BRIGHT
_YELLOW = Fore.YELLOW + Style.BRIGHT


def _configure_encoding() -> None:
    """Configure stdout for UTF-8 and initialize colorama for ANSI support."""
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    colorama.init()


def _banner() -> None:
    """Print the colored application banner."""
    w = 80
    border = f"{_CYAN}+{'=' * (w - 2)}+{_R}"
    pad = w - 4  # inner width excluding "| " and " |"
    line1 = "GRID PATH BENCHMARK -- Search Algorithm Comparison on 2D Grids"
    line2 = "search-library v1.0.0 | grid-path-benchmark v1.0.0"
    print(border)
    print(f"{_CYAN}| {_H}{line1:<{pad}}{_R}{_CYAN} |{_R}")
    print(f"{_CYAN}| {_D}{line2:<{pad}}{_R}{_CYAN} |{_R}")
    print(border)


def _section(title: str) -> None:
    """Print a colored section divider."""
    print(f"{_CYAN}{'=' * 80}{_R}")
    print(f"  {_CYAN}{title}{_R}")
    print(f"{_CYAN}{'=' * 80}{_R}")


def _step(text: str) -> None:
    """Print a colored step marker."""
    print(f"{Fore.CYAN}▶{_R} {_H}{text}{_R}")


def main() -> None:
    """Run the complete grid pathfinding benchmark demonstration."""
    _configure_encoding()

    print()
    _banner()
    print()

    # --- Step 1: Load scenarios ---
    scenarios = get_all_scenarios()
    _step(f"Loaded {len(scenarios)} benchmark scenarios")
    for i, s in enumerate(scenarios, 1):
        print(f"  {_D}{i}.{_R} {s.name}")
    print()

    # --- Step 2: Run full benchmark ---
    _step("Running benchmark (all algorithms × all scenarios)...")
    print()
    benchmark = run_full_benchmark(scenarios)

    # --- Step 3: Display results ---
    print(format_full_report(benchmark))

    # --- Step 4: Grid visualizations ---
    _section("GRID VISUALIZATIONS (selected scenarios)")
    print()

    visualized = 0
    for sr in benchmark.scenario_reports:
        if visualized >= 3:
            break
        has_path = any(r.success and r.path_length > 1 for r in sr.results)
        if not has_path:
            continue

        print(f"  {_D}──{_R} {_CYAN}{sr.scenario.name}{_R} {_D}──{_R}")
        print()

        algo_to_show = "A*"
        result = next(
            (r for r in sr.results if r.algorithm_name == algo_to_show and r.success),
            None,
        )
        if result is None:
            algo_to_show = "Dijkstra"

        print(
            f"  {_D}Algorithm:{_R} {_YELLOW}{algo_to_show}{_R}"
            f"  {_D}|{_R}"
            f"  Cost: {_H}{next(r.total_cost for r in sr.results if r.algorithm_name == algo_to_show):.2f}{_R}"
            f"  {_D}|{_R}"
            f"  Nodes: {_H}{next(r.nodes_explored for r in sr.results if r.algorithm_name == algo_to_show)}{_R}"
        )
        print()
        print(render_scenario_result(sr, algo_to_show))
        print()
        visualized += 1

    # --- Step 5: Academic analysis ---
    print(format_academic_analysis(benchmark))

    # --- Step 6: Final summary ---
    _print_final_summary(benchmark)


def _print_final_summary(benchmark: BenchmarkReport) -> None:
    """Print a colored final summary of the benchmark run."""
    total_runs = 0
    total_successes = 0
    optimal_count = 0

    for sr in benchmark.scenario_reports:
        for r in sr.results:
            total_runs += 1
            if r.success:
                total_successes += 1
            if r.is_optimal:
                optimal_count += 1

    _section("BENCHMARK SUMMARY")
    print(f"  Scenarios:     {_H}{benchmark.total_scenarios}{_R}")
    print(f"  Total runs:    {_H}{total_runs}{_R}")
    print(
        f"  Paths found:   {_GREEN}{total_successes}{_R}{_D}/{_R}{total_runs}"
    )
    print(
        f"  Optimal paths: {_GREEN}{optimal_count}{_R}{_D}/{_R}{total_successes}"
    )
    print()
    print(f"  {_GREEN}✓{_R} Benchmark complete. Results are deterministic and reproducible.")
    print()


if __name__ == "__main__":
    main()
