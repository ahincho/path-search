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

from grid_path_benchmark.comparison import (
    format_academic_analysis,
    format_full_report,
    run_full_benchmark,
)
from grid_path_benchmark.grid_data import get_all_scenarios
from grid_path_benchmark.visualization import (
    render_scenario_result,
)


def _configure_encoding() -> None:
    """Configure stdout to handle UTF-8 on Windows."""
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def main() -> None:
    """Run the complete grid pathfinding benchmark demonstration."""
    _configure_encoding()

    print()
    print("+==============================================================================+")
    print("|     GRID PATH BENCHMARK -- Search Algorithm Comparison on 2D Grids           |")
    print("|     search-library v1.0.0 | grid-path-benchmark v1.0.0                       |")
    print("+==============================================================================+")
    print()

    # --- Step 1: Load scenarios ---
    scenarios = get_all_scenarios()
    print(f"▶ Loaded {len(scenarios)} benchmark scenarios")
    for i, s in enumerate(scenarios, 1):
        print(f"  {i}. {s.name}")
    print()

    # --- Step 2: Run full benchmark ---
    print("▶ Running benchmark (all algorithms × all scenarios)...")
    print()
    benchmark = run_full_benchmark(scenarios)

    # --- Step 3: Display results ---
    print(format_full_report(benchmark))

    # --- Step 4: Show grid visualizations for key scenarios ---
    print("=" * 80)
    print("  GRID VISUALIZATIONS (selected scenarios)")
    print("=" * 80)
    print()

    # Show visualization for first 3 non-trivial scenarios that found paths
    visualized = 0
    for sr in benchmark.scenario_reports:
        if visualized >= 3:
            break
        # Skip trivial (start==goal) and no-path scenarios
        has_path = any(r.success and r.path_length > 1 for r in sr.results)
        if not has_path:
            continue

        print(f"  ── {sr.scenario.name} ──")
        print()

        # Show A* path if available, otherwise Dijkstra
        algo_to_show = "A*"
        result = next(
            (r for r in sr.results if r.algorithm_name == algo_to_show and r.success),
            None,
        )
        if result is None:
            algo_to_show = "Dijkstra"

        grid_viz = render_scenario_result(sr, algo_to_show)
        print(f"  Algorithm: {algo_to_show}")
        print(grid_viz)
        print()
        visualized += 1

    # --- Step 5: Academic analysis ---
    print(format_academic_analysis(benchmark))

    # --- Step 6: Final summary ---
    _print_final_summary(benchmark)


def _print_final_summary(benchmark: object) -> None:
    """Print a final summary of the benchmark run."""
    from grid_path_benchmark.comparison import BenchmarkReport

    if not isinstance(benchmark, BenchmarkReport):
        return

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

    print("=" * 80)
    print("  BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"  Scenarios:        {benchmark.total_scenarios}")
    print(f"  Total runs:       {total_runs}")
    print(f"  Paths found:      {total_successes}/{total_runs}")
    print(f"  Optimal paths:    {optimal_count}/{total_successes}")
    print()
    print("  ✓ Benchmark complete. Results are deterministic and reproducible.")
    print()


if __name__ == "__main__":
    main()
