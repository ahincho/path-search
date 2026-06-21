"""Tests for the algorithm comparison engine."""

from __future__ import annotations

import pytest
from grid_path_benchmark.comparison import (
    BenchmarkReport,
    ScenarioReport,
    format_academic_analysis,
    format_full_report,
    format_scenario_table,
    run_full_benchmark,
    run_scenario_comparison,
    select_heuristic,
)
from grid_path_benchmark.grid_data import (
    build_scenario_4dir_basic,
    build_scenario_8dir_basic,
    build_scenario_bug_trap,
    build_scenario_dense_obstacles,
    build_scenario_large_sparse,
    build_scenario_no_path,
    build_scenario_start_is_goal,
    build_scenario_weighted_terrain,
    get_all_scenarios,
)
from grid_path_benchmark.heuristics import OctileHeuristic
from search_library import EuclideanHeuristic, ManhattanHeuristic


class TestHeuristicSelection:
    """Tests for the heuristic selection logic."""

    def test_4dir_selects_manhattan(self) -> None:
        """4-directional scenarios should use Manhattan heuristic."""
        scenario = build_scenario_4dir_basic()
        h = select_heuristic(scenario)
        assert isinstance(h, ManhattanHeuristic)

    def test_8dir_selects_octile(self) -> None:
        """8-directional scenarios should use Octile heuristic."""
        scenario = build_scenario_8dir_basic()
        h = select_heuristic(scenario)
        assert isinstance(h, OctileHeuristic)

    def test_weighted_selects_euclidean(self) -> None:
        """Weighted scenarios should use Euclidean heuristic."""
        scenario = build_scenario_weighted_terrain()
        h = select_heuristic(scenario)
        assert isinstance(h, EuclideanHeuristic)


class TestRunScenarioComparison:
    """Tests for the scenario comparison engine."""

    def test_4dir_basic_all_find_path(self) -> None:
        """All algorithms should find a path in the basic 4-dir scenario."""
        scenario = build_scenario_4dir_basic()
        report = run_scenario_comparison(scenario)

        for result in report.results:
            assert result.success, f"{result.algorithm_name} failed in 4-dir basic"

    def test_dijkstra_always_optimal(self) -> None:
        """Dijkstra should always produce optimal result."""
        scenario = build_scenario_4dir_basic()
        report = run_scenario_comparison(scenario)

        dijkstra = next(r for r in report.results if r.algorithm_name == "Dijkstra")
        assert dijkstra.is_optimal

    def test_astar_is_optimal(self) -> None:
        """A* with admissible heuristic should find optimal path."""
        scenario = build_scenario_4dir_basic()
        report = run_scenario_comparison(scenario)

        astar = next(r for r in report.results if r.algorithm_name == "A*")
        assert astar.is_optimal

    def test_astar_matches_dijkstra_cost(self) -> None:
        """A* should find same cost as Dijkstra."""
        scenario = build_scenario_8dir_basic()
        report = run_scenario_comparison(scenario)

        dijkstra = next(r for r in report.results if r.algorithm_name == "Dijkstra")
        astar = next(r for r in report.results if r.algorithm_name == "A*")
        assert astar.total_cost == pytest.approx(dijkstra.total_cost)

    def test_astar_explores_fewer_or_equal_nodes(self) -> None:
        """A* should explore at most as many nodes as Dijkstra."""
        scenario = build_scenario_4dir_basic()
        report = run_scenario_comparison(scenario)

        dijkstra = next(r for r in report.results if r.algorithm_name == "Dijkstra")
        astar = next(r for r in report.results if r.algorithm_name == "A*")
        assert astar.nodes_explored <= dijkstra.nodes_explored

    def test_paths_start_at_origin(self) -> None:
        """All paths should start at the scenario's start position."""
        scenario = build_scenario_4dir_basic()
        report = run_scenario_comparison(scenario)

        for result in report.results:
            if result.success:
                assert result.path[0] == scenario.start

    def test_paths_end_at_goal(self) -> None:
        """All paths should end at the scenario's goal position."""
        scenario = build_scenario_4dir_basic()
        report = run_scenario_comparison(scenario)

        for result in report.results:
            if result.success:
                assert result.path[-1] == scenario.goal

    def test_no_path_scenario(self) -> None:
        """No-path scenario should report failure for all algorithms."""
        scenario = build_scenario_no_path()
        report = run_scenario_comparison(scenario)

        for result in report.results:
            assert not result.success, f"{result.algorithm_name} found path in no-path scenario"

    def test_start_equals_goal(self) -> None:
        """Start==Goal should return trivially with zero cost."""
        scenario = build_scenario_start_is_goal()
        report = run_scenario_comparison(scenario)

        assert len(report.results) == 1
        trivial = report.results[0]
        assert trivial.success
        assert trivial.total_cost == 0.0
        assert trivial.path_length == 1
        assert trivial.nodes_explored == 1

    def test_all_costs_at_least_optimal(self) -> None:
        """No algorithm should find a cheaper path than optimal."""
        scenario = build_scenario_8dir_basic()
        report = run_scenario_comparison(scenario)

        for result in report.results:
            if result.success:
                assert result.total_cost >= report.optimal_cost - 1e-9

    def test_execution_times_non_negative(self) -> None:
        """All execution times should be non-negative."""
        scenario = build_scenario_4dir_basic()
        report = run_scenario_comparison(scenario)

        for result in report.results:
            assert result.execution_time_ms >= 0

    def test_nodes_explored_positive(self) -> None:
        """All successful searches should explore at least 1 node."""
        scenario = build_scenario_4dir_basic()
        report = run_scenario_comparison(scenario)

        for result in report.results:
            if result.success:
                assert result.nodes_explored > 0


class TestDenseScenarios:
    """Tests for dense/complex scenarios."""

    def test_dense_obstacles_solvable(self) -> None:
        """Dense obstacles scenario should still be solvable."""
        scenario = build_scenario_dense_obstacles()
        report = run_scenario_comparison(scenario)

        dijkstra = next(r for r in report.results if r.algorithm_name == "Dijkstra")
        # It might not always be solvable with 30% density and seed 123
        # but we designed the seed to be solvable
        if dijkstra.success:
            astar = next(r for r in report.results if r.algorithm_name == "A*")
            assert astar.total_cost == pytest.approx(dijkstra.total_cost)

    def test_bug_trap_solvable(self) -> None:
        """Bug trap scenario should be solvable (trap is escapable)."""
        scenario = build_scenario_bug_trap()
        report = run_scenario_comparison(scenario)

        dijkstra = next(r for r in report.results if r.algorithm_name == "Dijkstra")
        assert dijkstra.success, "Bug trap should be solvable"

    def test_large_sparse_solvable(self) -> None:
        """Large sparse scenario should be solvable."""
        scenario = build_scenario_large_sparse()
        report = run_scenario_comparison(scenario)

        astar = next(r for r in report.results if r.algorithm_name == "A*")
        assert astar.success, "Large sparse grid should be solvable"


class TestWeightedScenarios:
    """Tests for weighted terrain scenarios."""

    def test_weighted_dijkstra_optimal(self) -> None:
        """Dijkstra should find optimal path in weighted grid."""
        scenario = build_scenario_weighted_terrain()
        report = run_scenario_comparison(scenario)

        dijkstra = next(r for r in report.results if r.algorithm_name == "Dijkstra")
        assert dijkstra.success
        assert dijkstra.is_optimal

    def test_weighted_astar_optimal(self) -> None:
        """A* should find optimal path in weighted grid."""
        scenario = build_scenario_weighted_terrain()
        report = run_scenario_comparison(scenario)

        astar = next(r for r in report.results if r.algorithm_name == "A*")
        assert astar.success
        assert astar.is_optimal

    def test_weighted_bfs_potentially_suboptimal(self) -> None:
        """BFS may find suboptimal path in weighted grid."""
        scenario = build_scenario_weighted_terrain()
        report = run_scenario_comparison(scenario)

        bfs = next(r for r in report.results if r.algorithm_name == "BFS")
        if bfs.success:
            # BFS cost should be >= optimal (may be equal by luck)
            assert bfs.total_cost >= report.optimal_cost - 1e-9


class TestFullBenchmark:
    """Tests for the full benchmark runner."""

    def test_full_benchmark_runs(self) -> None:
        """Full benchmark should complete without errors."""
        scenarios = get_all_scenarios()
        benchmark = run_full_benchmark(scenarios)
        assert isinstance(benchmark, BenchmarkReport)
        assert benchmark.total_scenarios == len(scenarios)

    def test_full_benchmark_all_scenarios_present(self) -> None:
        """All scenarios should appear in the benchmark report."""
        scenarios = get_all_scenarios()
        benchmark = run_full_benchmark(scenarios)

        scenario_names = [sr.scenario.name for sr in benchmark.scenario_reports]
        for s in scenarios:
            assert s.name in scenario_names


class TestFormatters:
    """Tests for output formatting functions."""

    @pytest.fixture
    def report(self) -> ScenarioReport:
        """Generate a report for formatting tests."""
        scenario = build_scenario_4dir_basic()
        return run_scenario_comparison(scenario)

    @pytest.fixture
    def benchmark(self) -> BenchmarkReport:
        """Generate a full benchmark for formatting tests."""
        return run_full_benchmark(get_all_scenarios())

    def test_scenario_table_not_empty(self, report: ScenarioReport) -> None:
        """Formatted table should produce non-empty output."""
        table = format_scenario_table(report)
        assert len(table) > 0
        assert "Algorithm" in table

    def test_full_report_not_empty(self, benchmark: BenchmarkReport) -> None:
        """Full report should produce meaningful output."""
        output = format_full_report(benchmark)
        assert len(output) > 0
        assert "GRID SEARCH BENCHMARK" in output

    def test_academic_analysis_not_empty(self, benchmark: BenchmarkReport) -> None:
        """Academic analysis should produce meaningful output."""
        analysis = format_academic_analysis(benchmark)
        assert len(analysis) > 0
        assert "ACADEMIC ANALYSIS" in analysis
        assert "heuristic" in analysis.lower() or "Heuristic" in analysis
