"""Tests for the grid scenario definitions and generators."""

from __future__ import annotations

import pytest
from grid_path_benchmark.grid_data import (
    Terrain,
    build_scenario_4dir_basic,
    build_scenario_8dir_basic,
    build_scenario_bug_trap,
    build_scenario_dense_obstacles,
    build_scenario_large_sparse,
    build_scenario_weighted_terrain,
    generate_bug_trap_grid,
    generate_grid,
    generate_weighted_grid,
    get_all_scenarios,
)


class TestTerrain:
    """Tests for the Terrain enum."""

    def test_terrain_costs(self) -> None:
        """Each terrain type should have the correct cost."""
        assert Terrain.FREE.cost == 1.0
        assert Terrain.DIFFICULT.cost == 2.0
        assert Terrain.WATER.cost == 5.0
        assert Terrain.BLOCKED.cost == float("inf")

    def test_passability(self) -> None:
        """Only BLOCKED terrain should be impassable."""
        assert Terrain.FREE.is_passable is True
        assert Terrain.DIFFICULT.is_passable is True
        assert Terrain.WATER.is_passable is True
        assert Terrain.BLOCKED.is_passable is False

    def test_symbols(self) -> None:
        """Each terrain should have a display symbol."""
        assert Terrain.FREE.symbol == " "
        assert Terrain.DIFFICULT.symbol == "~"
        assert Terrain.WATER.symbol == "≈"
        assert Terrain.BLOCKED.symbol == "█"


class TestGridScenario:
    """Tests for GridScenario configuration."""

    def test_scenario_immutable(self) -> None:
        """GridScenario should be frozen (immutable)."""
        scenario = build_scenario_4dir_basic()
        with pytest.raises(AttributeError):
            scenario.rows = 100  # type: ignore[misc]

    def test_all_scenarios_available(self) -> None:
        """get_all_scenarios should return all predefined scenarios."""
        scenarios = get_all_scenarios()
        assert len(scenarios) == 8

    def test_scenario_names_unique(self) -> None:
        """All scenario names should be unique."""
        scenarios = get_all_scenarios()
        names = [s.name for s in scenarios]
        assert len(names) == len(set(names))

    def test_scenarios_have_valid_dimensions(self) -> None:
        """All scenarios should have positive dimensions."""
        for scenario in get_all_scenarios():
            assert scenario.rows > 0
            assert scenario.cols > 0

    def test_scenarios_start_in_bounds(self) -> None:
        """All scenario starts should be within grid bounds."""
        for scenario in get_all_scenarios():
            r, c = scenario.start
            assert 0 <= r < scenario.rows
            assert 0 <= c < scenario.cols

    def test_scenarios_goal_in_bounds(self) -> None:
        """All scenario goals should be within grid bounds."""
        for scenario in get_all_scenarios():
            r, c = scenario.goal
            assert 0 <= r < scenario.rows
            assert 0 <= c < scenario.cols


class TestGenerateGrid:
    """Tests for the grid generator function."""

    def test_basic_generation(self) -> None:
        """Generate a basic grid from a scenario."""
        scenario = build_scenario_4dir_basic()
        grid = generate_grid(scenario)
        assert grid.rows == 20
        assert grid.cols == 20
        assert grid.allow_diagonal is False

    def test_8dir_generation(self) -> None:
        """Generate an 8-directional grid."""
        scenario = build_scenario_8dir_basic()
        grid = generate_grid(scenario)
        assert grid.allow_diagonal is True

    def test_start_not_obstacle(self) -> None:
        """Start position should never be an obstacle."""
        for scenario in get_all_scenarios():
            if scenario.weighted:
                grid = generate_weighted_grid(scenario)
            elif scenario.name.startswith("Bug Trap"):
                grid = generate_bug_trap_grid(scenario.rows, scenario.cols)
            else:
                grid = generate_grid(scenario)
            r, c = scenario.start
            assert not grid.is_obstacle(r, c), f"Start is obstacle in {scenario.name}"

    def test_goal_not_obstacle_in_solvable(self) -> None:
        """Goal should not be an obstacle in solvable scenarios."""
        solvable = [
            build_scenario_4dir_basic(),
            build_scenario_8dir_basic(),
            build_scenario_dense_obstacles(),
            build_scenario_bug_trap(),
            build_scenario_large_sparse(),
        ]
        for scenario in solvable:
            if scenario.weighted:
                grid = generate_weighted_grid(scenario)
            elif scenario.name.startswith("Bug Trap"):
                grid = generate_bug_trap_grid(scenario.rows, scenario.cols)
            else:
                grid = generate_grid(scenario)
            r, c = scenario.goal
            assert not grid.is_obstacle(r, c), f"Goal is obstacle in {scenario.name}"

    def test_deterministic_generation(self) -> None:
        """Same scenario should produce identical grids (determinism)."""
        scenario = build_scenario_4dir_basic()
        grid1 = generate_grid(scenario)
        grid2 = generate_grid(scenario)

        # Check same obstacles
        for r in range(scenario.rows):
            for c in range(scenario.cols):
                assert grid1.is_obstacle(r, c) == grid2.is_obstacle(r, c)

    def test_obstacle_density_approximate(self) -> None:
        """Generated obstacles should approximate the target density."""
        scenario = build_scenario_4dir_basic()  # 10% density
        grid = generate_grid(scenario)

        obstacle_count = sum(
            1
            for r in range(grid.rows)
            for c in range(grid.cols)
            if grid.is_obstacle(r, c)
        )
        total = grid.rows * grid.cols
        actual_density = obstacle_count / total

        # Allow some tolerance (start and goal exclusion)
        assert abs(actual_density - scenario.obstacle_density) < 0.05


class TestGenerateWeightedGrid:
    """Tests for the weighted grid generator."""

    def test_weighted_generation(self) -> None:
        """Weighted grid should have variable costs."""
        scenario = build_scenario_weighted_terrain()
        grid = generate_weighted_grid(scenario)

        # Should have some cells with cost > 1
        has_elevated_cost = False
        for r in range(grid.rows):
            for c in range(grid.cols):
                if grid.get_cost((r, c)) > 1.0:
                    has_elevated_cost = True
                    break
            if has_elevated_cost:
                break

        assert has_elevated_cost, "Weighted grid should have variable costs"

    def test_weighted_start_accessible(self) -> None:
        """Start should be accessible in weighted grid."""
        scenario = build_scenario_weighted_terrain()
        grid = generate_weighted_grid(scenario)
        r, c = scenario.start
        assert not grid.is_obstacle(r, c)

    def test_weighted_deterministic(self) -> None:
        """Weighted grid generation should be deterministic."""
        scenario = build_scenario_weighted_terrain()
        grid1 = generate_weighted_grid(scenario)
        grid2 = generate_weighted_grid(scenario)

        for r in range(scenario.rows):
            for c in range(scenario.cols):
                assert grid1.get_cost((r, c)) == grid2.get_cost((r, c))
                assert grid1.is_obstacle(r, c) == grid2.is_obstacle(r, c)


class TestBugTrapGrid:
    """Tests for the bug trap generator."""

    def test_bug_trap_has_obstacles(self) -> None:
        """Bug trap grid should have U-shaped obstacles."""
        grid = generate_bug_trap_grid(20, 20)
        obstacle_count = sum(
            1
            for r in range(20)
            for c in range(20)
            if grid.is_obstacle(r, c)
        )
        assert obstacle_count > 0, "Bug trap should have obstacles"

    def test_bug_trap_minimum_size(self) -> None:
        """Bug trap should enforce minimum grid size."""
        grid = generate_bug_trap_grid(5, 5)  # Too small, should clamp to 15
        assert grid.rows >= 15
        assert grid.cols >= 15

    def test_bug_trap_start_goal_clear(self) -> None:
        """Standard start/goal positions should be clear in bug trap."""
        grid = generate_bug_trap_grid(20, 20)
        # Standard positions (1,1) and (18,18) should be clear
        assert not grid.is_obstacle(1, 1)
        assert not grid.is_obstacle(18, 18)
