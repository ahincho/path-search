"""Tests for the extended heuristic functions (Chebyshev, Octile)."""

from __future__ import annotations

import math

import pytest
from grid_path_benchmark.heuristics import ChebyshevHeuristic, OctileHeuristic


class TestChebyshevHeuristic:
    """Tests for the Chebyshev (L∞) distance heuristic."""

    @pytest.fixture
    def heuristic(self) -> ChebyshevHeuristic:
        """Create a Chebyshev heuristic instance."""
        return ChebyshevHeuristic()

    def test_same_position_zero(self, heuristic: ChebyshevHeuristic) -> None:
        """Distance from a point to itself is zero."""
        assert heuristic.estimate((5, 5), (5, 5)) == 0.0

    def test_cardinal_distance(self, heuristic: ChebyshevHeuristic) -> None:
        """Pure cardinal movement should return the distance."""
        assert heuristic.estimate((0, 0), (0, 5)) == 5.0
        assert heuristic.estimate((0, 0), (3, 0)) == 3.0

    def test_diagonal_distance(self, heuristic: ChebyshevHeuristic) -> None:
        """Pure diagonal movement should return max component."""
        assert heuristic.estimate((0, 0), (3, 3)) == 3.0
        assert heuristic.estimate((0, 0), (5, 5)) == 5.0

    def test_mixed_distance(self, heuristic: ChebyshevHeuristic) -> None:
        """Mixed movement returns max of dx, dy."""
        assert heuristic.estimate((0, 0), (3, 7)) == 7.0
        assert heuristic.estimate((0, 0), (8, 2)) == 8.0

    def test_non_negative(self, heuristic: ChebyshevHeuristic) -> None:
        """All distances should be non-negative."""
        positions = [(0, 0), (5, 3), (10, 10), (0, 15)]
        for a in positions:
            for b in positions:
                assert heuristic.estimate(a, b) >= 0.0

    def test_symmetry(self, heuristic: ChebyshevHeuristic) -> None:
        """Distance should be symmetric: d(a,b) == d(b,a)."""
        pairs = [((0, 0), (3, 5)), ((2, 7), (9, 1)), ((0, 0), (0, 0))]
        for a, b in pairs:
            assert heuristic.estimate(a, b) == heuristic.estimate(b, a)

    def test_triangle_inequality(self, heuristic: ChebyshevHeuristic) -> None:
        """Chebyshev distance must satisfy triangle inequality."""
        points = [(0, 0), (3, 4), (7, 1), (2, 8), (5, 5)]
        for a in points:
            for b in points:
                for c in points:
                    d_ac = heuristic.estimate(a, c)
                    d_ab = heuristic.estimate(a, b)
                    d_bc = heuristic.estimate(b, c)
                    assert d_ac <= d_ab + d_bc + 1e-9

    def test_callable_interface(self, heuristic: ChebyshevHeuristic) -> None:
        """Heuristic should be callable via __call__."""
        result = heuristic((0, 0), (3, 4))
        assert result == heuristic.estimate((0, 0), (3, 4))

    def test_admissibility_8dir_uniform(self, heuristic: ChebyshevHeuristic) -> None:
        """Chebyshev is admissible for 8-dir grids with uniform cost 1.

        For any two positions, Chebyshev <= actual path cost when all
        moves (including diagonal) cost 1.
        """
        # Actual cost of (0,0)->(3,4) in 8-dir uniform = max(3,4) = 4 moves
        # Chebyshev gives exactly 4
        assert heuristic.estimate((0, 0), (3, 4)) == 4.0


class TestOctileHeuristic:
    """Tests for the Octile distance heuristic."""

    @pytest.fixture
    def heuristic(self) -> OctileHeuristic:
        """Create an Octile heuristic instance."""
        return OctileHeuristic()

    def test_same_position_zero(self, heuristic: OctileHeuristic) -> None:
        """Distance from a point to itself is zero."""
        assert heuristic.estimate((5, 5), (5, 5)) == 0.0

    def test_cardinal_distance(self, heuristic: OctileHeuristic) -> None:
        """Pure cardinal movement: octile == Manhattan == distance."""
        assert heuristic.estimate((0, 0), (0, 5)) == pytest.approx(5.0)
        assert heuristic.estimate((0, 0), (7, 0)) == pytest.approx(7.0)

    def test_diagonal_distance(self, heuristic: OctileHeuristic) -> None:
        """Pure diagonal: octile = n * sqrt(2)."""
        # (0,0) -> (3,3): 3 diagonal moves, cost = 3 * sqrt(2)
        expected = 3 * math.sqrt(2)
        assert heuristic.estimate((0, 0), (3, 3)) == pytest.approx(expected)

    def test_mixed_distance(self, heuristic: OctileHeuristic) -> None:
        """Mixed: move diagonally min(dx,dy) times, then cardinally."""
        # (0,0) -> (3,5): dx=3, dy=5
        # 3 diagonal + 2 cardinal = 3*sqrt(2) + 2
        expected = 3 * math.sqrt(2) + 2
        assert heuristic.estimate((0, 0), (3, 5)) == pytest.approx(expected)

    def test_formula_correctness(self, heuristic: OctileHeuristic) -> None:
        """Verify octile formula: max(dx,dy) + (sqrt(2)-1)*min(dx,dy)."""
        cases = [
            ((0, 0), (4, 7), max(4, 7) + (math.sqrt(2) - 1) * min(4, 7)),
            ((2, 3), (8, 5), max(6, 2) + (math.sqrt(2) - 1) * min(6, 2)),
        ]
        for a, b, expected in cases:
            assert heuristic.estimate(a, b) == pytest.approx(expected)

    def test_non_negative(self, heuristic: OctileHeuristic) -> None:
        """All distances should be non-negative."""
        positions = [(0, 0), (5, 3), (10, 10), (0, 15)]
        for a in positions:
            for b in positions:
                assert heuristic.estimate(a, b) >= 0.0

    def test_symmetry(self, heuristic: OctileHeuristic) -> None:
        """Octile distance should be symmetric."""
        pairs = [((0, 0), (3, 5)), ((2, 7), (9, 1)), ((10, 0), (0, 10))]
        for a, b in pairs:
            assert heuristic.estimate(a, b) == pytest.approx(heuristic.estimate(b, a))

    def test_triangle_inequality(self, heuristic: OctileHeuristic) -> None:
        """Octile distance must satisfy triangle inequality."""
        points = [(0, 0), (3, 4), (7, 1), (2, 8), (5, 5)]
        for a in points:
            for b in points:
                for c in points:
                    d_ac = heuristic.estimate(a, c)
                    d_ab = heuristic.estimate(a, b)
                    d_bc = heuristic.estimate(b, c)
                    assert d_ac <= d_ab + d_bc + 1e-9

    def test_admissibility_8dir_sqrt2(self, heuristic: OctileHeuristic) -> None:
        """Octile is admissible for 8-dir grids with sqrt(2) diagonal cost.

        Octile gives the exact cost in obstacle-free grids, so it's
        always <= actual cost (with obstacles, actual cost >= obstacle-free).
        """
        # (0,0) -> (0,5): pure cardinal, cost = 5
        assert heuristic.estimate((0, 0), (0, 5)) <= 5.0 + 1e-9
        # (0,0) -> (3,3): pure diagonal, cost = 3*sqrt(2)
        assert heuristic.estimate((0, 0), (3, 3)) <= 3 * math.sqrt(2) + 1e-9

    def test_octile_tighter_than_euclidean(self, heuristic: OctileHeuristic) -> None:
        """Octile should be >= Euclidean for all positions.

        Octile is tighter (larger or equal) than Euclidean for 8-dir grids,
        making A* with Octile more efficient.
        """
        from search_library import EuclideanHeuristic

        euclidean = EuclideanHeuristic()
        test_pairs = [((0, 0), (5, 3)), ((0, 0), (10, 10)), ((2, 1), (8, 7))]
        for a, b in test_pairs:
            oct_dist = heuristic.estimate(a, b)
            euc_dist = euclidean.estimate(a, b)
            assert oct_dist >= euc_dist - 1e-9
