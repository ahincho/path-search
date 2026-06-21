"""Extended heuristic functions for 2D grid pathfinding.

This module provides Chebyshev and Octile distance heuristics, complementing
the Manhattan and Euclidean heuristics available in search-library. Each
heuristic is designed for specific movement models:

- Manhattan: optimal for 4-directional grids with uniform cost
- Euclidean: admissible for any grid, but less tight than Octile for 8-dir
- Chebyshev: optimal for 8-directional grids with uniform diagonal cost (cost=1)
- Octile: optimal for 8-directional grids with diagonal cost = sqrt(2)

Admissibility and Consistency:
    All heuristics in this module are both admissible (h(n) <= h*(n)) and
    consistent (h(n) <= c(n,n') + h(n')) for their respective movement models.
    This guarantees A* will find optimal solutions.
"""

from __future__ import annotations

import math

from search_library.heuristics.base import Heuristic

# Type alias for grid positions
Position = tuple[int, int]


class ChebyshevHeuristic(Heuristic[Position]):
    """Chebyshev distance (L∞ norm) for 8-directional grids.

    Also known as chessboard distance. Computes max(|dx|, |dy|) which
    represents the minimum number of moves in an 8-directional grid where
    all moves (cardinal and diagonal) have equal cost of 1.

    Admissibility:
        h(n) = max(|dx|, |dy|) is admissible for 8-dir grids where
        diagonal movement costs 1. It equals the minimum number of
        moves required and never overestimates.

    Consistency:
        Satisfies triangle inequality for uniform-cost 8-dir movement.

    Use cases:
        - Chessboard movement (king moves)
        - 8-directional grids where all moves cost the same
    """

    def estimate(self, state: Position, goal: Position) -> float:
        """Calculate Chebyshev distance between two 2D positions.

        Args:
            state: Current position as (row, col).
            goal: Goal position as (row, col).

        Returns:
            Chebyshev distance (L∞ norm) between the positions.
        """
        dx = abs(state[0] - goal[0])
        dy = abs(state[1] - goal[1])
        return float(max(dx, dy))


class OctileHeuristic(Heuristic[Position]):
    """Octile distance heuristic for 8-directional grids with sqrt(2) diagonal cost.

    The octile distance is the tightest admissible heuristic for 8-directional
    grids where cardinal moves cost 1 and diagonal moves cost sqrt(2).

    Formula:
        h(n) = max(dx, dy) + (sqrt(2) - 1) * min(dx, dy)

    This represents: move diagonally min(dx,dy) times (each costing sqrt(2)),
    then move cardinally the remaining |dx - dy| times (each costing 1).

    Admissibility:
        The formula computes the exact optimal cost for an obstacle-free grid,
        thus h(n) <= h*(n) always holds.

    Consistency:
        For any adjacent cells a, b: h(a) <= c(a,b) + h(b).
        Proof follows from the L1/L2 properties of octile distance.

    Use cases:
        - Standard 8-directional grid pathfinding (most common in games/robotics)
        - When diagonal moves cost sqrt(2) times cardinal moves
    """

    _SQRT2_MINUS_1: float = math.sqrt(2) - 1.0

    def estimate(self, state: Position, goal: Position) -> float:
        """Calculate octile distance between two 2D positions.

        Args:
            state: Current position as (row, col).
            goal: Goal position as (row, col).

        Returns:
            Octile distance: the exact cost of the optimal path
            in an obstacle-free 8-directional grid.
        """
        dx = abs(state[0] - goal[0])
        dy = abs(state[1] - goal[1])
        return max(dx, dy) + self._SQRT2_MINUS_1 * min(dx, dy)
