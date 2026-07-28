#!/usr/bin/env python3
"""
Tests for the costmap searches: optimality, admissibility and failure modes.

Pure numpy — no ROS import — so these run on a bare checkout. Every grid is
built through ``build_costmap`` from synthetic point sets rather than by
hand-filling arrays, so the tests exercise the same rasterisation the nodes do.
"""

import math

import numpy as np
import pytest
from ransac_seg.grid import FREE, OCCUPIED, build_costmap, snap_to_traversable
from ransac_seg.planners import (
    astar,
    bfs,
    dfs,
    dijkstra,
    furthest_reachable,
    manhattan,
    normalised_bands,
    octile,
    path_cost,
    path_length_m,
    reconstruct_path,
    select_goal,
)
from scipy.ndimage import distance_transform_edt

RESOLUTION = 0.5


def sheet(x_range, y_range, step=0.2, z=0.0):
    """Return a dense sheet of returns covering a rectangle at height ``z``."""
    xs = np.arange(x_range[0], x_range[1] + step, step)
    ys = np.arange(y_range[0], y_range[1] + step, step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)])


def corridor_grid(**kwargs):
    """Build a straight corridor bounded by obstacle walls at y = +/- 4."""
    ground = sheet((0.0, 20.0), (-4.0, 4.0))
    walls = np.vstack(
        [sheet((0.0, 20.0), (3.9, 4.4)), sheet((0.0, 20.0), (-4.4, -3.9))]
    )
    return build_costmap(ground, walls, resolution=RESOLUTION, **kwargs)


def random_grid(seed=7):
    """Build an open patch peppered with random obstacle blobs."""
    rng = np.random.default_rng(seed)
    ground = sheet((0.0, 15.0), (-7.0, 7.0))
    centres = np.column_stack([rng.uniform(2.0, 13.0, 12), rng.uniform(-6.0, 6.0, 12)])
    blobs = [
        np.column_stack(
            [
                rng.normal(cx, 0.2, 40),
                rng.normal(cy, 0.2, 40),
                np.full(40, 1.0),
            ]
        )
        for cx, cy in centres
    ]
    return build_costmap(ground, np.vstack(blobs), resolution=RESOLUTION)


def clearance_field(grid):
    """Return per-cell distance in metres to the nearest OCCUPIED cell."""
    occupied = grid.state == OCCUPIED
    if not occupied.any():
        return np.full(grid.shape, np.inf)
    return distance_transform_edt(~occupied, sampling=grid.resolution)


def brute_force_flood(grid, start):
    """Flood the traversable mask by hand, as an independent BFS reference."""
    mask = grid.traversable_mask()
    hops = {start: 0}
    frontier = [start]
    while frontier:
        nxt = []
        for ix, iy in frontier:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                cell = (ix + dx, iy + dy)
                if not (0 <= cell[0] < mask.shape[0] and 0 <= cell[1] < mask.shape[1]):
                    continue
                if mask[cell] and cell not in hops:
                    hops[cell] = hops[(ix, iy)] + 1
                    nxt.append(cell)
        frontier = nxt
    return hops


def test_octile_and_manhattan_scale_with_resolution():
    """Both metrics report metres, not cells."""
    assert octile((0, 0), (3, 0), 0.5) == pytest.approx(1.5)
    assert octile((0, 0), (3, 3), 0.5) == pytest.approx(3 * math.sqrt(2.0) * 0.5)
    assert manhattan((0, 0), (3, 3), 0.5) == pytest.approx(3.0)


def test_astar_matches_dijkstra_cost_at_the_goal():
    """A* is optimal: its total equals the Dijkstra field at the same goal."""
    grid = random_grid()
    start = snap_to_traversable(grid, 0.5, 0.0, 2.0)
    goal = snap_to_traversable(grid, 14.0, 0.0, 3.0)
    assert start is not None and goal is not None

    cost_field, parent = dijkstra(grid, start, connectivity=8)
    path, cost = astar(grid, start, goal, connectivity=8)

    assert path[0] == start and path[-1] == goal
    assert cost == pytest.approx(float(cost_field[goal]), rel=1e-4)
    # The reconstructed Dijkstra route must cost the same as A*'s, even when
    # the two pick different tie-broken routes of equal cost.
    assert path_cost(grid, reconstruct_path(parent, goal)) == pytest.approx(
        cost, rel=1e-4
    )


def test_astar_never_enters_a_blocked_or_unobserved_cell():
    """Every cell on the path is FREE with a finite cost."""
    # The band between the two swept sheets got no returns at all and stays
    # UNKNOWN, which the old set-of-cells planner could not tell from free
    # space; only the far end at x > 11 was actually observed as drivable.
    ground = np.vstack(
        [
            sheet((0.0, 15.0), (-6.0, -1.0)),
            sheet((0.0, 15.0), (1.0, 6.0)),
            sheet((11.0, 15.0), (-1.0, 1.0)),
        ]
    )
    obstacles = sheet((6.0, 7.0), (-6.0, -3.0), z=1.0)
    # Ray carving is off here on purpose. This scene is physically impossible
    # for a real sweep — it places ground returns at x > 11 on azimuths whose
    # nearer span is declared unobserved, and a beam that reached 13 m must
    # have passed over 5 m. Carving is exercised in test_grid.py; what this
    # test isolates is the planner's own refusal to cross UNKNOWN, so it needs
    # a grid that still has an UNKNOWN band in it.
    grid = build_costmap(ground, obstacles, resolution=RESOLUTION, ray_fill=False)

    start = snap_to_traversable(grid, 0.5, -4.0, 2.0)
    goal = snap_to_traversable(grid, 14.0, 4.0, 3.0)
    path, cost = astar(grid, start, goal, connectivity=8)

    assert path, "expected a route around the obstacle"
    assert math.isfinite(cost)
    states = np.array([grid.state[c] for c in path])
    costs = np.array([grid.cost[c] for c in path])
    assert np.all(states == FREE)
    assert np.all(np.isfinite(costs))
    # It had to detour to the observed crossing rather than cut straight
    # across the unobserved band, which spans x < 11 at |y| < 0.5.
    centres = np.array([grid.cell_to_world(*c) for c in path])
    crossing = centres[np.abs(centres[:, 1]) < 0.5]
    assert crossing.shape[0] > 0
    assert float(crossing[:, 0].min()) > 10.5


def test_astar_prefers_the_cheap_centre_over_the_wall_hugging_shortcut():
    """The cost field pulls the route off the wall when the detour is cheap."""
    grid = corridor_grid(inflation_radius=3.0, cost_scaling=1.0)
    start = snap_to_traversable(grid, 1.0, 3.0, 1.0)
    goal = snap_to_traversable(grid, 19.0, 3.0, 1.0)
    assert start is not None and goal is not None

    path, cost = astar(grid, start, goal, connectivity=8)
    assert path and math.isfinite(cost)

    clearance = clearance_field(grid)
    chosen = min(float(clearance[c]) for c in path)

    # Reference: the geometrically shortest route, straight along y = 3.
    straight = [
        (ix, start[1])
        for ix in range(min(start[0], goal[0]), max(start[0], goal[0]) + 1)
    ]
    assert all(grid.is_traversable(*c) for c in straight)
    straight_clearance = min(float(clearance[c]) for c in straight)

    assert chosen > straight_clearance
    assert path_length_m(grid, path) > path_length_m(grid, straight)
    assert path_cost(grid, path) < path_cost(grid, straight)


def test_walled_off_goal_returns_empty_path_and_infinite_cost():
    """A sealed goal yields ([], inf) rather than a truncated best effort."""
    ground = sheet((0.0, 20.0), (-4.0, 4.0))
    wall = sheet((9.5, 10.5), (-4.5, 4.5), z=1.0)
    grid = build_costmap(ground, wall, resolution=RESOLUTION)

    start = snap_to_traversable(grid, 1.0, 0.0, 2.0)
    goal = snap_to_traversable(grid, 18.0, 0.0, 2.0)
    path, cost = astar(grid, start, goal, connectivity=8)

    assert path == []
    assert cost == math.inf


def test_heuristic_never_exceeds_the_true_cost():
    """Admissibility on 200 random start/goal pairs, both connectivities."""
    grid = random_grid(seed=11)
    traversable = np.argwhere(grid.traversable_mask())
    rng = np.random.default_rng(3)

    for connectivity, heuristic in ((8, octile), (4, manhattan)):
        starts = traversable[rng.integers(0, traversable.shape[0], 20)]
        for sx, sy in starts:
            cost_field, _ = dijkstra(grid, (sx, sy), connectivity=connectivity)
            reachable = np.argwhere(np.isfinite(cost_field))
            goals = reachable[rng.integers(0, reachable.shape[0], 10)]
            for gx, gy in goals:
                true_cost = float(cost_field[gx, gy])
                estimate = heuristic((sx, sy), (gx, gy), grid.resolution)
                # The field is stored as float32, so a route whose true cost
                # equals the heuristic exactly can read back a few ulps low.
                assert estimate <= true_cost + max(1e-9, 1e-6 * true_cost)


def test_bfs_matches_a_brute_force_flood():
    """The vectorised hop field agrees cell-for-cell with a naive flood."""
    grid = random_grid(seed=5)
    start = snap_to_traversable(grid, 0.5, 0.0, 2.0)
    hops = bfs(grid, start, connectivity=4)
    reference = brute_force_flood(grid, start)

    assert set(zip(*np.where(hops >= 0))) == set(reference)
    for cell, depth in reference.items():
        assert int(hops[cell]) == depth


def test_dfs_and_bfs_visit_exactly_the_same_cells():
    """Reachability is order-independent: DFS finds no cell BFS misses."""
    grid = random_grid(seed=13)
    start = snap_to_traversable(grid, 0.5, 0.0, 2.0)

    hops = bfs(grid, start, connectivity=4)
    order = dfs(grid, start, connectivity=4)

    bfs_cells = set(zip(*np.where(hops >= 0)))
    dfs_cells = set(zip(*np.where(order >= 0)))
    assert bfs_cells == dfs_cells
    # DFS orders that set differently, which is the only thing it adds.
    assert not np.array_equal(hops[hops >= 0], order[order >= 0])


def test_dfs_is_iterative_on_a_grid_that_would_blow_the_recursion_limit():
    """A 300x300 open grid completes; a recursive DFS could not."""
    ground = sheet((0.0, 149.0), (0.0, 149.0), step=0.25)
    grid = build_costmap(ground, np.zeros((0, 3)), resolution=RESOLUTION)
    assert min(grid.shape) >= 300

    start = snap_to_traversable(grid, 1.0, 1.0, 2.0)
    order = dfs(grid, start, connectivity=4)
    assert int((order >= 0).sum()) == int(grid.traversable_mask().sum())


def test_searches_reject_an_unsupported_connectivity():
    """Only 4 and 8 are neighbourhoods this grid knows how to walk."""
    grid = corridor_grid()
    start = snap_to_traversable(grid, 1.0, 0.0, 2.0)
    for call in (
        lambda: astar(grid, start, start, connectivity=6),
        lambda: dijkstra(grid, start, connectivity=6),
        lambda: bfs(grid, start, connectivity=6),
        lambda: dfs(grid, start, connectivity=6),
    ):
        with pytest.raises(ValueError):
            call()


def test_search_from_an_untraversable_start_is_empty_not_an_exception():
    """A start inside an obstacle yields empty fields, never a KeyError."""
    grid = corridor_grid()
    blocked = tuple(grid.cells_with_state(OCCUPIED)[0])

    assert astar(grid, blocked, blocked, connectivity=8) == ([], math.inf)
    cost_field, parent = dijkstra(grid, blocked)
    assert not np.isfinite(cost_field).any()
    assert int(parent.max()) == -1
    assert int(bfs(grid, blocked).max()) == -1
    assert int(dfs(grid, blocked).max()) == -1


def test_reconstruct_path_returns_empty_for_an_unreachable_goal():
    """No parent chain means no path, not a partial one."""
    ground = sheet((0.0, 20.0), (-4.0, 4.0))
    wall = sheet((9.5, 10.5), (-4.5, 4.5), z=1.0)
    grid = build_costmap(ground, wall, resolution=RESOLUTION)

    start = snap_to_traversable(grid, 1.0, 0.0, 2.0)
    goal = snap_to_traversable(grid, 18.0, 0.0, 2.0)
    _, parent = dijkstra(grid, start)

    assert reconstruct_path(parent, goal) == []
    assert reconstruct_path(parent, start) == [start]


def test_select_goal_returns_none_when_boxed_in():
    """With nothing traversable on the map there is no goal to snap to."""
    patch = sheet((0.0, 10.0), (-3.0, 3.0))
    grid = build_costmap(
        patch,
        patch + np.array([0.0, 0.0, 1.0]),
        resolution=RESOLUTION,
        min_obstacle_points=1,
    )
    assert not grid.traversable_mask().any()
    assert select_goal(grid, (5.0, 0.0), search_radius=5.0) is None


def test_select_goal_pulls_an_off_map_request_back_along_the_axis():
    """A goal past the end of this frame's map lands on its far edge."""
    grid = corridor_grid()
    cell = select_goal(grid, (200.0, 0.0), search_radius=2.0, fallback_axis=(1.0, 0.0))
    assert cell is not None
    x, _ = grid.cell_to_world(*cell)
    assert x > 18.0


def test_furthest_reachable_only_offers_connected_cells():
    """The fallback goal never lands on the far side of a wall."""
    ground = sheet((0.0, 20.0), (-4.0, 4.0))
    wall = sheet((9.5, 10.5), (-4.5, 4.5), z=1.0)
    grid = build_costmap(ground, wall, resolution=RESOLUTION)

    start = snap_to_traversable(grid, 1.0, 0.0, 2.0)
    cost_field, _ = dijkstra(grid, start)
    cell = furthest_reachable(cost_field, grid, (1.0, 0.0))

    assert cell is not None
    assert math.isfinite(float(cost_field[cell]))
    x, _ = grid.cell_to_world(*cell)
    assert x < 9.5

    with pytest.raises(ValueError):
        furthest_reachable(cost_field, grid, (0.0, 0.0))


def test_normalised_bands_covers_the_range_and_marks_invalid_cells():
    """Band 0 and the top band both appear; invalid cells stay at -1."""
    values = np.array([[0.0, 1.0], [np.inf, 3.0]])
    valid = np.array([[True, True], [True, True]])
    index = normalised_bands(values, valid, 4)

    assert index[1, 0] == -1
    assert index[0, 0] == 0
    assert index[1, 1] == 3
    assert set(np.unique(index[index >= 0])).issubset({0, 1, 2, 3})

    flat = normalised_bands(np.full((2, 2), 5.0), valid, 4)
    assert np.all(flat == 0)


def _diagonal_wall_grid():
    """Build a 12 m square of ground split corner to corner by a 1-cell wall."""
    axis = np.arange(0.0, 12.5, 0.5)
    xs, ys = np.meshgrid(axis, axis, indexing="ij")
    ground = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    wall = np.repeat(
        np.array([[k + 0.5, k + 0.5, 1.0] for k in range(1, 13)]), 5, axis=0
    )
    return build_costmap(ground, wall, resolution=1.0, robot_radius=0.1)


def test_astar_does_not_drive_through_a_diagonal_wall():
    """
    An unbroken diagonal of blocked cells is a wall, not a permeable lattice.

    8-connected search used to step from (5, 6) to (6, 5) between OCCUPIED
    (5, 5) and (6, 6) and call it a 13.6 m route, while the 4-connected search
    correctly paid 30.9 m to go around. The two must now agree that the far
    side is on the far side.
    """
    grid = _diagonal_wall_grid()
    start, goal = (1, 10), (10, 1)
    assert grid.is_traversable(*start) and grid.is_traversable(*goal)

    for connectivity in (4, 8):
        path, cost = astar(grid, start, goal, connectivity=connectivity)
        assert path, f"connectivity {connectivity} found no route at all"
        for a, b in zip(path[:-1], path[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            if dx and dy:
                assert grid.is_traversable(a[0] + dx, a[1])
                assert grid.is_traversable(a[0], a[1] + dy)
        # Any legal route has to walk around the wall's near end, which costs
        # strictly more than the 13.58 the corner-cutting shortcut reported.
        assert min(cell[0] for cell in path) <= 1
        assert cost > 15.0
