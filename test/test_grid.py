#!/usr/bin/env python3
"""Tests for the tri-state cost grid, its rasteriser and its neighbourhood."""

import numpy as np
import pytest
from ransac_seg.grid import (
    FREE,
    INFLATED,
    OCCUPIED,
    UNKNOWN,
    build_costmap,
    neighbors,
    plane_z,
    snap_to_traversable,
)
from ransac_seg.ground_plane import PlaneModel


def ground_patch(step=0.25, extent=10.0):
    """Return a dense flat sheet of ground returns covering a square patch."""
    axis = np.arange(-extent, extent + step, step)
    xs, ys = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])


def test_world_to_cell_round_trips_at_cell_centres():
    """A cell centre maps back to the cell it came from, scalars and arrays."""
    grid = build_costmap(ground_patch(), np.zeros((0, 3)), resolution=0.5)
    nx, ny = grid.shape

    for ix, iy in [(0, 0), (1, 3), (nx - 1, ny - 1)]:
        x, y = grid.cell_to_world(ix, iy)
        assert grid.world_to_cell(x, y) == (ix, iy)

    cells = np.array([[0, 0], [2, 5], [nx - 1, ny - 1]])
    xs, ys = grid.cell_to_world(cells[:, 0], cells[:, 1])
    back_x, back_y = grid.world_to_cell(xs, ys)
    np.testing.assert_array_equal(np.column_stack([back_x, back_y]), cells)


def test_origin_is_the_lower_left_corner_of_cell_zero():
    """Cell (0, 0) spans one resolution up and right of the origin."""
    grid = build_costmap(ground_patch(), np.zeros((0, 3)), resolution=0.5)
    x, y = grid.cell_to_world(0, 0)
    assert x == pytest.approx(grid.origin_x + 0.25)
    assert y == pytest.approx(grid.origin_y + 0.25)


def test_obstacle_beats_ground_in_the_same_cell():
    """A pole standing on inlier road must block that cell, not free it."""
    ground = np.array([[1.1, 1.1, 0.0]] * 20)
    obstacles = np.array([[1.1, 1.1, 1.0]] * 20)
    grid = build_costmap(ground, obstacles, resolution=0.5, bounds=(0.0, 2.0, 0.0, 2.0))

    cell = grid.world_to_cell(1.1, 1.1)
    assert grid.state[cell] == OCCUPIED
    assert not grid.is_traversable(*cell)


def test_unknown_is_distinct_from_free_and_untraversable():
    """Never-observed space is not drivable, and is distinguishable from road."""
    ground = np.array([[0.25, 0.25, 0.0]] * 5)
    grid = build_costmap(
        ground, np.zeros((0, 3)), resolution=0.5, bounds=(0.0, 5.0, 0.0, 5.0)
    )

    seen = grid.world_to_cell(0.25, 0.25)
    unseen = grid.world_to_cell(4.25, 4.25)

    assert grid.state[seen] == FREE
    assert grid.state[unseen] == UNKNOWN
    assert UNKNOWN != FREE
    assert grid.is_traversable(*seen)
    assert not grid.is_traversable(*unseen)
    assert not np.isfinite(grid.cost[unseen])


def test_single_obstacle_inflates_a_disc_of_the_right_radius():
    """Inflation covers roughly pi r^2 of ground, in metres not cells."""
    resolution = 0.25
    robot_radius = 1.0
    obstacles = np.array([[0.0, 0.0, 1.0]] * 10)
    grid = build_costmap(
        ground_patch(step=0.1),
        obstacles,
        resolution=resolution,
        robot_radius=robot_radius,
        bounds=(-10.0, 10.0, -10.0, 10.0),
    )

    inflated = int(np.count_nonzero(grid.state == INFLATED))
    occupied = int(np.count_nonzero(grid.state == OCCUPIED))
    # The disc is padded by half a cell diagonal, because the return that made
    # a cell OCCUPIED can sit that far from the centre the distance transform
    # measures from.
    padded = robot_radius + 0.5 * np.sqrt(2.0) * resolution
    analytic = np.pi * padded**2 / resolution**2

    assert occupied >= 1
    # The disc is quantised, and the occupied cells themselves are carved out
    # of it, so allow a generous band around the continuous area.
    assert 0.75 * analytic <= inflated + occupied <= 1.35 * analytic


def test_every_finite_cost_is_at_least_one():
    """A* admissibility depends on this: no step may be cheaper than its length."""
    rng = np.random.default_rng(5)
    obstacles = rng.uniform(-8.0, 8.0, size=(400, 2))
    grid = build_costmap(ground_patch(step=0.1), obstacles, resolution=0.25)

    finite = grid.cost[np.isfinite(grid.cost)]
    assert finite.size > 0
    assert finite.min() >= 1.0
    # Blocked states are never finitely costed.
    assert not np.isfinite(grid.cost[grid.state != FREE]).any()


def test_cost_decays_with_distance_from_the_obstacle():
    """Cost falls towards 1.0 as clearance grows, so paths centre themselves."""
    obstacles = np.array([[0.0, 0.0, 1.0]] * 10)
    grid = build_costmap(
        ground_patch(step=0.1),
        obstacles,
        resolution=0.25,
        robot_radius=0.5,
        inflation_radius=3.0,
        bounds=(-10.0, 10.0, -10.0, 10.0),
    )

    near = grid.cost[grid.world_to_cell(0.8, 0.0)]
    far = grid.cost[grid.world_to_cell(2.5, 0.0)]
    edge = grid.cost[grid.world_to_cell(8.0, 0.0)]

    assert 1.0 < far < near
    assert edge == pytest.approx(1.0)


def test_empty_inputs_give_a_one_cell_unknown_grid():
    """A blank frame must not raise; it yields nothing traversable."""
    grid = build_costmap(np.zeros((0, 3)), np.zeros((0, 2)))

    assert grid.shape == (1, 1)
    assert grid.state[0, 0] == UNKNOWN
    assert not grid.traversable_mask().any()
    assert grid.cells_with_state(FREE).shape == (0, 2)


def test_all_obstacle_input_leaves_nothing_traversable():
    """With no ground returns every observed cell is blocked."""
    rng = np.random.default_rng(2)
    obstacles = rng.uniform(-5.0, 5.0, size=(600, 3))
    grid = build_costmap(np.zeros((0, 3)), obstacles, resolution=0.5)

    assert not grid.traversable_mask().any()
    assert (grid.state != FREE).all()


def test_single_ground_point_is_free():
    """One return is enough at the default min_ground_points of 1."""
    grid = build_costmap(np.array([[2.0, 3.0, 0.0]]), np.zeros((0, 3)))

    cell = grid.world_to_cell(2.0, 3.0)
    assert grid.is_traversable(*cell)
    np.testing.assert_array_equal(grid.cells_with_state(FREE), [list(cell)])


def test_min_point_thresholds_are_respected():
    """Sparse hits below the thresholds do not commit a cell either way."""
    ground = np.array([[0.25, 0.25, 0.0]])
    obstacles = np.array([[1.25, 0.25, 1.0], [1.26, 0.26, 1.0]])
    grid = build_costmap(
        ground,
        obstacles,
        resolution=0.5,
        bounds=(0.0, 2.0, 0.0, 2.0),
        min_ground_points=2,
        min_obstacle_points=3,
    )

    assert grid.state[grid.world_to_cell(0.25, 0.25)] == UNKNOWN
    assert grid.state[grid.world_to_cell(1.25, 0.25)] == UNKNOWN


def test_points_outside_explicit_bounds_are_dropped():
    """An out-of-bounds return must be discarded, not wrapped to a valid cell."""
    ground = np.array([[0.25, 0.25, 0.0], [50.0, 50.0, 0.0], [-50.0, -50.0, 0.0]])
    grid = build_costmap(
        ground, np.zeros((0, 3)), resolution=0.5, bounds=(0.0, 2.0, 0.0, 2.0)
    )

    assert int(np.count_nonzero(grid.state == FREE)) == 1


def test_absurd_extent_is_clamped_to_the_ground():
    """A stray far-field return must not size the map into the millions."""
    ground = ground_patch(step=0.5, extent=5.0)
    obstacles = np.array([[10000.0, 10000.0, 1.0]] * 5)
    grid = build_costmap(ground, obstacles, resolution=0.05)

    nx, ny = grid.shape
    assert nx * ny <= 4_000_000
    assert grid.traversable_mask().any()


def test_cell_to_world_xyz_projects_onto_the_plane():
    """Cell centres are lifted onto the fitted road, or to z=0 without one."""
    tilt = np.radians(10.0)
    plane = PlaneModel(
        normal=np.array([-np.sin(tilt), 0.0, np.cos(tilt)]),
        offset=1.7,
        inlier_ratio=0.9,
        iterations=15,
        refined=True,
    )
    grid = build_costmap(ground_patch(), np.zeros((0, 3)), resolution=0.5, plane=plane)

    cells = grid.cells_with_state(FREE)[:20]
    xyz = grid.cell_to_world_xyz(cells)

    assert xyz.shape == (20, 3)
    np.testing.assert_allclose(plane.signed_distance_to(xyz), 0.0, atol=1e-9)

    flat = build_costmap(ground_patch(), np.zeros((0, 3)), resolution=0.5)
    np.testing.assert_array_equal(flat.cell_to_world_xyz(cells)[:, 2], 0.0)
    assert grid.cell_to_world_xyz(np.zeros((0, 2))).shape == (0, 3)


def test_plane_z_is_zero_for_a_vertical_normal():
    """A plane with no z component cannot be solved for z; degrade, don't blow up."""
    wall = PlaneModel(
        normal=np.array([1.0, 0.0, 0.0]),
        offset=0.0,
        inlier_ratio=0.5,
        iterations=5,
        refined=False,
    )
    np.testing.assert_array_equal(plane_z(wall, np.array([1.0, 2.0]), 0.0), [0.0, 0.0])
    np.testing.assert_array_equal(plane_z(None, np.array([1.0, 2.0]), 0.0), [0.0, 0.0])


def test_snap_returns_the_nearest_traversable_cell():
    """Snapping walks to the closest drivable cell within the search radius."""
    grid = build_costmap(ground_patch(), np.zeros((0, 3)), resolution=0.5)

    cell = snap_to_traversable(grid, 3.1, -2.4, radius=2.0)
    assert cell is not None
    assert grid.is_traversable(*cell)
    x, y = grid.cell_to_world(*cell)
    assert np.hypot(x - 3.1, y + 2.4) <= 0.5


def test_snap_returns_none_when_everything_in_radius_is_blocked():
    """No traversable cell nearby means None, not a blocked cell."""
    obstacles = np.repeat(ground_patch(step=0.25)[:, :2], 5, axis=0)
    grid = build_costmap(np.zeros((0, 3)), obstacles, resolution=0.5)

    assert snap_to_traversable(grid, 0.0, 0.0, radius=3.0) is None


def test_snap_ties_break_deterministically():
    """Equidistant candidates always resolve to the smaller (ix, iy)."""
    ground = np.array([[0.25, 0.25, 0.0], [0.75, 0.25, 0.0]])
    grid = build_costmap(
        ground, np.zeros((0, 3)), resolution=0.5, bounds=(0.0, 1.0, 0.0, 1.0)
    )

    midpoint_x = 0.5
    first = snap_to_traversable(grid, midpoint_x, 0.25, radius=1.0)
    assert first == min(
        tuple(c) for c in map(tuple, grid.cells_with_state(FREE).tolist())
    )


def test_neighbors_only_yields_traversable_in_bounds_cells():
    """Blocked, unknown and off-grid neighbours are never offered."""
    ground = ground_patch()
    obstacles = np.array([[0.6, 0.1, 1.0]] * 10)
    grid = build_costmap(ground, obstacles, resolution=0.5, robot_radius=0.4)

    for cell in map(tuple, grid.cells_with_state(FREE)[:200].tolist()):
        for nb, step in neighbors(grid, cell, connectivity=8):
            assert grid.in_bounds(*nb)
            assert grid.state[nb] == FREE
            assert np.isfinite(grid.cost[nb])
            assert step in (
                pytest.approx(0.5),
                pytest.approx(0.5 * np.sqrt(2.0)),
            )


def test_neighbors_connectivity_options():
    """4-connectivity offers only axis steps; anything else is a programming error."""
    grid = build_costmap(ground_patch(), np.zeros((0, 3)), resolution=0.5)
    centre = tuple(grid.cells_with_state(FREE)[len(grid.cells_with_state(FREE)) // 2])

    four = list(neighbors(grid, centre, connectivity=4))
    eight = list(neighbors(grid, centre, connectivity=8))
    assert len(four) == 4 and len(eight) == 8
    assert all(step == pytest.approx(0.5) for _, step in four)

    with pytest.raises(ValueError):
        list(neighbors(grid, centre, connectivity=6))


def test_diagonal_steps_do_not_cut_the_corner_between_two_blocked_cells():
    """
    An 8-connected step may not squeeze between two diagonally touching blockers.

    Without the check the search walked straight through an unbroken diagonal
    wall — a shorter "path" than the 4-connected detour, and solid rock.
    """
    axis = np.arange(0.0, 12.5, 0.5)
    xs, ys = np.meshgrid(axis, axis, indexing="ij")
    ground = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    wall = np.repeat(
        np.array([[k + 0.5, k + 0.5, 1.0] for k in range(1, 13)]), 5, axis=0
    )
    # robot_radius below the cell size, so the wall stays a bare diagonal of
    # OCCUPIED cells with no inflation to close the corners for us.
    grid = build_costmap(ground, wall, resolution=1.0, robot_radius=0.1)

    assert grid.cells_with_state(OCCUPIED).shape[0] >= 12, "the wall must survive"
    assert int(np.count_nonzero(grid.state == INFLATED)) == 0, "no inflation here"

    for cell in map(tuple, grid.cells_with_state(FREE).tolist()):
        for nb, _ in neighbors(grid, cell, connectivity=8):
            dx, dy = nb[0] - cell[0], nb[1] - cell[1]
            if dx and dy:
                assert grid.is_traversable(cell[0] + dx, cell[1])
                assert grid.is_traversable(cell[0], cell[1] + dy)


def test_free_cells_clear_the_robot_radius_from_the_actual_returns():
    """
    Clearance is measured to the obstacle return, not to its cell centre.

    The distance transform works centre to centre, so a return sitting in the
    corner of its cell used to look up to half a cell diagonal further away
    than it is; at the shipped 0.5 m resolution that let a cell 0.36 m from a
    real obstacle pass as clear of a 0.6 m robot.
    """
    resolution, robot_radius = 0.5, 0.6
    grid = build_costmap(
        ground_patch(step=0.25),
        np.zeros((0, 3)),
        resolution=resolution,
        robot_radius=robot_radius,
        bounds=(-10.0, 10.0, -10.0, 10.0),
    )
    # Park the return in the far corner of a cell, the worst case for the
    # centre-to-centre approximation.
    x0, y0 = grid.cell_to_world(20, 20)
    corner = np.array([[x0 + 0.49 * resolution, y0 + 0.49 * resolution, 1.0]] * 5)

    grid = build_costmap(
        ground_patch(step=0.25),
        corner,
        resolution=resolution,
        robot_radius=robot_radius,
        bounds=(-10.0, 10.0, -10.0, 10.0),
    )
    free = grid.cells_with_state(FREE)
    fx, fy = grid.cell_to_world(free[:, 0], free[:, 1])
    closest = float(np.min(np.hypot(fx - corner[0, 0], fy - corner[0, 1])))

    assert closest >= robot_radius


def test_an_unaffordable_resolution_coarsens_rather_than_crops():
    """
    Over the cell cap the map keeps the whole surface and grows the cell.

    Rescaling nx/ny at a fixed resolution used to shrink the map onto a corner
    of the observed ground and silently drop every return outside it, so the
    planner reported "no route" for anything beyond that corner.
    """
    ground = ground_patch(step=0.5, extent=30.0)
    grid = build_costmap(ground, np.zeros((0, 3)), resolution=0.02)
    nx, ny = grid.shape

    assert nx * ny <= 4_000_000
    assert grid.resolution > 0.02
    far = ground[np.argmax(np.hypot(ground[:, 0], ground[:, 1]))]
    assert grid.in_bounds(*grid.world_to_cell(far[0], far[1]))
    assert int(np.count_nonzero(grid.state == FREE)) > 0


def ring_returns(ranges, n_azimuth=720, y_span=None):
    """Return ground hits arranged in concentric rings, as a spinning LiDAR gives."""
    out = []
    for r in ranges:
        angles = np.linspace(-np.pi, np.pi, n_azimuth, endpoint=False)
        xs, ys = r * np.cos(angles), r * np.sin(angles)
        if y_span is not None:
            keep = (ys >= y_span[0]) & (ys <= y_span[1])
            xs, ys = xs[keep], ys[keep]
        out.append(np.column_stack([xs, ys, np.zeros(xs.size)]))
    return np.vstack(out)


def test_ray_carving_bridges_the_gaps_between_beam_rings():
    """
    Rings spaced wider than a cell leave UNKNOWN arcs that carving closes.

    This is the defect that made a real 60 m sweep unplannable past 13.5 m:
    every cell between two rings held no return and was therefore untraversable.
    """
    ground = ring_returns(np.arange(4.0, 30.0, 3.0))
    empty = np.zeros((0, 3))

    without = build_costmap(ground, empty, resolution=0.5, ray_fill=False)
    with_fill = build_costmap(ground, empty, resolution=0.5, ray_fill=True)

    assert int(np.count_nonzero(with_fill.state == FREE)) > 3 * int(
        np.count_nonzero(without.state == FREE)
    )
    # A point squarely between two rings must go from unobserved to drivable.
    cell = with_fill.world_to_cell(5.5, 0.0)
    assert without.state[cell] == UNKNOWN
    assert with_fill.state[cell] == FREE


def test_ray_carving_never_overrides_an_obstacle():
    """A return standing in a carved cell still blocks it."""
    ground = ring_returns(np.arange(4.0, 20.0, 3.0))
    obstacles = np.column_stack(
        [
            np.full(40, 8.0) + np.linspace(-0.2, 0.2, 40),
            np.zeros(40),
            np.full(40, 1.0),
        ]
    )
    grid = build_costmap(ground, obstacles, resolution=0.5, ray_fill=True)

    assert grid.state[grid.world_to_cell(8.0, 0.0)] == OCCUPIED


def test_ray_carving_stops_at_the_farthest_return():
    """Carving infers free space behind the sensor's reach, never past it."""
    ground = ring_returns(np.arange(4.0, 16.0, 2.0))
    grid = build_costmap(ground, np.zeros((0, 3)), resolution=0.5, ray_fill=True)

    free = grid.cells_with_state(FREE)
    fx, fy = grid.cell_to_world(free[:, 0], free[:, 1])
    # One cell diagonal of slack: the farthest ring sits at 14 m.
    assert float(np.max(np.hypot(fx, fy))) <= 14.0 + grid.resolution * 1.5


def test_a_lone_return_does_not_carve_a_spoke():
    """One stray hit on an azimuth is not evidence the whole ray was clear."""
    ground = np.array([[25.0, 0.0, 0.0], [3.0, 8.0, 0.0], [3.2, 8.1, 0.0]])
    grid = build_costmap(
        ground, np.zeros((0, 3)), resolution=0.5, ray_fill=True, ray_fill_min_cells=2
    )

    assert grid.state[grid.world_to_cell(12.0, 0.0)] == UNKNOWN


def test_ray_carving_reaches_the_sensor_blind_spot():
    """
    The unobserved disc around the vehicle becomes drivable.

    A real sweep returns nothing inside its minimum range, which left the ego
    origin untraversable and the planner with no start cell at all.
    """
    ground = ring_returns(np.arange(6.0, 24.0, 2.0))

    without = build_costmap(ground, np.zeros((0, 3)), resolution=0.5, ray_fill=False)
    with_fill = build_costmap(ground, np.zeros((0, 3)), resolution=0.5, ray_fill=True)

    assert snap_to_traversable(without, 0.0, 0.0, 2.0) is None
    assert snap_to_traversable(with_fill, 0.0, 0.0, 2.0) is not None
