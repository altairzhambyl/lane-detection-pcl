#!/usr/bin/env python3
"""
Tri-state cost grid shared by the perception and planning halves of the stack.

The nodes used to quantise ground returns into a Python ``set`` of cell keys.
That representation cannot tell "no data" from "free", so a planner reading it
happily routed through the region behind a wall where the LiDAR had no returns
at all, and it carried no clearance margin, so paths hugged obstacles. The grid
here keeps UNKNOWN distinct from FREE, lets an obstacle override the ground it
stands on, and inflates obstacles by the robot radius so the planner does not
have to know the vehicle's width.

Costs are the planner's contract: every finite cost is >= 1.0, so a metric
step-length heuristic stays admissible for A*.

Pure numpy/scipy — no ROS import — so planner tests run on a bare checkout.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

UNKNOWN = -1
FREE = 0
OCCUPIED = 1
INFLATED = 2

# A sweep that produces a few million cells is a bug (bad units, a stray return
# at 10 km), not a map; rasterising it would stall the callback.
MAX_CELLS = 4_000_000

_SQRT2 = float(np.sqrt(2.0))
_STEPS_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_STEPS_8 = _STEPS_4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))


def plane_z(plane, x, y):
    """
    Solve the fitted plane for z at the given xy; zeros when plane is ``None``.

    A near-horizontal normal makes the solve blow up, so it degrades to zero
    rather than projecting markers to infinity.
    """
    zeros = np.zeros_like(np.asarray(x, dtype=np.float64))
    if plane is None:
        return zeros
    normal = np.asarray(plane.normal, dtype=np.float64)
    if abs(float(normal[2])) < 1e-6:
        return zeros
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    return -(normal[0] * xs + normal[1] * ys + float(plane.offset)) / float(normal[2])


@dataclass(frozen=True)
class CostGrid:
    """A rasterised, inflated traversability map in the sensor's xy plane."""

    origin_x: float
    origin_y: float
    resolution: float
    state: np.ndarray
    cost: np.ndarray
    plane: object

    @property
    def shape(self):
        """Return the ``(nx, ny)`` cell dimensions."""
        return self.state.shape

    def in_bounds(self, ix, iy):
        """Return True when the cell index lies inside the grid."""
        nx, ny = self.state.shape
        return 0 <= int(ix) < nx and 0 <= int(iy) < ny

    def world_to_cell(self, x, y):
        """Return the ``(ix, iy)`` cell containing a world point."""
        ix = np.floor(
            (np.asarray(x, dtype=np.float64) - self.origin_x) / self.resolution
        )
        iy = np.floor(
            (np.asarray(y, dtype=np.float64) - self.origin_y) / self.resolution
        )
        if np.ndim(x) == 0 and np.ndim(y) == 0:
            return int(ix), int(iy)
        return ix.astype(np.int32), iy.astype(np.int32)

    def cell_to_world(self, ix, iy):
        """Return the world ``(x, y)`` of a cell centre."""
        x = self.origin_x + (np.asarray(ix, dtype=np.float64) + 0.5) * self.resolution
        y = self.origin_y + (np.asarray(iy, dtype=np.float64) + 0.5) * self.resolution
        if np.ndim(ix) == 0 and np.ndim(iy) == 0:
            return float(x), float(y)
        return x, y

    def cell_to_world_xyz(self, cells):
        """Return an ``(M, 3)`` array of cell centres lifted onto the plane."""
        idx = np.asarray(cells, dtype=np.int64).reshape(-1, 2)
        if idx.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float64)
        x = self.origin_x + (idx[:, 0] + 0.5) * self.resolution
        y = self.origin_y + (idx[:, 1] + 0.5) * self.resolution
        z = np.broadcast_to(plane_z(self.plane, x, y), x.shape)
        return np.column_stack([x, y, z]).astype(np.float64)

    def is_traversable(self, ix, iy):
        """Return True when a cell is in bounds, FREE and finitely costed."""
        if not self.in_bounds(ix, iy):
            return False
        ix, iy = int(ix), int(iy)
        return bool(self.state[ix, iy] == FREE and np.isfinite(self.cost[ix, iy]))

    def traversable_mask(self):
        """Return the ``(nx, ny)`` boolean mask of traversable cells."""
        return (self.state == FREE) & np.isfinite(self.cost)

    def cells_with_state(self, value):
        """Return an ``(M, 2)`` int32 array of the cells holding ``value``."""
        return np.argwhere(self.state == value).astype(np.int32)


def _xy(points):
    """Coerce an ``(N, 2)`` or ``(N, 3)`` array to finite ``(N, 2)`` float xy."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    arr = arr.reshape(-1, arr.shape[-1])[:, :2]
    return arr[np.isfinite(arr).all(axis=1)]


def _extent(*clouds):
    """Return ``(xmin, xmax, ymin, ymax)`` over the non-empty clouds, or None."""
    non_empty = [c for c in clouds if c.shape[0] > 0]
    if not non_empty:
        return None
    stacked = np.vstack(non_empty)
    return (
        float(stacked[:, 0].min()),
        float(stacked[:, 0].max()),
        float(stacked[:, 1].min()),
        float(stacked[:, 1].max()),
    )


def _dims(bounds, resolution):
    """Return ``(origin_x, origin_y, nx, ny)`` for a padded bounding box."""
    xmin, xmax, ymin, ymax = bounds
    origin_x = xmin - resolution
    origin_y = ymin - resolution
    nx = max(1, int(np.ceil((xmax + resolution - origin_x) / resolution)))
    ny = max(1, int(np.ceil((ymax + resolution - origin_y) / resolution)))
    return origin_x, origin_y, nx, ny


def _counts(cells, nx, ny):
    """Accumulate per-cell hit counts from an ``(M, 2)`` index array."""
    counts = np.zeros((nx, ny), dtype=np.int32)
    if cells.shape[0]:
        np.add.at(counts, (cells[:, 0], cells[:, 1]), 1)
    return counts


def _quantise(xy, origin_x, origin_y, resolution, nx, ny):
    """Map world xy to in-bounds integer cell indices, dropping the rest."""
    if xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    ix = np.floor((xy[:, 0] - origin_x) / resolution).astype(np.int64)
    iy = np.floor((xy[:, 1] - origin_y) / resolution).astype(np.int64)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    return np.column_stack([ix[inside], iy[inside]])


def _carve_ray_free(state, origin_x, origin_y, resolution, sensor_xy, bins, min_cells):
    """
    Mark cells the sensor must have seen over as FREE, in place.

    A spinning LiDAR samples in rings, so the spacing between consecutive
    ground returns grows with range — quadratically, for a fixed vertical
    angular resolution. Rasterising returns alone therefore leaves the map
    striped with UNKNOWN arcs that no planner can cross: measured on a real
    77k-point sweep, reachability died at 13.5 m of a 60 m map.

    Those arcs are not unobserved. If a beam reached ground at 40 m along some
    azimuth, everything nearer along that azimuth was empty at ground height,
    or the beam would have been blocked first. So per azimuth bin, take the
    farthest ground return and carve everything closer to it free. This is the
    same free-space reasoning an occupancy grid uses for its rays, restricted
    to the ground layer.

    It cannot see negative obstacles: a trench or a kerb drop between the
    sensor and a far return is carved free like everything else. That is a real
    limitation of ground-layer carving and the reason this only ever *adds*
    FREE — obstacle returns are applied afterwards and override it.
    """
    ground = state == FREE
    if not ground.any():
        return state

    nx, ny = state.shape
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    # Cell centres relative to the sensor, which sits at sensor_xy in the same
    # frame the returns were expressed in.
    dx = origin_x + (ix + 0.5) * resolution - float(sensor_xy[0])
    dy = origin_y + (iy + 0.5) * resolution - float(sensor_xy[1])
    radius = np.hypot(dx, dy)
    azimuth = np.arctan2(dy, dx)

    bins = max(int(bins), 1)
    bin_index = np.minimum(
        ((azimuth + np.pi) / (2 * np.pi) * bins).astype(np.int64), bins - 1
    )

    # Farthest ground return per azimuth, and how many cells voted for it.
    reach = np.zeros(bins, dtype=np.float64)
    votes = np.zeros(bins, dtype=np.int64)
    np.maximum.at(reach, bin_index[ground], radius[ground])
    np.add.at(votes, bin_index[ground], 1)

    # A single stray return must not carve a whole spoke across the map.
    reach[votes < max(int(min_cells), 1)] = 0.0

    carved = (state == UNKNOWN) & (radius <= reach[bin_index])
    state[carved] = FREE
    return state


def build_costmap(
    ground_xy,
    obstacle_xy,
    *,
    resolution=0.5,
    bounds=None,
    robot_radius=0.6,
    inflation_radius=1.5,
    cost_scaling=3.0,
    min_ground_points=1,
    min_obstacle_points=3,
    plane=None,
    ray_fill=True,
    ray_fill_bins=720,
    ray_fill_min_cells=2,
    sensor_xy=(0.0, 0.0),
):
    """
    Rasterise ground and non-ground returns into a tri-state cost grid.

    ``ground_xy`` and ``obstacle_xy`` are ``(N, 2)`` or ``(N, 3)`` arrays and
    either may be empty; with both empty the result is a 1x1 UNKNOWN grid
    rather than an exception, so a callback on a blank frame still has
    something to publish a clear-marker against.

    ``resolution`` is the requested cell size; read the honoured one back off
    ``grid.resolution``, which coarsens when the requested spacing would blow
    past ``MAX_CELLS``. Every cell left FREE is guaranteed at least
    ``robot_radius`` of clearance from the nearest obstacle *return*, not
    merely from the nearest obstacle cell centre.
    """
    ground = _xy(ground_xy)
    obstacles = _xy(obstacle_xy)

    if bounds is None:
        extent = _extent(ground, obstacles)
    else:
        extent = tuple(float(v) for v in bounds)

    if extent is None:
        return CostGrid(
            origin_x=0.0,
            origin_y=0.0,
            resolution=float(resolution),
            state=np.full((1, 1), UNKNOWN, dtype=np.int8),
            cost=np.full((1, 1), np.inf, dtype=np.float32),
            plane=plane,
        )

    resolution = float(resolution)
    origin_x, origin_y, nx, ny = _dims(extent, resolution)
    if nx * ny > MAX_CELLS:
        # A single stray far-field return must not size the whole map; fall
        # back to the extent of the surface we actually intend to drive on.
        ground_extent = _extent(ground)
        fallback = extent if ground_extent is None else ground_extent
        origin_x, origin_y, nx, ny = _dims(fallback, resolution)
        if nx * ny > MAX_CELLS:
            # Coarsen, never crop. Rescaling nx/ny at a fixed resolution used
            # to shrink the map to a corner of the observed ground and drop
            # every return outside it, so the planner reported "no route" for
            # anything beyond that corner with nothing logged. A bigger cell
            # keeps the whole surface on the map; the caller reads the actual
            # spacing back off ``grid.resolution``.
            # _dims rounds up, so one rescale can land marginally over the
            # cap; iterate until it is genuinely under it.
            while nx * ny > MAX_CELLS:
                resolution *= float(np.sqrt(nx * ny / float(MAX_CELLS))) + 1e-3
                origin_x, origin_y, nx, ny = _dims(fallback, resolution)

    state = np.full((nx, ny), UNKNOWN, dtype=np.int8)
    ground_counts = _counts(
        _quantise(ground, origin_x, origin_y, resolution, nx, ny), nx, ny
    )
    obstacle_counts = _counts(
        _quantise(obstacles, origin_x, origin_y, resolution, nx, ny), nx, ny
    )

    state[ground_counts >= max(1, int(min_ground_points))] = FREE

    if ray_fill:
        # Close the gaps between beam rings before obstacles are stamped, so a
        # carved cell can still be overridden by a return that lands in it.
        _carve_ray_free(
            state,
            origin_x,
            origin_y,
            resolution,
            sensor_xy,
            ray_fill_bins,
            ray_fill_min_cells,
        )
    # Obstacles overwrite ground: a pole planted on inlier road still blocks.
    occupied = obstacle_counts >= max(1, int(min_obstacle_points))
    state[occupied] = OCCUPIED

    if occupied.any():
        distance = distance_transform_edt(~occupied, sampling=resolution)
        # The transform measures centre to centre, but the return that made a
        # cell OCCUPIED sits anywhere inside it, up to half a cell diagonal
        # from that centre. Taking the raw distance as clearance overstates it
        # by that much, which at the default 0.5 m resolution let a cell 0.36 m
        # from a real obstacle pass as clear of a 0.6 m robot radius. Subtract
        # the worst case so the clearance the map reports is one the geometry
        # actually guarantees; the price is up to half a cell of over-inflation.
        clearance = np.maximum(distance - 0.5 * _SQRT2 * resolution, 0.0)
    else:
        # With nothing occupied every cell is infinitely clear; the transform
        # of an all-true mask is meaningless, so short-circuit it.
        clearance = np.full((nx, ny), np.inf, dtype=np.float64)

    inflated = (state == FREE) & (clearance <= float(robot_radius))
    state[inflated] = INFLATED

    cost = np.full((nx, ny), np.inf, dtype=np.float32)
    free = state == FREE
    # Decays from just under 2.0 at the clearance boundary towards 1.0, so the
    # planner prefers the middle of the corridor without ever going below the
    # admissible floor of 1.0.
    near = free & (clearance < float(inflation_radius))
    cost[free] = 1.0
    cost[near] = 1.0 + np.exp(
        -float(cost_scaling) * (clearance[near] - float(robot_radius))
    )

    return CostGrid(
        origin_x=float(origin_x),
        origin_y=float(origin_y),
        resolution=float(resolution),
        state=state,
        cost=cost,
        plane=plane,
    )


def snap_to_traversable(grid, x, y, radius):
    """
    Return the traversable cell nearest to a world point, or ``None``.

    Only cells whose centre lies within ``radius`` metres are considered.
    Ties resolve to the smaller ``(ix, iy)`` so repeated calls on the same
    frame give the planner an identical start cell.
    """
    nx, ny = grid.shape
    if nx == 0 or ny == 0 or radius < 0.0:
        return None

    cx, cy = grid.world_to_cell(x, y)
    span = int(np.ceil(radius / grid.resolution)) + 1
    lo_x, hi_x = max(0, cx - span), min(nx, cx + span + 1)
    lo_y, hi_y = max(0, cy - span), min(ny, cy + span + 1)
    if lo_x >= hi_x or lo_y >= hi_y:
        return None

    window = grid.traversable_mask()[lo_x:hi_x, lo_y:hi_y]
    local = np.argwhere(window)
    if local.shape[0] == 0:
        return None

    cells = local + np.array([lo_x, lo_y], dtype=local.dtype)
    wx, wy = grid.cell_to_world(cells[:, 0], cells[:, 1])
    dist = np.hypot(wx - float(x), wy - float(y))
    within = dist <= float(radius)
    if not within.any():
        return None

    cells, dist = cells[within], dist[within]
    order = np.lexsort((cells[:, 1], cells[:, 0], dist))
    best = cells[order[0]]
    return int(best[0]), int(best[1])


def neighbors(grid, cell, connectivity=8):
    """
    Yield ``(neighbour_cell, step_length_in_metres)`` for traversable neighbours.

    Diagonal steps cost ``resolution * sqrt(2)``, which is what keeps a
    Euclidean heuristic admissible on an 8-connected grid.

    A diagonal step is only offered when both cells it squeezes past are
    traversable too. Without that check an 8-connected search walks through the
    shared corner of two diagonally touching blocked cells: on an unbroken
    diagonal wall it produced a straight path *through* the wall, and under the
    shipped inflation defaults the same hole stayed open through the UNKNOWN
    cells of a LiDAR shadow. The vehicle has width; a corner it cannot fit
    through is not an edge of the graph.
    """
    if connectivity == 4:
        steps = _STEPS_4
    elif connectivity == 8:
        steps = _STEPS_8
    else:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity!r}")

    ix, iy = int(cell[0]), int(cell[1])
    for dx, dy in steps:
        nb = (ix + dx, iy + dy)
        if not grid.is_traversable(nb[0], nb[1]):
            continue
        if dx and dy:
            if not (
                grid.is_traversable(ix + dx, iy) and grid.is_traversable(ix, iy + dy)
            ):
                continue
            yield nb, float(grid.resolution * _SQRT2)
        else:
            yield nb, float(grid.resolution)
