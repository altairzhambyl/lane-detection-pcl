#!/usr/bin/env python3
"""
Graph searches over the tri-state cost grid, as pure functions.

The four planner nodes each carried their own private copy of a search that
ran over a bare Python ``set`` of cells that happened to hold a ground return.
That representation cannot express "occupied" or "unobserved" — both are simply
absent — so an obstacle was only visible through the hole its LiDAR shadow left
in the ground returns, and every step cost exactly one, which meant no route
ever preferred the middle of a corridor over scraping past a wall.

Everything here reads a ``CostGrid`` instead and shares one cost model:

    edge(a, b) = step_length_metres(a, b) * 0.5 * (cost[a] + cost[b])

with ``step_length`` exactly as ``grid.neighbors`` reports it. Both the nodes
and the documentation quote that formula, so it lives in one place.

The two field-quantising helpers at the bottom live here rather than in
``markers`` for the same reason: they are pure array maths that three nodes
need in order to colour a search result, and they must stay testable without
ROS.

No ROS import lives in this module, so the planner tests run on a bare
checkout without a sourced ROS installation.
"""

import heapq
import math
from collections import deque

import numpy as np

from ransac_seg.grid import neighbors, snap_to_traversable

_SQRT2 = math.sqrt(2.0)


def octile(a, b, resolution):
    """Return the octile distance in metres between two cells."""
    dx = abs(int(a[0]) - int(b[0]))
    dy = abs(int(a[1]) - int(b[1]))
    return float(resolution) * ((dx + dy) + (_SQRT2 - 2.0) * min(dx, dy))


def manhattan(a, b, resolution):
    """Return the 4-connected grid distance in metres between two cells."""
    dx = abs(int(a[0]) - int(b[0]))
    dy = abs(int(a[1]) - int(b[1]))
    return float(resolution) * (dx + dy)


def _heuristic(connectivity):
    """Return the tightest admissible heuristic for a connectivity."""
    if connectivity == 8:
        return octile
    if connectivity == 4:
        return manhattan
    raise ValueError(f"connectivity must be 4 or 8, got {connectivity!r}")


def validate_connectivity(connectivity):
    """Raise ValueError unless the connectivity is one grid.neighbors accepts."""
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity!r}")


def _step_length(grid, a, b):
    """Return the metric length of one step between two adjacent cells."""
    dx = abs(int(a[0]) - int(b[0]))
    dy = abs(int(a[1]) - int(b[1]))
    if dx > 1 or dy > 1 or (dx == 0 and dy == 0):
        raise ValueError(f"cells {a!r} and {b!r} are not adjacent")
    return float(grid.resolution) * (_SQRT2 if dx and dy else 1.0)


def _edge_cost(grid, a, b, step):
    """Return the cost of traversing one edge under the shared cost model."""
    return step * 0.5 * (float(grid.cost[a]) + float(grid.cost[b]))


def _trace(parents, start, goal):
    """Walk a ``{cell: predecessor}`` map backwards from goal to start."""
    path = [goal]
    node = goal
    while node != start:
        node = parents[node]
        path.append(node)
    path.reverse()
    return path


def astar(grid, start, goal, *, connectivity=8):
    """
    Return (path, cost) minimising accumulated cell cost, or ([], inf).

    Admissibility: ``build_costmap`` guarantees every finite cell cost is
    >= 1.0, so the cheapest possible edge is its own step length in metres.
    The cheapest possible route to the goal therefore costs at least the
    minimum walkable distance to it, which on an 8-connected grid is exactly
    ``octile(cell, goal, resolution)`` and on a 4-connected grid exactly
    ``manhattan(...)``. Multiplying by the cost floor of 1.0 leaves both
    heuristics admissible and consistent, so the first pop of a cell is final
    and the CLOSED set below can discard every later pop of it.

    Using a Euclidean heuristic on a 4-connected grid — what this package did
    before — is still admissible but looser by up to sqrt(2), which is what
    made A* degenerate towards Dijkstra on open ground.
    """
    heuristic = _heuristic(connectivity)
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    if not grid.is_traversable(*start) or not grid.is_traversable(*goal):
        return [], math.inf

    resolution = float(grid.resolution)
    best = {start: 0.0}
    parents = {}
    closed = set()
    queue = [(heuristic(start, goal, resolution), start)]

    while queue:
        _, current = heapq.heappop(queue)
        if current in closed:
            continue
        closed.add(current)
        if current == goal:
            return _trace(parents, start, goal), best[goal]

        current_g = best[current]
        for nb, step in neighbors(grid, current, connectivity):
            if nb in closed:
                continue
            tentative = current_g + _edge_cost(grid, current, nb, step)
            if tentative < best.get(nb, math.inf):
                best[nb] = tentative
                parents[nb] = current
                heapq.heappush(queue, (tentative + heuristic(nb, goal, resolution), nb))

    return [], math.inf


def dijkstra(grid, start, *, connectivity=8):
    """
    Return (cost, parent) fields over the whole grid from one source.

    ``cost`` is ``(nx, ny)`` float32 holding inf for every cell the source
    cannot reach; ``parent`` is ``(nx, ny, 2)`` int32 holding -1 there. The
    source is its own parent, which is what terminates ``reconstruct_path``
    and what distinguishes "this is the source" from "this is unreachable".

    The queue uses lazy deletion — stale entries are skipped on pop rather
    than decreased in place, because heapq exposes no decrease-key.
    """
    validate_connectivity(connectivity)
    nx, ny = grid.shape
    parent = np.full((nx, ny, 2), -1, dtype=np.int32)
    # Accumulate in float64 and narrow once at the end: rounding every partial
    # sum to float32 would make the field disagree with A*'s float64 total on
    # the same optimal path.
    dist = np.full((nx, ny), np.inf, dtype=np.float64)

    start = (int(start[0]), int(start[1]))
    if not grid.is_traversable(*start):
        return dist.astype(np.float32), parent

    dist[start] = 0.0
    parent[start] = start
    queue = [(0.0, start)]

    while queue:
        cost_so_far, current = heapq.heappop(queue)
        if cost_so_far > dist[current]:
            continue
        for nb, step in neighbors(grid, current, connectivity):
            candidate = cost_so_far + _edge_cost(grid, current, nb, step)
            if candidate < dist[nb]:
                dist[nb] = candidate
                parent[nb] = current
                heapq.heappush(queue, (candidate, nb))

    return dist.astype(np.float32), parent


def bfs(grid, start, *, connectivity=4):
    """
    Return the hop count to every reachable cell; -1 where unreachable.

    Hops, not metres and not cost: this is the unweighted reachability
    baseline the weighted Dijkstra field is compared against.
    """
    validate_connectivity(connectivity)
    nx, ny = grid.shape
    hops = np.full((nx, ny), -1, dtype=np.int32)

    start = (int(start[0]), int(start[1]))
    if not grid.is_traversable(*start):
        return hops

    hops[start] = 0
    queue = deque([start])
    while queue:
        current = queue.popleft()
        depth = int(hops[current]) + 1
        for nb, _ in neighbors(grid, current, connectivity):
            if hops[nb] < 0:
                hops[nb] = depth
                queue.append(nb)
    return hops


def dfs(grid, start, *, connectivity=4):
    """
    Return depth-first discovery order per cell; -1 where unvisited.

    The stack is explicit. A 200x200 costmap is 40,000 cells and a recursive
    depth-first search would exhaust CPython's recursion limit long before
    covering it, which is why this never recurses.
    """
    validate_connectivity(connectivity)
    nx, ny = grid.shape
    order = np.full((nx, ny), -1, dtype=np.int32)

    start = (int(start[0]), int(start[1]))
    if not grid.is_traversable(*start):
        return order

    stack = [start]
    counter = 0
    while stack:
        current = stack.pop()
        if order[current] >= 0:
            continue
        order[current] = counter
        counter += 1
        for nb, _ in neighbors(grid, current, connectivity):
            if order[nb] < 0:
                stack.append(nb)
    return order


def reconstruct_path(parent, goal):
    """
    Return the cell path from a Dijkstra parent field, or [] if unreachable.

    The walk terminates on the source, which is stored as its own parent.
    """
    goal = (int(goal[0]), int(goal[1]))
    nx, ny, _ = parent.shape
    if not (0 <= goal[0] < nx and 0 <= goal[1] < ny):
        return []
    if parent[goal[0], goal[1], 0] < 0:
        return []

    path = [goal]
    node = goal
    while True:
        previous = (int(parent[node[0], node[1], 0]), int(parent[node[0], node[1], 1]))
        if previous == node:
            break
        path.append(previous)
        node = previous
    path.reverse()
    return path


def path_cost(grid, path):
    """Return the summed edge cost of a cell path; 0.0 for a path of one."""
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        total += _edge_cost(grid, a, b, _step_length(grid, a, b))
    return total


def path_length_m(grid, path):
    """Return the geometric length of a cell path in metres."""
    return float(sum(_step_length(grid, a, b) for a, b in zip(path[:-1], path[1:])))


def _grid_bounds(grid):
    """Return the ``(lo, hi)`` world-space corners of the grid as arrays."""
    nx, ny = grid.shape
    lo = np.array([grid.origin_x, grid.origin_y], dtype=np.float64)
    return lo, lo + np.array([nx, ny], dtype=np.float64) * grid.resolution


def _pull_onto_grid(grid, point, axis):
    """
    Return the first point on the map when walking back along ``-axis``.

    Returns ``None`` when the request already lies on the map — a snap failure
    there is a real "nothing traversable here", not an off-map goal — and when
    the backward ray never enters the map at all.
    """
    lo, hi = _grid_bounds(grid)
    origin = np.asarray(point, dtype=np.float64)
    if np.all(origin >= lo) and np.all(origin <= hi):
        return None

    direction = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        return None
    direction = -direction / norm

    with np.errstate(divide="ignore", invalid="ignore"):
        near = (lo - origin) / direction
        far = (hi - origin) / direction
    entry = float(np.nanmax(np.minimum(near, far)))
    exit_ = float(np.nanmin(np.maximum(near, far)))
    entry = max(entry, 0.0)
    if not exit_ >= entry:
        return None

    # Step half a cell past the boundary so the landing point is inside a cell
    # rather than exactly on its edge, where floor() is ambiguous.
    travel = min(entry + 0.5 * grid.resolution, 0.5 * (entry + exit_))
    landing = origin + travel * direction
    return float(landing[0]), float(landing[1])


def select_goal(grid, preferred_xy, *, search_radius=5.0, fallback_axis=(1.0, 0.0)):
    """
    Snap a requested goal onto a reachable cell, or fall back along an axis.

    Geometry only: the returned cell is traversable but may sit in a different
    connected component than the planner's start, which is the caller's
    problem to notice and resolve with ``furthest_reachable``.

    A goal parameter tuned for a long bag routinely lands past the edge of a
    single frame's map, so when the request is off-map it is pulled back along
    ``fallback_axis`` to the map boundary before snapping. An on-map request
    that has nothing traversable within ``search_radius`` returns ``None``.
    """
    x, y = float(preferred_xy[0]), float(preferred_xy[1])
    cell = snap_to_traversable(grid, x, y, search_radius)
    if cell is not None:
        return cell

    landing = _pull_onto_grid(grid, (x, y), fallback_axis)
    if landing is None:
        return None
    return snap_to_traversable(grid, landing[0], landing[1], search_radius)


def furthest_reachable(cost, grid, axis):
    """
    Return the reachable cell furthest along ``axis``, or ``None``.

    Reading the reachable set off a Dijkstra cost field rather than off the
    raw traversability mask is what makes this fallback goal safe: every cell
    with a finite cost is provably connected to the source, so the planner
    cannot pick a goal on the far side of a wall and then fail every frame.
    Ties resolve to the smaller ``(ix, iy)`` so a static scene keeps producing
    the same goal frame after frame instead of flickering between two cells.
    """
    direction = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        raise ValueError("fallback axis must have non-zero length")
    direction = direction / norm

    cells = np.argwhere(np.isfinite(cost)).astype(np.int64)
    if cells.shape[0] == 0:
        return None

    xs, ys = grid.cell_to_world(cells[:, 0], cells[:, 1])
    projection = xs * direction[0] + ys * direction[1]
    order = np.lexsort((cells[:, 1], cells[:, 0], -projection))
    best = cells[order[0]]
    return int(best[0]), int(best[1])


def normalised_bands(values, valid, bands):
    """
    Quantise a scalar field into ``bands`` equal-width levels over ``valid``.

    Returns an int32 array holding a band index per valid cell and -1
    elsewhere. Publishing one uniformly coloured CUBE_LIST per band keeps the
    marker path vectorised; a per-cell colour would mean building one
    ColorRGBA object per cell, and a 200x200 costmap is 40,000 of them every
    frame.
    """
    bands = max(1, int(bands))
    field = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(field)
    index = np.full(field.shape, -1, dtype=np.int32)
    if not mask.any():
        return index

    low = float(field[mask].min())
    high = float(field[mask].max())
    if high - low < 1e-12:
        index[mask] = 0
        return index

    scaled = (field[mask] - low) / (high - low)
    index[mask] = np.clip((scaled * bands).astype(np.int32), 0, bands - 1)
    return index


def band_color(band, bands, alpha=0.7):
    """Return an ``(r, g, b, a)`` blue-to-red ramp colour for a band index."""
    bands = max(1, int(bands))
    t = float(band) / max(1.0, float(bands - 1))
    t = min(max(t, 0.0), 1.0)
    return (t, 0.35 * (1.0 - abs(2.0 * t - 1.0)) + 0.2, 1.0 - t, float(alpha))
