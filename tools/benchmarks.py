#!/usr/bin/env python3
"""
Reproduce every performance and accuracy figure quoted in the README.

The README used to say "the exact commands are in CHANGELOG.md" while the
CHANGELOG carried two commands covering two of the ten figure groups. A reader
who wanted to check a number had nothing to run. This script is that command:
each section prints the numbers the docs quote, on scenes defined here in full,
so a figure that drifts shows up as a diff rather than as prose nobody can
falsify.

Every scene is synthetic. Nothing here has ever seen a real sensor, and the
README says so in the same breath as the numbers.

    python3 tools/benchmarks.py            # every section
    python3 tools/benchmarks.py --section segmentation

The node-callback section needs a sourced ROS 2 install; the rest is
numpy/scipy only and runs on a bare checkout.
"""

import argparse
import math
import os
import sys
import time

import numpy as np

# The README documents this as `python3 tools/benchmarks.py`, which puts tools/
# on sys.path rather than the repo root, so the package would not import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ransac_seg.cluster import cluster_obstacles  # noqa: E402
from ransac_seg.grid import FREE, INFLATED, OCCUPIED, UNKNOWN, build_costmap  # noqa: E402
from ransac_seg.ground_plane import segment_ground  # noqa: E402
from ransac_seg.lane import fit_lane  # noqa: E402
from ransac_seg.planners import astar, bfs, dfs, dijkstra, path_cost  # noqa: E402

SLOPE_DEG = 10.0
SENSOR_HEIGHT = 1.7
REPEATS = 7


def median_ms(fn, repeats=REPEATS):
    """Return the median wall time of ``repeats`` warm calls, in milliseconds."""
    fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1e3)
    return float(np.median(times))


def slope_cloud(n_inliers=78000, n_outliers=42000, noise=0.01, extent=25.0, seed=7):
    """
    Build the 120k-point sloped-road cloud, and its ground-truth mask.

    Identical construction to ``test_ground_plane.make_plane_cloud``: inliers
    sampled on the plane itself through two in-plane basis vectors, outliers
    lifted 0.6-4.0 m along the normal.
    """
    rng = np.random.default_rng(seed)
    angle = math.radians(SLOPE_DEG)
    normal = np.array([math.sin(angle), 0.0, math.cos(angle)])
    normal = normal / np.linalg.norm(normal)

    seed_vec = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, seed_vec)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    origin = -SENSOR_HEIGHT * normal
    coeffs = rng.uniform(-extent, extent, size=(n_inliers, 2))
    inliers = origin + coeffs[:, :1] * u + coeffs[:, 1:] * v
    inliers += rng.normal(0.0, noise, size=inliers.shape)

    outliers = origin + rng.uniform(-extent, extent, size=(n_outliers, 2)) @ np.vstack(
        [u, v]
    )
    outliers += normal * rng.uniform(0.6, 4.0, size=(n_outliers, 1))

    cloud = np.vstack([inliers, outliers])
    truth = np.zeros(cloud.shape[0], dtype=bool)
    truth[:n_inliers] = True
    shuffle = rng.permutation(cloud.shape[0])
    return cloud[shuffle], truth[shuffle]


def parked_cars_scene(seed=11):
    """Build a 40 m road corridor with three parked cars along its right edge."""
    rng = np.random.default_rng(seed)
    n = 14000
    xs = rng.uniform(0.6, 40.0, n)
    ys = rng.uniform(-4.0, 4.0, n)
    zs = -SENSOR_HEIGHT + xs * math.tan(math.radians(SLOPE_DEG))
    ground = np.column_stack([xs, ys, zs + rng.normal(0.0, 0.01, n)])

    cars = []
    for centre_x in (12.0, 22.0, 32.0):
        count = 900
        cx = rng.uniform(centre_x - 2.1, centre_x + 2.1, count)
        cy = rng.uniform(2.6, 4.4, count)
        base = -SENSOR_HEIGHT + cx * math.tan(math.radians(SLOPE_DEG))
        cars.append(np.column_stack([cx, cy, base + rng.uniform(0.2, 1.6, count)]))
    return ground, np.vstack(cars)


def sweep_scene(n=120000, obstacle_fraction=0.2, seed=13):
    """
    Build one full-size forward sweep: a road corridor flanked by roadside clutter.

    Sized like a real 10 Hz spinning LiDAR rather than like the 14k-point demo
    scene, because a per-callback budget measured on a scene an order of
    magnitude smaller says nothing about whether the sweep fits in 100 ms.
    """
    rng = np.random.default_rng(seed)
    n_obstacle = int(n * obstacle_fraction)
    n_ground = n - n_obstacle

    xs = rng.uniform(0.6, 45.0, n_ground)
    ys = rng.uniform(-5.0, 5.0, n_ground)
    zs = -SENSOR_HEIGHT + xs * math.tan(math.radians(SLOPE_DEG))
    ground = np.column_stack([xs, ys, zs + rng.normal(0.0, 0.01, n_ground)])

    ox = rng.uniform(0.6, 45.0, n_obstacle)
    oy = np.where(rng.random(n_obstacle) < 0.5, -1.0, 1.0) * rng.uniform(
        5.2, 6.5, n_obstacle
    )
    obase = -SENSOR_HEIGHT + ox * math.tan(math.radians(SLOPE_DEG))
    obstacles = np.column_stack([ox, oy, obase + rng.uniform(0.4, 2.5, n_obstacle)])
    return np.vstack([ground, obstacles])


def straight_road(width=7.00, n=60000, length=40.0, seed=1):
    """Return ground xy for a straight road of constant width."""
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [rng.uniform(0.0, length, n), rng.uniform(-width / 2.0, width / 2.0, n)]
    )


def curved_road(width=6.00, n=60000, length=40.0, seed=2):
    """Return ground xy straddling the centreline ``y = 0.01 x^2`` (R ~ 60 m)."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, length, n)
    lateral = rng.uniform(-width / 2.0, width / 2.0, n)
    slope = 0.02 * t
    norm = np.hypot(slope, 1.0)
    return np.column_stack([t - lateral * slope / norm, 0.01 * t**2 + lateral / norm])


def bench_decode():
    """Vectorised PointCloud2 decode against the per-point read_points loop."""
    try:
        from ransac_seg.cloud_io import pointcloud2_to_xyz
        from sensor_msgs_py import point_cloud2
        from std_msgs.msg import Header
    except ImportError:
        print("  skipped: needs a sourced ROS 2 install")
        return

    cloud, _ = slope_cloud()
    header = Header()
    header.frame_id = "lidar_front"
    msg = point_cloud2.create_cloud_xyz32(header, cloud.astype(np.float32).tolist())

    def loop():
        return np.array(
            [
                (p[0], p[1], p[2])
                for p in point_cloud2.read_points(
                    msg, field_names=("x", "y", "z"), skip_nans=True
                )
            ],
            dtype=np.float64,
        )

    fast = pointcloud2_to_xyz(msg)
    slow = loop()
    print(f"  points                {fast.shape[0]}")
    print(
        f"  pointcloud2_to_xyz    {median_ms(lambda: pointcloud2_to_xyz(msg)):.2f} ms"
    )
    print(f"  read_points loop      {median_ms(loop, repeats=3):.1f} ms")
    print(f"  max abs difference    {np.max(np.abs(fast - slow)):.1e}")


def bench_segmentation():
    """RANSAC ground segmentation against the fixed z < 0.2 cut it replaced."""
    cloud, truth = slope_cloud()

    def score(mask):
        hit = int(np.count_nonzero(mask & truth))
        return hit / int(truth.sum()), hit / max(1, int(np.count_nonzero(mask)))

    mask, model = segment_ground(cloud)
    recall, precision = score(mask)
    legacy_recall, legacy_precision = score(cloud[:, 2] < 0.2)

    print(
        f"  scene                 {cloud.shape[0]} points, {SLOPE_DEG:.0f} deg slope,"
    )
    print(f"                        {1 - truth.mean():.0%} non-ground returns")
    print(f"  RANSAC                recall {recall:.3f} / precision {precision:.3f}")
    print(f"  RANSAC time           {median_ms(lambda: segment_ground(cloud)):.1f} ms")
    print(f"  fitted tilt           {model.tilt_deg:.2f} deg")
    print(
        f"  legacy z < 0.2        recall {legacy_recall:.3f} / "
        f"precision {legacy_precision:.3f}"
    )


def bench_clustering_and_costmap():
    """Obstacle clustering and cost-grid rasterisation on the parked-car scene."""
    ground, obstacles = parked_cars_scene()
    clusters = cluster_obstacles(obstacles)
    extents = np.array([c.extent for c in clusters])

    print(f"  non-ground returns    {obstacles.shape[0]}")
    print(f"  clusters              {len(clusters)} (planted 3)")
    print(
        f"  median extent         {np.median(extents[:, 0]):.2f} x "
        f"{np.median(extents[:, 1]):.2f} m (planted 4.20 x 1.80 m)"
    )
    print(
        f"  clustering time       "
        f"{median_ms(lambda: cluster_obstacles(obstacles)):.1f} ms"
    )

    grid = build_costmap(ground, obstacles)
    counts = {
        name: int(np.count_nonzero(grid.state == value))
        for name, value in (
            ("free", FREE),
            ("occupied", OCCUPIED),
            ("inflated", INFLATED),
            ("unknown", UNKNOWN),
        )
    }
    finite = grid.cost[np.isfinite(grid.cost)]
    print(
        f"  grid                  {grid.shape[0]} x {grid.shape[1]} cells at "
        f"{grid.resolution:.2f} m"
    )
    print(
        "  cells                 " + " / ".join(f"{v} {k}" for k, v in counts.items())
    )
    print(f"  free-cell cost        {finite.min():.3f} - {finite.max():.3f}")
    print(
        f"  build_costmap time    "
        f"{median_ms(lambda: build_costmap(ground, obstacles)):.1f} ms"
    )


def bench_lane():
    """Lane width recovery on a straight and a curving road."""
    for label, xy, nominal in (
        ("straight 7.00 m", straight_road(), 7.00),
        ("curving 6.00 m ", curved_road(), 6.00),
    ):
        fit = fit_lane(xy)
        # Percentile edges are inset by construction: on a road of uniform
        # return density the 2nd percentile sits 2% of the width inside the
        # true edge, so this, not the nominal width, is the estimand.
        estimand = nominal * (1.0 - 2.0 * 0.02)
        width = float(np.median(fit.width))
        print(
            f"  {label}       median width {width:.3f} m, estimand "
            f"{estimand:.3f} m, error {abs(width - estimand):.3f} m"
        )


def bench_planners():
    """A*/Dijkstra agreement, DFS/BFS reachability and search timings."""
    ground, obstacles = parked_cars_scene()
    grid = build_costmap(ground, obstacles)
    free = grid.cells_with_state(FREE)
    start = tuple(int(v) for v in free[np.argmin(free[:, 0])])
    goal = tuple(int(v) for v in free[np.argmax(free[:, 0])])

    path, cost = astar(grid, start, goal)
    field, _ = dijkstra(grid, start)
    hops = bfs(grid, start)
    order = dfs(grid, start)

    print(f"  start / goal          {start} -> {goal}")
    print(f"  A* path               {len(path)} cells, cost {cost:.6f}")
    print(f"  Dijkstra at goal      {float(field[goal]):.6f}")
    print(f"  path_cost(path)       {path_cost(grid, path):.6f}")
    print(
        f"  BFS / DFS cells       {int(np.count_nonzero(hops >= 0))} / "
        f"{int(np.count_nonzero(order >= 0))}"
    )
    print(
        f"  A* time               {median_ms(lambda: astar(grid, start, goal)):.1f} ms"
    )
    print(f"  Dijkstra time         {median_ms(lambda: dijkstra(grid, start)):.1f} ms")
    print(f"  BFS time              {median_ms(lambda: bfs(grid, start)):.1f} ms")
    print(f"  DFS time              {median_ms(lambda: dfs(grid, start)):.1f} ms")


def bench_callbacks():
    """Per-sweep cost of every node's subscription callback, at sweep size."""
    try:
        import rclpy
        from ransac_seg.astar import AStarPlanner
        from ransac_seg.bfs import BFSPlanner
        from ransac_seg.curve import LaneBoundaryNode
        from ransac_seg.dfs import DFSBaseline
        from ransac_seg.dijkstra import DijkstraPlanner
        from ransac_seg.ransac_node import GroundSegmentationNode
        from sensor_msgs_py import point_cloud2
        from std_msgs.msg import Header
    except ImportError:
        print("  skipped: needs a sourced ROS 2 install")
        return

    cloud = sweep_scene()
    header = Header()
    header.frame_id = "lidar_front"
    msg = point_cloud2.create_cloud_xyz32(header, cloud.astype(np.float32).tolist())

    rclpy.init()
    try:
        for node_cls in (
            GroundSegmentationNode,
            LaneBoundaryNode,
            AStarPlanner,
            BFSPlanner,
            DFSBaseline,
            DijkstraPlanner,
        ):
            node = node_cls()
            for attr in dir(node):
                pub = getattr(node, attr, None)
                if hasattr(pub, "publish") and hasattr(pub, "topic_name"):
                    pub.publish = lambda _msg: None
            try:
                elapsed = median_ms(lambda: node._on_cloud(msg), repeats=5)
            finally:
                node.destroy_node()
            print(f"  {node_cls.__name__:<24} {elapsed:6.1f} ms")
    finally:
        rclpy.shutdown()
    print(f"  cloud                 {cloud.shape[0]} points; a 10 Hz sweep has 100 ms")


SECTIONS = {
    "decode": bench_decode,
    "segmentation": bench_segmentation,
    "clustering": bench_clustering_and_costmap,
    "lane": bench_lane,
    "planners": bench_planners,
    "callbacks": bench_callbacks,
}


def main(argv=None):
    """Run the requested benchmark sections and print their figures."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--section",
        action="append",
        choices=sorted(SECTIONS),
        help="Run only this section; repeatable. Default: all of them.",
    )
    args = parser.parse_args(argv)

    for name in args.section or list(SECTIONS):
        print(f"\n[{name}]")
        SECTIONS[name]()
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
