#!/usr/bin/env python3
"""
Measure per-stage pipeline latency on a recorded rosbag — real sensor data.

`tools/benchmarks.py` builds its own clouds, so it measures the implementation
against a scene chosen to suit it. `tools/eval_semantickitti.py` measures
accuracy against labelled data. Neither measures what the pipeline costs on the
sensor this package was actually written for, which is what this does: decode,
segment, cluster, rasterise, fit and plan, timed separately, over frames pulled
straight out of a recording.

There are no accuracy figures here. The bag carries no ground-truth labels, so
anything this printed about correctness would be invented — timing and shape
statistics are all a bag without labels can honestly support.

Usage:
    python3 tools/bench_rosbag.py BAG_URI [--frames 300] [--topic ...]
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rosbag2_py  # noqa: E402
from ransac_seg.cloud_io import pointcloud2_to_xyz  # noqa: E402
from ransac_seg.cluster import cluster_obstacles  # noqa: E402
from ransac_seg.grid import build_costmap, snap_to_traversable  # noqa: E402
from ransac_seg.ground_plane import segment_ground  # noqa: E402
from ransac_seg.lane import fit_lane  # noqa: E402
from ransac_seg.plane_filter import PlaneFilter  # noqa: E402
from ransac_seg.planners import (  # noqa: E402
    astar,
    dijkstra,
    furthest_reachable,
    select_goal,
)
from rclpy.serialization import deserialize_message  # noqa: E402
from sensor_msgs.msg import PointCloud2  # noqa: E402


class Timer:
    """Records elapsed milliseconds per named stage."""

    def __init__(self):
        """Start with no samples."""
        self.samples = defaultdict(list)

    def time(self, name, fn, *args, **kwargs):
        """Run ``fn``, record how long it took under ``name``, return its result."""
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        self.samples[name].append((time.perf_counter() - start) * 1000.0)
        return result

    def report(self):
        """Return per-stage p50/p95/p99/max in milliseconds."""
        out = {}
        for name, values in self.samples.items():
            arr = np.asarray(values)
            out[name] = {
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "max": float(arr.max()),
                "n": arr.size,
            }
        return out


def plan_like_the_node(grid, start, goal_xy=(25.0, 0.0), axis=(1.0, 0.0)):
    """
    Reproduce AStarPlanner._plan: the requested goal, else the furthest reachable.

    Calling astar() alone under-reports what the pipeline does. On a real drive
    the requested goal is frequently off the track or behind a barrier, and the
    node answers that with a Dijkstra field and a route to the furthest cell
    provably connected to the start rather than by publishing nothing.
    """
    goal = select_goal(grid, goal_xy, search_radius=5.0, fallback_axis=axis)
    if goal is not None:
        path, _ = astar(grid, start, goal)
        if path:
            return path, "goal"
    cost, _ = dijkstra(grid, start)
    fallback = furthest_reachable(cost, grid, axis)
    if fallback is None or fallback == start:
        return [], "none"
    path, _ = astar(grid, start, fallback)
    return path, "fallback" if path else "none"


def read_frames(bag_uri, topic, limit, storage_id="mcap"):
    """Yield deserialised PointCloud2 messages from ``bag_uri``."""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_uri, storage_id=storage_id),
        rosbag2_py.ConverterOptions("", ""),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    count = 0
    while reader.has_next() and count < limit:
        _, data, _ = reader.read_next()
        yield deserialize_message(data, PointCloud2)
        count += 1


def main():
    """Time every pipeline stage over frames from a recorded bag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag_uri")
    parser.add_argument("--topic", default="/sensor/lidar_front/points")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--storage-id", default="mcap")
    parser.add_argument("--max-range", type=float, default=60.0)
    args = parser.parse_args()

    timer = Timer()
    plane_filter = PlaneFilter()
    point_counts, gated_counts, cluster_counts, planned = [], [], [], 0
    frames = no_start = 0
    routes = defaultdict(int)

    for msg in read_frames(args.bag_uri, args.topic, args.frames, args.storage_id):
        points = timer.time("decode", pointcloud2_to_xyz, msg)
        point_counts.append(points.shape[0])

        # The nodes gate by range before anything else; returns at 400 m are
        # noise for a ground fit and would dominate the costmap bounds.
        radius = np.linalg.norm(points[:, :2], axis=1)
        points = points[(radius > 0.5) & (radius < args.max_range)]
        gated_counts.append(points.shape[0])
        if points.shape[0] < 100:
            continue

        mask, model = timer.time("segment", segment_ground, points)
        plane = plane_filter.update(model)
        ground, obstacles = points[mask], points[~mask]

        clusters = timer.time("cluster", cluster_obstacles, obstacles)
        cluster_counts.append(len(clusters))

        grid = timer.time(
            "costmap", build_costmap, ground[:, :2], obstacles[:, :2], plane=plane
        )
        timer.time("lane", fit_lane, ground)

        start = snap_to_traversable(grid, 0.0, 0.0, 2.0)
        if start is None:
            no_start += 1
        else:
            path, how = timer.time("plan", plan_like_the_node, grid, start)
            planned += bool(path)
            routes[how] += 1
        frames += 1

    if not frames:
        print("no frames processed")
        return 1

    report = timer.report()
    total_p50 = sum(s["p50"] for s in report.values())

    print(f"\nbag   : {args.bag_uri}")
    print(f"topic : {args.topic}")
    print(f"frames: {frames}")
    print(
        f"points: {int(np.median(point_counts))} median raw, "
        f"{int(np.median(gated_counts))} after the {args.max_range:.0f} m range gate"
    )
    print(f"clusters/frame: {np.median(cluster_counts):.1f} median")
    print(
        f"route found   : {planned}/{frames} frames "
        f"({routes['goal']} to the requested goal, {routes['fallback']} via fallback)"
    )
    print(f"no start cell : {no_start}/{frames} frames\n")

    print(f"{'stage':<12} {'p50 ms':>8} {'p95 ms':>8} {'p99 ms':>8} {'max ms':>8}")
    print("-" * 46)
    for name in ("decode", "segment", "cluster", "costmap", "lane", "plan"):
        if name not in report:
            continue
        s = report[name]
        print(
            f"{name:<12} {s['p50']:>8.1f} {s['p95']:>8.1f} {s['p99']:>8.1f} {s['max']:>8.1f}"
        )
    print("-" * 46)
    print(f"{'sum of p50':<12} {total_p50:>8.1f}")
    budget = 100.0
    print(
        f"\n10 Hz sensor allows {budget:.0f} ms/sweep: "
        f"{'within budget' if total_p50 <= budget else 'OVER BUDGET'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
