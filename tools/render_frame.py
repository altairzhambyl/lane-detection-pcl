#!/usr/bin/env python3
"""
Render one frame of the pipeline running on a recorded drive, as a PNG.

Foxglove and RViz show this live, but a README needs a still that anybody can
regenerate without a running ROS graph, a viewer, or the same version of either.
This reads a frame straight out of a rosbag2 recording, runs the real pipeline
over it, and draws what each stage produced.

Everything drawn is measured, not illustrative: the ground/obstacle split comes
from `segment_ground`, the cells from `build_costmap`, the route from `astar`.

Usage:
    python3 tools/render_frame.py BAG_URI [--frame 120] [--out docs/screenshots/x.png]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import rosbag2_py  # noqa: E402
from ransac_seg.cloud_io import pointcloud2_to_xyz  # noqa: E402
from ransac_seg.cluster import cluster_obstacles  # noqa: E402
from ransac_seg.grid import (  # noqa: E402
    FREE,
    INFLATED,
    OCCUPIED,
    UNKNOWN,
    build_costmap,
    snap_to_traversable,
)
from ransac_seg.ground_plane import segment_ground  # noqa: E402
from ransac_seg.lane import fit_lane  # noqa: E402
from ransac_seg.planners import (  # noqa: E402
    astar,
    dijkstra,
    furthest_reachable,
    select_goal,
)
from rclpy.serialization import deserialize_message  # noqa: E402
from sensor_msgs.msg import PointCloud2  # noqa: E402

STATE_COLORS = {
    UNKNOWN: "#e9edf2",
    FREE: "#3cc07a",
    INFLATED: "#f2c14e",
    OCCUPIED: "#d94f4f",
}
STATE_LABELS = {
    UNKNOWN: "unobserved",
    FREE: "free",
    INFLATED: "inflated",
    OCCUPIED: "occupied",
}


def read_frame(bag_uri, topic, index, storage_id="mcap"):
    """Return the ``index``-th PointCloud2 on ``topic``."""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_uri, storage_id=storage_id),
        rosbag2_py.ConverterOptions("", ""),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    seen = 0
    while reader.has_next():
        _, payload, _ = reader.read_next()
        if seen == index:
            return deserialize_message(payload, PointCloud2)
        seen += 1
    raise IndexError(f"bag holds only {seen} messages on {topic}")


def plan(grid, start):
    """Plan as AStarPlanner does: requested goal, else furthest reachable."""
    goal = select_goal(grid, (25.0, 0.0), search_radius=5.0, fallback_axis=(1.0, 0.0))
    if goal is not None:
        path, _ = astar(grid, start, goal)
        if path:
            return path, "requested goal"
    cost, _ = dijkstra(grid, start)
    fallback = furthest_reachable(cost, grid, (1.0, 0.0))
    if fallback is None or fallback == start:
        return [], "none"
    path, _ = astar(grid, start, fallback)
    return path, "furthest reachable"


def main():
    """Render the segmentation, costmap and route for one recorded sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag_uri")
    parser.add_argument("--topic", default="/sensor/lidar_front/points")
    parser.add_argument("--frame", type=int, default=120)
    parser.add_argument("--storage-id", default="mcap")
    parser.add_argument("--max-range", type=float, default=60.0)
    parser.add_argument("--out", default="docs/screenshots/real_frame.png")
    args = parser.parse_args()

    msg = read_frame(args.bag_uri, args.topic, args.frame, args.storage_id)
    raw = pointcloud2_to_xyz(msg)
    radius = np.linalg.norm(raw[:, :2], axis=1)
    points = raw[(radius > 0.5) & (radius < args.max_range)]

    mask, model = segment_ground(points)
    ground, obstacles = points[mask], points[~mask]
    clusters = cluster_obstacles(obstacles)
    grid = build_costmap(ground[:, :2], obstacles[:, :2], plane=model)
    lane = fit_lane(ground)

    start = snap_to_traversable(grid, 0.0, 0.0, 2.0)
    path, how = plan(grid, start) if start is not None else ([], "no start cell")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5))

    ax1.scatter(
        obstacles[:, 0], obstacles[:, 1], s=0.6, c="#d94f4f", label="non-ground"
    )
    ax1.scatter(ground[:, 0], ground[:, 1], s=0.6, c="#3cc07a", label="ground (RANSAC)")
    for cluster in clusters:
        lo, hi = cluster.min_bound, cluster.max_bound
        ax1.add_patch(
            plt.Rectangle(
                (lo[0], lo[1]),
                hi[0] - lo[0],
                hi[1] - lo[1],
                fill=False,
                edgecolor="#1f2933",
                linewidth=0.8,
            )
        )
    tilt = f"{model.tilt_deg:.1f}°" if model is not None else "no fit"
    ax1.set_title(
        f"Segmentation + clustering — {ground.shape[0]} ground / "
        f"{obstacles.shape[0]} non-ground, {len(clusters)} clusters, plane tilt {tilt}",
        fontsize=10,
    )
    ax1.legend(loc="upper right", markerscale=12, fontsize=9)

    # Cells are drawn from the grid's own state array, not re-derived.
    extent = (
        grid.origin_x,
        grid.origin_x + grid.state.shape[0] * grid.resolution,
        grid.origin_y,
        grid.origin_y + grid.state.shape[1] * grid.resolution,
    )
    rgb = np.zeros(grid.state.shape + (3,))
    for value, colour in STATE_COLORS.items():
        rgb[grid.state == value] = matplotlib.colors.to_rgb(colour)
    ax2.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower", extent=extent)

    if lane.valid:
        ax2.plot(
            lane.left[:, 0], lane.left[:, 1], color="#1f6feb", lw=1.6, label="lane left"
        )
        ax2.plot(
            lane.right[:, 0],
            lane.right[:, 1],
            color="#8250df",
            lw=1.6,
            label="lane right",
        )
    if path:
        xy = grid.cell_to_world_xyz(np.asarray(path, dtype=np.int64))
        ax2.plot(xy[:, 0], xy[:, 1], color="#111111", lw=2.4, label=f"A* route ({how})")
    ax2.plot(0, 0, marker="o", color="#111111", markersize=7)

    counts = {v: int(np.count_nonzero(grid.state == v)) for v in STATE_COLORS}
    ax2.set_title(
        f"Cost grid at {grid.resolution:.2f} m — "
        + ", ".join(f"{counts[v]} {STATE_LABELS[v]}" for v in STATE_COLORS),
        fontsize=10,
    )
    ax2.legend(loc="upper right", fontsize=9)

    for ax in (ax1, ax2):
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)

    fig.suptitle(
        f"lane-detection-pcl on recorded LiDAR — {raw.shape[0]} returns, "
        f"{points.shape[0]} after the {args.max_range:.0f} m gate "
        f"(frame {args.frame} of {os.path.basename(args.bag_uri)})",
        fontsize=12,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")
    print(
        f"  ground {ground.shape[0]}, non-ground {obstacles.shape[0]}, clusters {len(clusters)}"
    )
    print(f"  cells {counts}")
    print(f"  route {len(path)} cells via {how}; lane valid={lane.valid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
