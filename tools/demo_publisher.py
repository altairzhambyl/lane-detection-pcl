#!/usr/bin/env python3
"""
Publish a synthetic LiDAR stream so the pipeline can be watched live in Foxglove.

Feeds `/sensor/lidar_front/points` — the topic the real nodes subscribe to — so
`ransac_node`, `curve` and the planners run unmodified against it. Alongside
that it runs the same pipeline in-process and publishes the result under
`/demo/...` for visualisation.

That preview lives under its own prefix on purpose. The real nodes publish the
costmap, boundaries and path on `/semantic_map/...` and `/planning/...`; a
second publisher on those topics would interleave two sources into one display
and neither would be trustworthy. Watch `/demo/...` when this runs alone, and
`/semantic_map/...` when the nodes are up.

Usage:
    ros2 launch foxglove_bridge foxglove_bridge_launch.xml   # in one shell
    python3 tools/demo_publisher.py                          # in another

Then connect Foxglove to ws://localhost:8765.
"""

import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from foxglove_demo import (  # noqa: E402
    LEGACY_HEIGHT,
    _pack_xyz_word,
    make_cell_markers,
    make_cloud_msg,
    make_cluster_markers,
    make_lane_markers,
    make_path_marker,
    make_plane_marker,
    make_scene,
    make_stats_marker,
    run_pipeline,
    score,
)
from ransac_seg.planners import path_length_m  # noqa: E402

N_FRAMES = 60


def make_raw_cloud(nanos, points):
    """Pack a plain XYZ cloud, the way a real sensor driver would publish it."""
    msg = make_cloud_msg(nanos, points, np.zeros(points.shape[0], dtype=bool))
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    # Same 16-byte stride, but the 4th word carries intensity rather than the
    # packed colour, so the buffer is rebuilt through the shared vectorised
    # packer instead of a per-point struct.pack loop.
    msg.data = _pack_xyz_word(points, np.float32(1.0), "<f4")
    return msg


class DemoLidar(Node):
    """Replays a looping synthetic sweep over ground of changing slope."""

    def __init__(self, rate_hz=10.0, seed=17):
        super().__init__("demo_lidar_publisher")
        self.rng = np.random.default_rng(seed)
        self.frame = 0

        self.pub_raw = self.create_publisher(
            PointCloud2, "/sensor/lidar_front/points", 10
        )
        self.pub_seg = self.create_publisher(PointCloud2, "/lidar/points", 10)
        self.pub_legacy = self.create_publisher(PointCloud2, "/lidar/legacy_points", 10)
        self.pub_plane = self.create_publisher(Marker, "/ransac/plane", 10)
        self.pub_stats = self.create_publisher(Marker, "/ransac/stats", 10)
        self.pub_cells = self.create_publisher(Marker, "/demo/costmap", 10)
        self.pub_boxes = self.create_publisher(MarkerArray, "/demo/obstacles", 10)
        self.pub_lane = [
            self.create_publisher(Marker, topic, 10)
            for topic in ("/demo/road_left", "/demo/road_right", "/demo/road_center")
        ]
        self.pub_path = self.create_publisher(Marker, "/demo/astar_path", 10)

        self.create_timer(1.0 / rate_hz, self.tick)
        self.get_logger().info(
            "publishing /sensor/lidar_front/points at "
            f"{rate_hz:g} Hz — connect Foxglove to ws://localhost:8765"
        )

    def tick(self):
        """Build one sweep, segment it, and publish every visualisation topic."""
        nanos = self.get_clock().now().nanoseconds
        points, truth = make_scene(self.frame % N_FRAMES, N_FRAMES, self.rng)
        self.frame += 1

        result = run_pipeline(points)
        ground_mask, model = result["ground_mask"], result["model"]
        grid, lane, path = result["grid"], result["lane"], result["path"]
        legacy_mask = points[:, 2] < LEGACY_HEIGHT

        r_recall, r_precision = score(ground_mask, truth)
        l_recall, l_precision = score(legacy_mask, truth)
        tilt = model.tilt_deg if model else float("nan")
        inliers = model.inlier_ratio if model else float("nan")
        width = float(np.median(lane.width)) if lane.valid else float("nan")

        text = (
            f"slope {tilt:4.1f} deg\n"
            f"RANSAC   inliers {inliers:5.1%}   "
            f"recall {r_recall:.3f}  precision {r_precision:.3f}\n"
            f"legacy   z < {LEGACY_HEIGHT}          "
            f"recall {l_recall:.3f}  precision {l_precision:.3f}\n"
            f"lane {width:.2f} m   obstacles {len(result['clusters'])}   "
            f"path {path_length_m(grid, path) if path else 0.0:.1f} m"
        )

        self.pub_raw.publish(make_raw_cloud(nanos, points))
        self.pub_seg.publish(make_cloud_msg(nanos, points, ground_mask))
        self.pub_legacy.publish(make_cloud_msg(nanos, points, legacy_mask))
        self.pub_plane.publish(make_plane_marker(nanos, model))
        self.pub_stats.publish(make_stats_marker(nanos, text))
        for marker in make_cell_markers(nanos, grid):
            self.pub_cells.publish(marker)
        self.pub_boxes.publish(make_cluster_markers(nanos, result["clusters"]))
        for publisher, marker in zip(
            self.pub_lane, make_lane_markers(nanos, lane, model)
        ):
            publisher.publish(marker)
        self.pub_path.publish(make_path_marker(nanos, grid, path))


def main(args=None):
    """Spin the demo publisher until interrupted."""
    rclpy.init(args=args)
    node = DemoLidar()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
