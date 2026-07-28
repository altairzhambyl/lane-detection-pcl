#!/usr/bin/env python3
"""
End-to-end smoke tests driving each ROS 2 node callback with a real PointCloud2.

Skipped automatically when ROS 2 is not on the path, so the numpy-only test
suite still runs on a bare checkout.
"""

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
point_cloud2 = pytest.importorskip("sensor_msgs_py.point_cloud2")
std_msgs = pytest.importorskip("std_msgs.msg")

from ransac_seg.astar import RoadSquares as AStarNode  # noqa: E402
from ransac_seg.curve import RoadSquares as CurveNode  # noqa: E402
from ransac_seg.ransac_node import RoadSquares  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def ros_context():
    """Initialise and tear down rclpy once for the module."""
    rclpy.init()
    yield
    rclpy.shutdown()


def make_cloud_msg(frame_id="lidar_front"):
    """Build a PointCloud2 holding a sloped road plus obstacle clutter."""
    rng = np.random.default_rng(19)
    tilt = np.radians(6.0)

    # Road surface: a 40 x 30 m patch pitched 6 degrees, sensor 1.7 m up.
    xs = rng.uniform(0.0, 40.0, 6000)
    ys = rng.uniform(-15.0, 15.0, 6000)
    zs = -1.7 + xs * np.tan(tilt) + rng.normal(0.0, 0.01, 6000)
    road = np.column_stack([xs, ys, zs])

    # Obstacles: a wall of returns standing well above the surface.
    ox = rng.uniform(0.0, 40.0, 2000)
    oy = rng.uniform(-15.0, 15.0, 2000)
    oz = -1.7 + ox * np.tan(tilt) + rng.uniform(0.8, 3.0, 2000)
    obstacles = np.column_stack([ox, oy, oz])

    header = std_msgs.Header()
    header.frame_id = frame_id
    cloud = np.vstack([road, obstacles]).astype(np.float32)
    return point_cloud2.create_cloud_xyz32(header, cloud.tolist())


@pytest.mark.parametrize("node_cls", [RoadSquares, CurveNode, AStarNode])
def test_callback_publishes_without_error(node_cls):
    """Each node consumes a real cloud and publishes markers."""
    node = node_cls()
    published = []

    for attr in dir(node):
        pub = getattr(node, attr)
        if hasattr(pub, "publish") and hasattr(pub, "topic_name"):
            pub.publish = lambda msg, sink=published: sink.append(msg)

    try:
        node.callback(make_cloud_msg())
    finally:
        node.destroy_node()

    assert published, f"{node_cls.__name__} published nothing"
    assert any(len(m.points) > 0 for m in published), "markers were empty"


def test_empty_cloud_is_handled():
    """A cloud with no returns must not raise."""
    header = std_msgs.Header()
    header.frame_id = "lidar_front"
    empty = point_cloud2.create_cloud_xyz32(header, [])

    node = RoadSquares()
    try:
        node.callback(empty)
    finally:
        node.destroy_node()
