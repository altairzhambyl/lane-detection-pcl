#!/usr/bin/env python3
"""
Lane boundary node: publishes the fitted road edges and centreline.

This node used to draw the grid squares as well, on the same topic, namespace
and id as ransac_node.py — two publishers fighting over one namespace, so RViz
showed whichever arrived last. Squares now belong to ransac_node alone and this
node publishes only what it computes: the two boundaries and the centreline
they were measured against, lifted onto the fitted plane instead of z=0.0.
"""

import numpy as np
import rclpy
from visualization_msgs.msg import Marker

from ransac_seg import markers
from ransac_seg.grid import plane_z
from ransac_seg.lane import fit_lane
from ransac_seg.params import declare
from ransac_seg.perception_node import PerceptionNode

_LEFT_NS = "road_left"
_RIGHT_NS = "road_right"
_CENTER_NS = "road_center"

# ROS parameter name -> (fit_lane keyword, default, description). The defaults
# are the function's own, so a launch file overriding nothing gets the tested
# behaviour.
_LANE_PARAMS = (
    ("lane_bin_length", "bin_length", 2.0, "Arc length of one boundary sample bin"),
    (
        "lane_min_points_per_bin",
        "min_points_per_bin",
        8,
        "Returns a bin needs before its edges are trusted",
    ),
    (
        "lane_edge_percentile",
        "edge_percentile",
        2.0,
        "Percentile taken as the road edge; tolerates that fraction of outliers",
    ),
    (
        "lane_centerline_degree",
        "centerline_degree",
        2,
        "Polynomial degree of the fitted centreline",
    ),
    (
        "lane_robust_iterations",
        "robust_iterations",
        3,
        "Median re-fit passes of the centreline",
    ),
    ("lane_max_gap", "max_gap", 6.0, "Boundaries are truncated at gaps this long"),
    (
        "lane_smoothing_per_point",
        "smoothing_per_point",
        0.02,
        "Spline residual budget per occupied bin",
    ),
    ("lane_n_samples", "n_samples", 120, "Points published along each boundary"),
    ("lane_min_width", "min_width", 2.0, "Fits narrower than this are rejected"),
    ("lane_max_width", "max_width", 12.0, "Fits wider than this are rejected"),
    (
        "lane_min_bin_fill",
        "min_bin_fill",
        0.5,
        "Bin population needed relative to the median, dropping partial end bins",
    ),
)


class LaneBoundaryNode(PerceptionNode):
    """Fits and publishes the road boundaries and centreline for each frame."""

    # Boundary fitting reads the ground returns directly; rasterising a costmap
    # this node never consults would cost milliseconds per frame for nothing.
    needs_costmap = False

    def __init__(self):
        """Declare the lane fitting parameters and create the three publishers."""
        super().__init__("lane_boundary_node")
        self.lane_kwargs = {
            keyword: type(default)(declare(self, name, default, description))
            for name, keyword, default, description in _LANE_PARAMS
        }
        topics = (
            ("left_topic", "/semantic_map/road_left", "Topic for the left boundary"),
            ("right_topic", "/semantic_map/road_right", "Topic for the right boundary"),
            ("center_topic", "/semantic_map/road_center", "Topic for the centreline"),
        )
        self.pub_left, self.pub_right, self.pub_center = (
            self.create_publisher(
                Marker,
                declare(self, name, default, description),
                self.params.qos_depth,
            )
            for name, default, description in topics
        )

    def _strip(self, frame, publisher, namespace, xy, color):
        """Publish one boundary as a LINE_STRIP lifted onto the fitted plane."""
        points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        z = plane_z(frame.plane, points[:, 0], points[:, 1])
        publisher.publish(
            markers.line_strip(
                frame.frame_id,
                frame.stamp,
                namespace,
                0,
                np.column_stack([points, np.broadcast_to(z, points[:, 0].shape)]),
                color,
                lifetime_sec=self.params.marker_lifetime_sec,
            )
        )

    def on_frame(self, frame):
        """Fit the lane for one frame and publish it, or clear it."""
        fit = fit_lane(frame.ground_xyz, **self.lane_kwargs)
        if not fit.valid:
            self.get_logger().info(
                f"no lane boundary published: {fit.reason}",
                throttle_duration_sec=5.0,
            )
            for publisher, namespace in (
                (self.pub_left, _LEFT_NS),
                (self.pub_right, _RIGHT_NS),
                (self.pub_center, _CENTER_NS),
            ):
                publisher.publish(
                    markers.delete_all(frame.frame_id, frame.stamp, namespace)
                )
            return

        self._strip(frame, self.pub_left, _LEFT_NS, fit.left, (0.0, 0.4, 1.0, 0.9))
        self._strip(frame, self.pub_right, _RIGHT_NS, fit.right, (1.0, 0.2, 0.0, 0.9))
        self._strip(
            frame, self.pub_center, _CENTER_NS, fit.centerline, (1.0, 0.9, 0.0, 0.7)
        )

    # test_nodes_end_to_end.py drives the subscription callback by its old
    # name; the integrate step renames it there.
    callback = PerceptionNode._on_cloud


# The setup.py entry point and the end-to-end tests still reference the old
# class name. Kept until the integrate step updates both.
RoadSquares = LaneBoundaryNode


def main(args=None):
    """Spin the lane boundary node."""
    rclpy.init(args=args)
    node = LaneBoundaryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
