#!/usr/bin/env python3
"""
BFS node: unweighted reachability and hop distance from the ego origin.

This is the baseline the weighted Dijkstra field is read against. BFS answers
one question honestly and completely — which cells are connected to the
vehicle, and how many 4-connected steps away — and it answers nothing about
cost, so a cell squeezed against an obstacle's clearance margin looks exactly
like one in the middle of the road. Publishing both fields side by side is
what makes the cost model visible.
"""

import numpy as np
import rclpy
from visualization_msgs.msg import MarkerArray

from ransac_seg.grid import snap_to_traversable
from ransac_seg.markers import cube_list, delete_all
from ransac_seg.params import declare
from ransac_seg.perception_node import PerceptionNode
from ransac_seg.planners import (
    band_color,
    bfs,
    normalised_bands,
    validate_connectivity,
)

MARKER_NS = "bfs_hops"


class BFSPlanner(PerceptionNode):
    """Publishes the 4-connected hop field reachable from the ego origin."""

    needs_costmap = True

    def __init__(self):
        """Declare the field tunables and create the marker publisher."""
        super().__init__("bfs_planner_node", default_marker_topic="/planning/bfs_hops")
        self.start_search_radius = float(
            declare(
                self,
                "start_search_radius",
                2.0,
                "Radius in metres searched for a traversable cell at the ego origin",
            )
        )
        self.connectivity = int(
            declare(self, "connectivity", 4, "Grid neighbourhood: 4 or 8")
        )
        # Fail here rather than raising out of every callback: a bad
        # launch value should stop the node, not flood the log.
        validate_connectivity(self.connectivity)
        self.field_bands = int(
            declare(
                self, "field_bands", 6, "Colour levels the published field is cut into"
            )
        )
        self.publisher = self.create_publisher(
            MarkerArray, self.params.marker_topic, self.params.qos_depth
        )

    def _clear(self, frame):
        """Publish an array that erases the previous frame's field."""
        array = MarkerArray()
        array.markers.append(delete_all(frame.frame_id, frame.stamp, MARKER_NS))
        self.publisher.publish(array)

    def _publish_field(self, frame, grid, values, valid):
        """Publish one uniformly coloured CUBE_LIST per hop band."""
        array = MarkerArray()
        # The leading DELETEALL is what retires cells that dropped out of the
        # map this frame; without it a band left empty keeps its old cubes.
        array.markers.append(delete_all(frame.frame_id, frame.stamp, MARKER_NS))
        bands = normalised_bands(values, valid, self.field_bands)
        for band in range(self.field_bands):
            cells = np.argwhere(bands == band)
            if cells.shape[0] == 0:
                continue
            array.markers.append(
                cube_list(
                    frame.frame_id,
                    frame.stamp,
                    MARKER_NS,
                    band,
                    grid.cell_to_world_xyz(cells),
                    grid.resolution,
                    band_color(band, self.field_bands),
                    lifetime_sec=self.params.marker_lifetime_sec,
                )
            )
        self.publisher.publish(array)

    def on_frame(self, frame):
        """Flood the costmap from the ego origin and publish the hop field."""
        grid = frame.grid
        if grid is None:
            self._clear(frame)
            return

        start = snap_to_traversable(grid, 0.0, 0.0, self.start_search_radius)
        if start is None:
            self.get_logger().warn(
                "no traversable cell within "
                f"{self.start_search_radius:.1f} m of the ego origin; not searching",
                throttle_duration_sec=1.0,
            )
            self._clear(frame)
            return

        hops = bfs(grid, start, connectivity=self.connectivity)
        reached = hops >= 0
        self._publish_field(frame, grid, hops, reached)
        if reached.any():
            self.get_logger().info(
                f"reachable cells={int(reached.sum())} "
                f"max hops={int(hops[reached].max())}",
                throttle_duration_sec=1.0,
            )


# The launch files and the end-to-end tests have referred to every node in this
# package as RoadSquares since the first commit; the alias keeps that working.
RoadSquares = BFSPlanner


def main(args=None):
    """Spin the BFS reachability node."""
    rclpy.init(args=args)
    node = BFSPlanner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
