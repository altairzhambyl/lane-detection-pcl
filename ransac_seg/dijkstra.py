#!/usr/bin/env python3
"""
Dijkstra node: the accumulated traversal cost from the ego origin outwards.

The file this replaces (``djikstra.py``, misspelled) ran a Dijkstra over a set
of ground cells with a uniform step cost of 1.0, threw the distances away and
published only the set of cells it had reached — which is exactly what a flood
fill gives you, so the search was doing no work the BFS node did not already
do. Here the distances are the product: every cell is coloured by what it
actually costs to drive there over the weighted costmap, with cost rising near
obstacles. That field is the thing BFS structurally cannot produce, because
BFS counts hops and ignores cost entirely.
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
    dijkstra,
    normalised_bands,
    validate_connectivity,
)

MARKER_NS = "dijkstra_cost"


class DijkstraPlanner(PerceptionNode):
    """Publishes the weighted cost-from-ego field over the whole costmap."""

    needs_costmap = True

    def __init__(self):
        """Declare the field tunables and create the marker publisher."""
        super().__init__(
            "dijkstra_planner_node", default_marker_topic="/planning/dijkstra_cost"
        )
        self.start_search_radius = float(
            declare(
                self,
                "start_search_radius",
                2.0,
                "Radius in metres searched for a traversable cell at the ego origin",
            )
        )
        self.connectivity = int(
            declare(self, "connectivity", 8, "Grid neighbourhood: 4 or 8")
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
        """Publish one uniformly coloured CUBE_LIST per cost band."""
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
        """Run Dijkstra from the ego origin and publish the cost field."""
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

        cost_field, _ = dijkstra(grid, start, connectivity=self.connectivity)
        reachable = np.isfinite(cost_field)
        self._publish_field(frame, grid, cost_field, reachable)
        if reachable.any():
            self.get_logger().info(
                f"reachable cells={int(reachable.sum())} "
                f"max cost={float(cost_field[reachable].max()):.1f}",
                throttle_duration_sec=1.0,
            )


# The launch files and the end-to-end tests have referred to every node in this
# package as RoadSquares since the first commit; the alias keeps that working.
RoadSquares = DijkstraPlanner


def main(args=None):
    """Spin the Dijkstra cost-field node."""
    rclpy.init(args=args)
    node = DijkstraPlanner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
