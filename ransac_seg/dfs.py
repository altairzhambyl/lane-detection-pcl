#!/usr/bin/env python3
"""
DFS node: a traversal-order baseline, and explicitly not a planner.

Say this plainly, because the repository used to imply otherwise by shipping
DFS alongside A* as if they were alternatives: depth-first search explores the
exact same reachable set as breadth-first search. Connectivity is a property of
the graph, not of the order you walk it, so DFS finds nothing BFS misses and
misses nothing BFS finds. It produces no shortest path, no hop distance and no
cost. Nobody should drive behind it.

What it does produce, and the only thing that differs from BFS, is the order of
discovery — a long snaking chain instead of expanding rings — which is worth
seeing once because it is the clearest picture of why "reachable" and "optimal"
are different questions. To keep that claim honest rather than decorative, this
node re-runs BFS on the same grid every second and logs whether the two visited
sets actually match, so the assertion is measured on live data instead of
asserted in a README.
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
    dfs,
    normalised_bands,
    validate_connectivity,
)

MARKER_NS = "dfs_order"


class DFSBaseline(PerceptionNode):
    """Publishes depth-first discovery order and audits it against BFS."""

    needs_costmap = True

    def __init__(self):
        """Declare the field tunables and create the marker publisher."""
        super().__init__(
            "dfs_baseline_node", default_marker_topic="/planning/dfs_order"
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
        """Publish one uniformly coloured CUBE_LIST per discovery-order band."""
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
        """Walk the costmap depth-first, publish the order, audit against BFS."""
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

        order = dfs(grid, start, connectivity=self.connectivity)
        visited = order >= 0
        self._publish_field(frame, grid, order, visited)
        self._log_equivalence(grid, start, visited)

    def _log_equivalence(self, grid, start, visited):
        """Log the DFS-vs-BFS visited-set comparison this node's claim rests on."""
        reached = bfs(grid, start, connectivity=self.connectivity) >= 0
        identical = bool(np.array_equal(visited, reached))
        self.get_logger().info(
            f"dfs visited={int(visited.sum())} bfs visited={int(reached.sum())} "
            f"sets identical={identical}; dfs gives order only, no path and no cost",
            throttle_duration_sec=1.0,
        )


# The launch files and the end-to-end tests have referred to every node in this
# package as RoadSquares since the first commit; the alias keeps that working.
RoadSquares = DFSBaseline


def main(args=None):
    """Spin the DFS traversal-order baseline node."""
    rclpy.init(args=args)
    node = DFSBaseline()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
