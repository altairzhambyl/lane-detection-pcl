#!/usr/bin/env python3
"""
A* planner node: one least-cost route across the frame's costmap.

The previous version searched a bare set of cells holding ground returns,
started from "whichever cell was nearest the origin this frame" and aimed at
"whichever cell was furthest away this frame", so the goal jumped every time
the sweep changed shape. It also used a Euclidean heuristic on a 4-connected
graph and had no CLOSED set, so it re-expanded cells it had already settled.

Now the start is the ego origin, the goal is a parameter in metres snapped
onto the map, and the search runs over the tri-state costmap so obstacles are
obstacles rather than holes in the ground returns.
"""

import numpy as np
import rclpy
from visualization_msgs.msg import Marker

from ransac_seg.grid import snap_to_traversable
from ransac_seg.markers import delete_all, line_strip
from ransac_seg.params import declare
from ransac_seg.perception_node import PerceptionNode
from ransac_seg.planners import (
    astar,
    dijkstra,
    furthest_reachable,
    path_cost,
    path_length_m,
    select_goal,
    validate_connectivity,
)

MARKER_NS = "astar_path"
PATH_COLOR = (1.0, 0.25, 0.2, 0.95)


class AStarPlanner(PerceptionNode):
    """Plans and publishes a single least-cost route to a parameterised goal."""

    needs_costmap = True

    def __init__(self):
        """Declare the planner tunables and create the path publisher."""
        super().__init__(
            "astar_planner_node", default_marker_topic="/planning/astar_path"
        )
        self.goal_x = float(declare(self, "goal_x", 25.0, "Goal x in the sensor frame"))
        self.goal_y = float(declare(self, "goal_y", 0.0, "Goal y in the sensor frame"))
        self.goal_search_radius = float(
            declare(
                self,
                "goal_search_radius",
                5.0,
                "Radius in metres searched for a traversable cell near the goal",
            )
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
        self.path_width = float(
            declare(self, "path_width", 0.15, "Published path line width in metres")
        )
        # Which way "onwards" points when the requested goal is unusable. It is
        # +x for a forward-facing sensor, but a rear or side mount needs its own.
        self.fallback_axis = tuple(
            float(v)
            for v in declare(
                self,
                "goal_fallback_axis",
                [1.0, 0.0],
                "Direction the fallback goal is chosen furthest along",
            )
        )
        self.publisher = self.create_publisher(
            Marker, self.params.marker_topic, self.params.qos_depth
        )

    def _clear(self, frame):
        """Publish a marker that erases the previous frame's path."""
        self.publisher.publish(delete_all(frame.frame_id, frame.stamp, MARKER_NS))

    def on_frame(self, frame):
        """Plan from the ego origin to the goal and publish the route."""
        grid = frame.grid
        if grid is None:
            self._clear(frame)
            return

        start = snap_to_traversable(grid, 0.0, 0.0, self.start_search_radius)
        if start is None:
            self.get_logger().warn(
                "no traversable cell within "
                f"{self.start_search_radius:.1f} m of the ego origin; not planning",
                throttle_duration_sec=1.0,
            )
            self._clear(frame)
            return

        path = self._plan(grid, start)
        if not path:
            self.get_logger().warn(
                "no route to any goal in this frame", throttle_duration_sec=1.0
            )
            self._clear(frame)
            return

        self.publisher.publish(
            line_strip(
                frame.frame_id,
                frame.stamp,
                MARKER_NS,
                0,
                grid.cell_to_world_xyz(np.asarray(path, dtype=np.int64)),
                PATH_COLOR,
                width=self.path_width,
                lifetime_sec=self.params.marker_lifetime_sec,
            )
        )
        self.get_logger().info(
            f"path cost={path_cost(grid, path):.1f} "
            f"length={path_length_m(grid, path):.1f} m over {len(path)} cells",
            throttle_duration_sec=1.0,
        )

    def _plan(self, grid, start):
        """Return the cell route to the requested goal, or to a reachable one."""
        axis = self.fallback_axis
        goal = select_goal(
            grid,
            (self.goal_x, self.goal_y),
            search_radius=self.goal_search_radius,
            fallback_axis=axis,
        )
        if goal is not None:
            path, _ = astar(grid, start, goal, connectivity=self.connectivity)
            if path:
                return path

        # The requested goal is either off the map or walled off from the ego
        # position. Rather than publish nothing, aim at the furthest cell that
        # is provably connected to the start — the Dijkstra field is what
        # makes "provably" true and keeps the choice stable between frames.
        cost_field, _ = dijkstra(grid, start, connectivity=self.connectivity)
        fallback = furthest_reachable(cost_field, grid, axis)
        if fallback is None or fallback == start:
            return []
        path, _ = astar(grid, start, fallback, connectivity=self.connectivity)
        return path


# The launch files and the end-to-end tests have referred to every node in this
# package as RoadSquares since the first commit; the alias keeps that working.
RoadSquares = AStarPlanner


def main(args=None):
    """Spin the A* planner node."""
    rclpy.init(args=args)
    node = AStarPlanner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
