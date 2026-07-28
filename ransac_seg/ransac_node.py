#!/usr/bin/env python3
"""
Ground, obstacle and inflation visualiser for the fitted costmap.

This node used to quantise ground returns into a Python set of cell keys and
draw them at z=0.0 under the namespace "road_squares" — the same namespace and
id that curve.py published on the same topic, so in RViz whichever node ran
last silently erased the other. It also had no obstacle channel at all.

Now it draws the three costmap states in separate namespaces, publishes
per-object boxes for the clustered obstacles, and places every marker on the
fitted plane: on the validated 10 degree slope case, pinning geometry to z=0.0
puts it metres off the surface at the far end of the sweep.
"""

import rclpy
from visualization_msgs.msg import Marker, MarkerArray

from ransac_seg import markers
from ransac_seg.cluster import cluster_obstacles
from ransac_seg.grid import FREE, INFLATED, OCCUPIED
from ransac_seg.params import declare
from ransac_seg.perception_node import PerceptionNode

_GROUND_NS = "ground"
_OBSTACLE_NS = "obstacles"
_INFLATION_NS = "inflation"
_CLUSTER_NS = "obstacle_boxes"


class GroundSegmentationNode(PerceptionNode):
    """Publishes the tri-state costmap and the clustered obstacle boxes."""

    def __init__(self):
        """Declare the marker and clustering parameters and create publishers."""
        super().__init__(
            "ground_segmentation_node", default_marker_topic="/semantic_map/squares"
        )
        self.marker_alpha = float(
            declare(
                self, "marker_alpha", 0.5, "Opacity of the ground and obstacle tiles"
            )
        )
        self.inflation_alpha = float(
            declare(
                self,
                "inflation_alpha",
                0.25,
                "Opacity of the inflation tiles; kept low so they read as a margin",
            )
        )
        self.cluster_kwargs = {
            "voxel_size": float(
                declare(
                    self, "cluster_voxel_size", 0.2, "Voxel size used to link obstacles"
                )
            ),
            "link_radius": float(
                declare(
                    self,
                    "cluster_link_radius",
                    0.6,
                    "Returns within this distance join the same obstacle",
                )
            ),
            "min_points": int(
                declare(self, "cluster_min_points", 12, "Smallest reportable obstacle")
            ),
            "min_height": float(
                declare(
                    self,
                    "cluster_min_height",
                    0.15,
                    "Clusters shorter than this are ground scraps, not obstacles",
                )
            ),
            "max_height": float(
                declare(
                    self,
                    "cluster_max_height",
                    4.0,
                    "Clusters taller than this are ceilings or overhangs",
                )
            ),
        }
        self.obstacle_marker_topic = declare(
            self,
            "obstacle_marker_topic",
            "/semantic_map/obstacles",
            "Topic for the per-obstacle bounding boxes",
        )

        self.pub_cells = self.create_publisher(
            Marker, self.params.marker_topic, self.params.qos_depth
        )
        self.pub_boxes = self.create_publisher(
            MarkerArray, self.obstacle_marker_topic, self.params.qos_depth
        )

    def on_frame(self, frame):
        """Draw the costmap states and the obstacle boxes for one frame."""
        if frame.grid is None:
            # An ADD marker holding zero points reads as "no change", leaving
            # the previous frame's grid on screen; a DELETEALL actually clears.
            for namespace in (_GROUND_NS, _OBSTACLE_NS, _INFLATION_NS):
                self.pub_cells.publish(
                    markers.delete_all(frame.frame_id, frame.stamp, namespace)
                )
            empty = MarkerArray()
            empty.markers.append(
                markers.delete_all(frame.frame_id, frame.stamp, _CLUSTER_NS)
            )
            self.pub_boxes.publish(empty)
            return

        grid = frame.grid
        layers = (
            (_GROUND_NS, 0, FREE, (0.0, 1.0, 0.0, self.marker_alpha)),
            (_OBSTACLE_NS, 1, OCCUPIED, (1.0, 0.0, 0.0, self.marker_alpha)),
            (_INFLATION_NS, 2, INFLATED, (1.0, 0.55, 0.0, self.inflation_alpha)),
        )
        for namespace, marker_id, state, color in layers:
            centres = grid.cell_to_world_xyz(grid.cells_with_state(state))
            self.pub_cells.publish(
                markers.cube_list(
                    frame.frame_id,
                    frame.stamp,
                    namespace,
                    marker_id,
                    centres,
                    grid.resolution,
                    color,
                    lifetime_sec=self.params.marker_lifetime_sec,
                )
            )

        clusters = cluster_obstacles(frame.obstacle_xyz, **self.cluster_kwargs)
        centres = [c.centroid for c in clusters]
        extents = [c.extent for c in clusters]
        self.pub_boxes.publish(
            markers.box_array(
                frame.frame_id,
                frame.stamp,
                _CLUSTER_NS,
                0,
                centres,
                extents,
                (1.0, 0.3, 0.0, 0.6),
                lifetime_sec=self.params.marker_lifetime_sec,
            )
        )

    # test_nodes_end_to_end.py drives the subscription callback by its old
    # name; the integrate step renames it there.
    callback = PerceptionNode._on_cloud


# The setup.py entry point and the end-to-end tests still reference the old
# class name. Kept until the integrate step updates both.
RoadSquares = GroundSegmentationNode


def main(args=None):
    """Spin the ground segmentation node."""
    rclpy.init(args=args)
    node = GroundSegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
