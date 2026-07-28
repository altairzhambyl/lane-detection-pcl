#!/usr/bin/env python3
"""
Bring up the C++ stack: one perception process feeding all six consumers.

algorithms.launch.py starts six Python processes, each of which subscribes to
the same PointCloud2 and independently decodes, range-gates, segments and
rasterises it. This starts ONE `perception_stack` node that does that work once
per sweep and hands the result to six consumers in-process.

`separate_nodes:=true` instead starts six single-consumer nodes, reproducing
the Python topology in C++ so the two can be benchmarked head to head rather
than compared by assertion. The arguments mirror algorithms.launch.py's, so
retargeting a bag is the same flag in both.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# (consumer name, node name). The consumer names are the parameter prefixes the
# C++ consumers declare under; the node names are the ones the Python nodes
# register with, so `ros2 node list` reads the same for either topology.
CONSUMERS = (
    ("ground", "ground_segmentation_node"),
    ("lane", "lane_boundary_node"),
    ("astar", "astar_planner_node"),
    ("bfs", "bfs_planner_node"),
    ("dfs", "dfs_baseline_node"),
    ("dijkstra", "dijkstra_planner_node"),
)


def generate_launch_description():
    """Declare the shared arguments and start one topology or the other."""
    # A launch argument arrives as a string; resolution is declared a double,
    # so it has to be typed explicitly or the parameter set is rejected.
    shared = {
        "input_topic": ParameterValue(
            LaunchConfiguration("input_topic"), value_type=str
        ),
        "resolution": ParameterValue(
            LaunchConfiguration("resolution"), value_type=float
        ),
    }
    separate = LaunchConfiguration("separate_nodes")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "input_topic",
                default_value="/sensor/lidar_front/points",
                description="PointCloud2 topic the perception node reads",
            ),
            DeclareLaunchArgument(
                "resolution",
                default_value="0.5",
                description="Costmap cell size in metres",
            ),
            DeclareLaunchArgument(
                "separate_nodes",
                default_value="false",
                description=(
                    "Start six single-consumer nodes instead of one shared "
                    "perception node, reproducing the Python topology"
                ),
            ),
            Node(
                package="ransac_seg_cpp",
                executable="perception_stack",
                name="perception_stack",
                output="screen",
                parameters=[shared],
                condition=UnlessCondition(separate),
            ),
        ]
        + [
            Node(
                package="ransac_seg_cpp",
                executable="perception_node",
                name=node_name,
                output="screen",
                parameters=[dict(shared, consumer=consumer)],
                condition=IfCondition(separate),
            )
            for consumer, node_name in CONSUMERS
        ]
    )
