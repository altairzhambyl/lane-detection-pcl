#!/usr/bin/env python3
"""
Bring up the whole stack: perception plus all four planning nodes.

This used to start five of the six nodes — BFS was missing — while the README
advertised that it started everything and shipped an A*/BFS/DFS comparison
screenshot. All six start here, each with the node name it registers under, so
the comparison the documentation describes is the one you actually get.

The input topic and the costmap resolution are launch arguments passed to every
node, so retargeting a different bag is one flag rather than six parameter
files.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# (executable, node name) — the executables must match the console_scripts
# entry points declared in setup.py. The planner names are the ones their
# modules register with, so the logs and `ros2 node list` agree.
NODES = (
    ("ransac_node", "ground_segmentation_node"),
    ("curve", "lane_boundary_node"),
    ("astar", "astar_planner_node"),
    ("bfs", "bfs_planner_node"),
    ("dfs", "dfs_baseline_node"),
    ("dijkstra", "dijkstra_planner_node"),
)


def generate_launch_description():
    """Declare the shared arguments and start every node in the package."""
    # A launch argument arrives as a string; resolution is declared a double,
    # so it has to be typed explicitly or rclpy rejects the assignment.
    shared = {
        "input_topic": ParameterValue(
            LaunchConfiguration("input_topic"), value_type=str
        ),
        "resolution": ParameterValue(
            LaunchConfiguration("resolution"), value_type=float
        ),
    }

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "input_topic",
                default_value="/sensor/lidar_front/points",
                description="PointCloud2 topic every node subscribes to",
            ),
            DeclareLaunchArgument(
                "resolution",
                default_value="0.5",
                description="Costmap cell size in metres",
            ),
        ]
        + [
            Node(
                package="ransac_seg",
                executable=executable,
                name=name,
                output="screen",
                parameters=[shared],
            )
            for executable, name in NODES
        ]
    )
