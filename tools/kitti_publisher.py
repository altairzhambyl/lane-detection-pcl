#!/usr/bin/env python3
"""
Publish recorded KITTI/nuScenes sweeps onto the topic the real nodes subscribe to.

The pipeline could previously only be fed by ``demo_publisher.py``'s procedural
scene, so nothing here had ever seen a real sensor. Replaying a directory of
``.bin``/``.pcd`` files onto ``/sensor/lidar_front/points`` makes every node run
against recorded data with no node changes at all.

Usage:
    python3 tools/kitti_publisher.py PATH [--topic ...] [--frame-id ...]
                                          [--rate 10.0] [--loop]

``PATH`` is a single ``.bin``/``.pcd`` file or a directory of them.
"""

import argparse
import os
import sys

import rclpy
from rclpy.node import Node
from rclpy.task import Future

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from ransac_seg.cloud_io import xyz_to_pointcloud2  # noqa: E402
from ransac_seg.point_files import iter_cloud_files, read_cloud  # noqa: E402
from sensor_msgs.msg import PointCloud2  # noqa: E402

DEFAULT_TOPIC = "/sensor/lidar_front/points"
DEFAULT_FRAME_ID = "lidar_front"
DEFAULT_RATE = 10.0


class CloudFilePublisher(Node):
    """Publishes a list of on-disk sweeps on a timer, optionally looping."""

    def __init__(
        self,
        paths,
        topic=DEFAULT_TOPIC,
        frame_id=DEFAULT_FRAME_ID,
        rate_hz=DEFAULT_RATE,
        loop=False,
    ):
        super().__init__("kitti_publisher")

        # The CLI values are the parameter defaults rather than the values
        # themselves, so a launch file or `--ros-args -p` override still wins.
        self.topic = self.declare_parameter(
            "topic",
            topic,
        ).value
        self.frame_id = self.declare_parameter(
            "frame_id",
            frame_id,
        ).value
        self.rate_hz = float(self.declare_parameter("rate", float(rate_hz)).value)
        self.loop = bool(self.declare_parameter("loop", loop).value)

        if self.rate_hz <= 0.0:
            raise ValueError(f"rate must be positive, got {self.rate_hz}")

        self.paths = list(paths)
        self.index = 0
        self.published = 0
        # Completed when a non-looping replay runs out of sweeps; cancelling the
        # timer alone would leave the process spinning on an idle executor.
        self.finished = Future()

        self.publisher = self.create_publisher(PointCloud2, self.topic, 10)
        self.timer = self.create_timer(1.0 / self.rate_hz, self.tick)
        self.get_logger().info(
            f"replaying {len(self.paths)} sweep(s) on {self.topic} "
            f"at {self.rate_hz:g} Hz in frame {self.frame_id!r}"
            + (" (looping)" if self.loop else "")
        )

    def tick(self):
        """Read and publish the next sweep; stop or wrap at the end."""
        if self.index >= len(self.paths):
            if not self.loop:
                self.get_logger().info(
                    f"published {self.published} sweep(s); replay finished"
                )
                self.timer.cancel()
                self.finished.set_result(self.published)
                return
            self.index = 0

        path = self.paths[self.index]
        self.index += 1
        try:
            points = read_cloud(path)
        except (OSError, ValueError, NotImplementedError) as exc:
            # One unreadable file must not end the replay: recorded datasets
            # routinely carry a truncated final sweep.
            self.get_logger().warning(f"skipping {path}: {exc}")
            return

        stamp = self.get_clock().now().to_msg()
        self.publisher.publish(xyz_to_pointcloud2(points, self.frame_id, stamp))
        self.published += 1
        self.get_logger().debug(f"{path}: {points.shape[0]} points")


def resolve_paths(target):
    """Return the sweep files named by a file or directory argument."""
    if os.path.isdir(target):
        paths = list(iter_cloud_files(target))
        if not paths:
            raise ValueError(f"{target}: no .bin or .pcd files found")
        return paths
    if os.path.isfile(target):
        return [target]
    raise ValueError(f"{target}: no such file or directory")


def parse_args(argv):
    """Parse the command line for the replay publisher."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="a .bin/.pcd sweep or a directory of them")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="output topic")
    parser.add_argument(
        "--frame-id", default=DEFAULT_FRAME_ID, help="header frame_id to stamp"
    )
    parser.add_argument(
        "--rate", type=float, default=DEFAULT_RATE, help="publish rate in Hz"
    )
    parser.add_argument(
        "--loop", action="store_true", help="restart at the first sweep when done"
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Spin the file publisher until the replay ends or it is interrupted."""
    argv = list(sys.argv[1:] if argv is None else argv)
    ros_args = []
    if "--ros-args" in argv:
        split = argv.index("--ros-args")
        argv, ros_args = argv[:split], argv[split:]
    args = parse_args(argv)

    paths = resolve_paths(args.path)

    rclpy.init(args=ros_args or None)
    node = CloudFilePublisher(
        paths,
        topic=args.topic,
        frame_id=args.frame_id,
        rate_hz=args.rate,
        loop=args.loop,
    )
    try:
        if node.loop:
            rclpy.spin(node)
        else:
            rclpy.spin_until_future_complete(node, node.finished)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
