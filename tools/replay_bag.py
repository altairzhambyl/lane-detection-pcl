#!/usr/bin/env python3
"""
Replay PointCloud2 frames out of a rosbag2 recording onto a live topic.

``rosbag2_py`` appeared in this repo only as a *writer* (``foxglove_demo.py``),
so a recorded drive could be produced but never fed back through the nodes.
This is the read direction: it republishes a bag's cloud topic under the name
the nodes subscribe to, which is what lets a recording captured on a vehicle
drive the same pipeline as the synthetic scene.

Usage:
    python3 tools/replay_bag.py BAG_URI [--in-topic ...] [--out-topic ...]
                                        [--rate 1.0] [--storage-id mcap] [--loop]
"""

import argparse
import os
import sys
import time

import rclpy
import rosbag2_py
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

CLOUD_TYPE = "sensor_msgs/msg/PointCloud2"
DEFAULT_OUT_TOPIC = "/sensor/lidar_front/points"
DEFAULT_STORAGE_ID = "mcap"
DEFAULT_RATE = 1.0

# A bag whose frames are far apart (or whose first frame is late) must not stall
# the replay for minutes, so inter-frame waits are clamped.
MAX_SLEEP_SEC = 5.0


def open_reader(uri, storage_id):
    """
    Open a rosbag2 SequentialReader at ``uri``.

    The StorageOptions/ConverterOptions pair mirrors the writer side in
    ``tools/foxglove_demo.py`` so a bag this repo writes is a bag it can read.
    """
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=uri, storage_id=storage_id),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    return reader


def select_cloud_topic(reader, requested=None):
    """
    Return the bag topic to replay, and the full name -> type mapping.

    With no ``--in-topic`` the first PointCloud2 topic is chosen; the caller
    logs it, because silently picking one of several clouds is a trap.
    """
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    listing = ", ".join(sorted(types)) or "(none)"
    if requested is not None:
        if requested not in types:
            raise ValueError(f"{requested!r} is not in the bag; topics are {listing}")
        return requested, types

    clouds = [name for name, type_name in types.items() if type_name == CLOUD_TYPE]
    if not clouds:
        raise ValueError(
            f"bag has no {CLOUD_TYPE} topic; topics are "
            + (", ".join(f"{n} ({t})" for n, t in sorted(types.items())) or "(none)")
        )
    return sorted(clouds)[0], types


class BagReplayer(Node):
    """Streams a bag's cloud topic onto a live topic with wall-clock pacing."""

    def __init__(
        self,
        uri,
        in_topic=None,
        out_topic=DEFAULT_OUT_TOPIC,
        rate=DEFAULT_RATE,
        storage_id=DEFAULT_STORAGE_ID,
        loop=False,
    ):
        super().__init__("bag_replayer")

        self.uri = self.declare_parameter("bag_uri", uri).value
        self.out_topic = self.declare_parameter("out_topic", out_topic).value
        self.storage_id = self.declare_parameter("storage_id", storage_id).value
        self.rate = float(self.declare_parameter("rate", float(rate)).value)
        self.loop = bool(self.declare_parameter("loop", loop).value)
        # An empty string is the "unset" sentinel: ROS parameters have no None.
        requested = self.declare_parameter("in_topic", in_topic or "").value
        self.requested_topic = requested or None

        if self.rate <= 0.0:
            raise ValueError(f"rate must be positive, got {self.rate}")

        self.publisher = self.create_publisher(PointCloud2, self.out_topic, 10)
        self.in_topic = None
        self.published = 0

    def _open(self):
        """Open the bag and apply a storage filter to the chosen topic."""
        reader = open_reader(self.uri, self.storage_id)
        topic, types = select_cloud_topic(reader, self.requested_topic)
        if types[topic] != CLOUD_TYPE:
            raise ValueError(f"{topic!r} carries {types[topic]}, not {CLOUD_TYPE}")
        if self.in_topic is None:
            self.get_logger().info(
                f"replaying {topic!r} ({types[topic]}) from {self.uri} "
                f"onto {self.out_topic!r} at {self.rate:g}x"
                + (" (looping)" if self.loop else "")
            )
        self.in_topic = topic
        reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
        return reader

    def run(self):
        """Replay the bag once, or forever when ``--loop`` was given."""
        while rclpy.ok():
            self._replay_once()
            if not self.loop:
                break
        self.get_logger().info(f"published {self.published} cloud(s)")

    def _replay_once(self):
        """Read every filtered frame and republish it in bag-timestamp order."""
        reader = self._open()
        previous_ns = None
        while reader.has_next() and rclpy.ok():
            _, payload, stamp_ns = reader.read_next()
            if previous_ns is not None:
                # Bag timestamps are the ground truth for pacing; --rate scales
                # them, so 2.0 replays a 10 Hz recording at 20 Hz.
                wait = (stamp_ns - previous_ns) / 1e9 / self.rate
                if wait > 0.0:
                    # A plain sleep, not clock.sleep_for: nothing spins this
                    # node, so a ROS-clock wait would never be serviced.
                    time.sleep(min(wait, MAX_SLEEP_SEC))
            previous_ns = stamp_ns

            msg = deserialize_message(payload, PointCloud2)
            self.publisher.publish(msg)
            self.published += 1


def parse_args(argv):
    """Parse the command line for the bag replayer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag_uri", help="rosbag2 directory or file")
    parser.add_argument(
        "--in-topic",
        default=None,
        help="bag topic to read (default: first PointCloud2 topic)",
    )
    parser.add_argument(
        "--out-topic", default=DEFAULT_OUT_TOPIC, help="topic to republish on"
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_RATE,
        help="playback speed multiplier applied to the bag timestamps",
    )
    parser.add_argument(
        "--storage-id", default=DEFAULT_STORAGE_ID, help="rosbag2 storage plugin"
    )
    parser.add_argument(
        "--loop", action="store_true", help="restart the bag when it ends"
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Replay the bag until it ends or the process is interrupted."""
    argv = list(sys.argv[1:] if argv is None else argv)
    ros_args = []
    if "--ros-args" in argv:
        split = argv.index("--ros-args")
        argv, ros_args = argv[:split], argv[split:]
    args = parse_args(argv)

    rclpy.init(args=ros_args or None)
    node = BagReplayer(
        args.bag_uri,
        in_topic=args.in_topic,
        out_topic=args.out_topic,
        rate=args.rate,
        storage_id=args.storage_id,
        loop=args.loop,
    )
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
