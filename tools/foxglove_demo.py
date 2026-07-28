#!/usr/bin/env python3
"""
Generate a Foxglove-ready MCAP recording of the RANSAC ground-plane segmentation.

Simulates a drive over ground whose slope changes under the sensor, runs the
real `segment_ground()` on every frame, and records both the RANSAC result and
the legacy fixed-height filter side by side so the difference is visible on the
timeline rather than asserted in a README.

Usage:
    python3 tools/foxglove_demo.py [--frames 60] [--out docs/ransac_demo]

Then open the resulting `.mcap` in Foxglove (app.foxglove.dev or the desktop
app) and add a 3D panel. Topics:

    /lidar/points          full cloud, coloured green=road / red=obstacle by RANSAC
    /lidar/legacy_points   same cloud coloured by the old z < 0.2 filter
    /ransac/plane          the fitted plane, drawn as a translucent quad
    /ransac/stats          per-frame tilt, inlier ratio, recall, precision
    /semantic_map/squares  the 0.5 m road grid the planners consume
"""

import argparse
import math
import os
import shutil
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rosbag2_py  # noqa: E402
from builtin_interfaces.msg import Time  # noqa: E402
from geometry_msgs.msg import Point  # noqa: E402
from ransac_seg.ground_plane import segment_ground  # noqa: E402
from rclpy.serialization import serialize_message  # noqa: E402
from sensor_msgs.msg import PointCloud2, PointField  # noqa: E402
from std_msgs.msg import ColorRGBA, Header  # noqa: E402
from visualization_msgs.msg import Marker  # noqa: E402

FRAME_ID = "lidar_front"
SENSOR_HEIGHT = 1.7
LEGACY_HEIGHT = 0.2
GRID_SIZE = 0.5

ROAD_COLOUR = (60, 220, 90)
OBSTACLE_COLOUR = (230, 70, 70)
_POINT_STEP = 16


def _stamp(nanos):
    """Build a builtin_interfaces/Time from a nanosecond count."""
    return Time(sec=int(nanos // 10**9), nanosec=int(nanos % 10**9))


def _header(nanos):
    """Build a std_msgs/Header stamped at ``nanos`` in the sensor frame."""
    header = Header()
    header.frame_id = FRAME_ID
    header.stamp = _stamp(nanos)
    return header


def make_scene(frame_index, n_frames, rng):
    """
    Build one LiDAR sweep over sloped ground with obstacles.

    Returns ``(points, is_road)`` where ``is_road`` is the ground truth used
    only for scoring — the estimator never sees it.
    """
    # Slope sweeps 0 -> 12 degrees and back, as if cresting a rise.
    phase = frame_index / max(n_frames - 1, 1)
    tilt = math.radians(12.0 * math.sin(math.pi * phase))

    n_road = 9000
    xs = rng.uniform(1.0, 45.0, n_road)
    ys = rng.uniform(-14.0, 14.0, n_road)
    zs = -SENSOR_HEIGHT + xs * math.tan(tilt) + rng.normal(0.0, 0.012, n_road)
    road = np.column_stack([xs, ys, zs])

    # Obstacles: two vehicle-sized blocks and a guardrail down the right side.
    blocks = []
    for cx, cy in ((16.0, -3.5), (30.0, 4.0)):
        n = 1200
        bx = rng.uniform(cx, cx + 4.5, n)
        by = rng.uniform(cy, cy + 1.9, n)
        bz = -SENSOR_HEIGHT + bx * math.tan(tilt) + rng.uniform(0.1, 1.6, n)
        blocks.append(np.column_stack([bx, by, bz]))

    n_rail = 1800
    rx = rng.uniform(1.0, 45.0, n_rail)
    ry = rng.uniform(13.4, 13.8, n_rail)
    rz = -SENSOR_HEIGHT + rx * math.tan(tilt) + rng.uniform(0.3, 0.9, n_rail)
    blocks.append(np.column_stack([rx, ry, rz]))

    obstacles = np.vstack(blocks)
    points = np.vstack([road, obstacles])
    is_road = np.zeros(points.shape[0], dtype=bool)
    is_road[:n_road] = True

    order = rng.permutation(points.shape[0])
    return points[order], is_road[order]


def make_cloud_msg(nanos, points, road_mask):
    """
    Pack an XYZ+RGB PointCloud2, colouring road points green and the rest red.

    RGB is written as a packed UINT32 so Foxglove's 3D panel picks it up as a
    colour field without any manual configuration.
    """
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
    ]

    colours = np.where(
        road_mask,
        (ROAD_COLOUR[0] << 16) | (ROAD_COLOUR[1] << 8) | ROAD_COLOUR[2],
        (OBSTACLE_COLOUR[0] << 16) | (OBSTACLE_COLOUR[1] << 8) | OBSTACLE_COLOUR[2],
    ).astype(np.uint32)

    buffer = bytearray()
    for (x, y, z), colour in zip(points, colours):
        buffer += struct.pack("<fffI", float(x), float(y), float(z), int(colour))

    msg = PointCloud2()
    msg.header = _header(nanos)
    msg.height = 1
    msg.width = points.shape[0]
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = _POINT_STEP
    msg.row_step = _POINT_STEP * points.shape[0]
    msg.is_dense = True
    msg.data = bytes(buffer)
    return msg


def make_plane_marker(nanos, model, reach=45.0, half_width=15.0):
    """Draw the fitted plane as a translucent quad spanning the sensor's view."""
    marker = Marker()
    marker.header = _header(nanos)
    marker.ns = "ransac_plane"
    marker.id = 0
    marker.type = Marker.TRIANGLE_LIST
    marker.action = Marker.ADD
    marker.scale.x = marker.scale.y = marker.scale.z = 1.0
    marker.color = ColorRGBA(r=0.25, g=0.55, b=1.0, a=0.28)
    marker.pose.orientation.w = 1.0

    if model is None:
        marker.action = Marker.DELETE
        return marker

    normal, offset = model.normal, model.offset

    def height_at(x, y):
        """Solve the plane equation for z at a given (x, y)."""
        return -(offset + normal[0] * x + normal[1] * y) / normal[2]

    corners = [
        (1.0, -half_width),
        (reach, -half_width),
        (reach, half_width),
        (1.0, half_width),
    ]
    quad = [Point(x=float(x), y=float(y), z=float(height_at(x, y))) for x, y in corners]
    marker.points = [quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]]
    return marker


def make_stats_marker(nanos, text):
    """Render per-frame metrics as a text marker floating above the sensor."""
    marker = Marker()
    marker.header = _header(nanos)
    marker.ns = "ransac_stats"
    marker.id = 0
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD
    marker.pose.position = Point(x=6.0, y=0.0, z=3.5)
    marker.pose.orientation.w = 1.0
    marker.scale.z = 0.9
    marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)
    marker.text = text
    return marker


def make_grid_marker(nanos, road_points):
    """Quantise road points into the 0.5 m grid the path planners consume."""
    marker = Marker()
    marker.header = _header(nanos)
    marker.ns = "road_squares"
    marker.id = 0
    marker.type = Marker.CUBE_LIST
    marker.action = Marker.ADD
    marker.scale.x = marker.scale.y = GRID_SIZE
    marker.scale.z = 0.05
    marker.color = ColorRGBA(r=0.0, g=1.0, b=0.4, a=0.45)
    marker.pose.orientation.w = 1.0

    if road_points.shape[0] == 0:
        return marker

    cells = np.unique(np.floor(road_points[:, :2] / GRID_SIZE).astype(int), axis=0)
    half = GRID_SIZE / 2.0
    marker.points = [
        Point(x=float(gx * GRID_SIZE + half), y=float(gy * GRID_SIZE + half), z=0.0)
        for gx, gy in cells
    ]
    return marker


def score(mask, truth):
    """Return (recall, precision) of a ground mask against the ground truth."""
    hit = int(np.count_nonzero(mask & truth))
    selected = int(np.count_nonzero(mask))
    recall = hit / max(int(np.count_nonzero(truth)), 1)
    precision = hit / selected if selected else 0.0
    return recall, precision


TOPICS = {
    "/lidar/points": "sensor_msgs/msg/PointCloud2",
    "/lidar/legacy_points": "sensor_msgs/msg/PointCloud2",
    "/ransac/plane": "visualization_msgs/msg/Marker",
    "/ransac/stats": "visualization_msgs/msg/Marker",
    "/semantic_map/squares": "visualization_msgs/msg/Marker",
}


def open_writer(uri):
    """Open an MCAP bag writer at ``uri``, replacing any previous recording."""
    if os.path.isdir(uri):
        shutil.rmtree(uri)

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=uri, storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    for name, type_name in TOPICS.items():
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=name, type=type_name, serialization_format="cdr"
            )
        )
    return writer


def render_preview(path, frame_index, n_frames, seed):
    """
    Save a side-view PNG of one frame: RANSAC plane vs the fixed-height cut.

    Foxglove is the real deliverable; this exists so the difference is legible
    in a README or a pull request without opening a viewer.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    for i in range(frame_index + 1):
        points, truth = make_scene(i, n_frames, rng)

    ground_mask, model = segment_ground(points)
    legacy_mask = points[:, 2] < LEGACY_HEIGHT

    # Look down the y axis: a narrow slice keeps the slope readable.
    slice_mask = np.abs(points[:, 1]) < 2.0
    x, z = points[slice_mask, 0], points[slice_mask, 2]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, sharey=True)
    for ax, mask, title in (
        (axes[0], ground_mask, f"RANSAC plane fit  —  tilt {model.tilt_deg:.1f}°"),
        (axes[1], legacy_mask, f"legacy fixed threshold  —  z < {LEGACY_HEIGHT} m"),
    ):
        sel = mask[slice_mask]
        ax.scatter(x[~sel], z[~sel], s=1.5, c="#e04646", label="classed obstacle")
        ax.scatter(x[sel], z[sel], s=1.5, c="#3cdc5a", label="classed road")
        recall, precision = score(mask, truth)
        ax.set_title(
            f"{title}   recall {recall:.3f} / precision {precision:.3f}", fontsize=11
        )
        ax.set_ylabel("z (m)")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left", markerscale=6, fontsize=9)

    axes[1].axhline(LEGACY_HEIGHT, color="#3b82f6", lw=1.2, ls="--")
    if model is not None:
        xs = np.linspace(1.0, 45.0, 2)
        zs = -(model.offset + model.normal[0] * xs) / model.normal[2]
        axes[0].plot(xs, zs, color="#3b82f6", lw=1.4, ls="--", label="fitted plane")

    axes[1].set_xlabel("x (m), looking down the road")
    fig.suptitle("Ground-plane segmentation on sloped terrain", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    """Build the demo recording and print a summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=60, help="number of sweeps")
    parser.add_argument("--rate", type=float, default=10.0, help="sweep rate in Hz")
    parser.add_argument(
        "--out", default="docs/ransac_demo", help="output bag directory"
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--preview",
        default="docs/screenshots/ransac_vs_threshold.png",
        help="also render a side-view PNG here ('' to skip)",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    writer = open_writer(args.out)
    period_ns = int(1e9 / args.rate)

    ransac_scores, legacy_scores, fallbacks = [], [], 0

    for i in range(args.frames):
        nanos = (i + 1) * period_ns
        points, truth = make_scene(i, args.frames, rng)

        ground_mask, model = segment_ground(points)
        legacy_mask = points[:, 2] < LEGACY_HEIGHT

        if model is None:
            fallbacks += 1
        ransac_scores.append(score(ground_mask, truth))
        legacy_scores.append(score(legacy_mask, truth))

        tilt = model.tilt_deg if model else float("nan")
        inliers = model.inlier_ratio if model else float("nan")
        text = (
            f"frame {i + 1}/{args.frames}\n"
            f"RANSAC  tilt {tilt:5.1f} deg  inliers {inliers:5.1%}  "
            f"recall {ransac_scores[-1][0]:.3f}  precision {ransac_scores[-1][1]:.3f}\n"
            f"legacy z<{LEGACY_HEIGHT}      "
            f"recall {legacy_scores[-1][0]:.3f}  precision {legacy_scores[-1][1]:.3f}"
        )

        writer.write(
            "/lidar/points",
            serialize_message(make_cloud_msg(nanos, points, ground_mask)),
            nanos,
        )
        writer.write(
            "/lidar/legacy_points",
            serialize_message(make_cloud_msg(nanos, points, legacy_mask)),
            nanos,
        )
        writer.write(
            "/ransac/plane", serialize_message(make_plane_marker(nanos, model)), nanos
        )
        writer.write(
            "/ransac/stats", serialize_message(make_stats_marker(nanos, text)), nanos
        )
        writer.write(
            "/semantic_map/squares",
            serialize_message(make_grid_marker(nanos, points[ground_mask])),
            nanos,
        )

    del writer

    r_recall = float(np.mean([s[0] for s in ransac_scores]))
    r_prec = float(np.mean([s[1] for s in ransac_scores]))
    l_recall = float(np.mean([s[0] for s in legacy_scores]))
    l_prec = float(np.mean([s[1] for s in legacy_scores]))

    print(f"wrote {args.frames} frames to {args.out}/")
    print(f"  RANSAC        mean recall {r_recall:.3f}  mean precision {r_prec:.3f}")
    print(
        f"  legacy z<{LEGACY_HEIGHT}   mean recall {l_recall:.3f}  mean precision {l_prec:.3f}"
    )
    print(f"  fallback frames: {fallbacks}/{args.frames}")

    if args.preview:
        # Mid-run frame, where the slope is steepest and the contrast is clearest.
        written = render_preview(args.preview, args.frames // 2, args.frames, args.seed)
        print(f"  preview image:   {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
