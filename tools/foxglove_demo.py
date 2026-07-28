#!/usr/bin/env python3
"""
Generate a Foxglove-ready MCAP recording of the whole perception pipeline.

Simulates a drive along a road corridor whose slope changes under the sensor
and runs the real node code on every sweep — `segment_ground`, `build_costmap`,
`cluster_obstacles`, `fit_lane` and `astar` — so what the recording shows is
what the nodes compute, not a re-implementation. The legacy fixed-height filter
is recorded alongside it so the difference is visible on the timeline rather
than asserted in a README.

Usage:
    python3 tools/foxglove_demo.py [--frames 60] [--out docs/ransac_demo]

Then open the resulting `.mcap` in Foxglove (app.foxglove.dev or the desktop
app) and add a 3D panel. The marker topics and namespaces are the ones the live
nodes publish, so one Foxglove layout works for the bag and for a running graph.

    /lidar/points             full cloud, green=road / red=obstacle by RANSAC
    /lidar/legacy_points      same cloud coloured by the old z < 0.2 filter
    /ransac/plane             the fitted plane, drawn as a translucent quad
    /ransac/stats             per-frame tilt, inliers, lane width, path cost
    /semantic_map/squares     costmap cells: ns ground / obstacles / inflation
    /semantic_map/obstacles   one box per clustered obstacle
    /semantic_map/road_left   fitted left road edge
    /semantic_map/road_right  fitted right road edge
    /semantic_map/road_center the centreline the edges were measured against
    /planning/astar_path      the least-cost route across the costmap
"""

import argparse
import array
import math
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rosbag2_py  # noqa: E402
from builtin_interfaces.msg import Time  # noqa: E402
from geometry_msgs.msg import Point  # noqa: E402
from ransac_seg import markers  # noqa: E402
from ransac_seg.cloud_io import xyz_to_pointcloud2  # noqa: E402
from ransac_seg.cluster import cluster_obstacles  # noqa: E402
from ransac_seg.grid import (  # noqa: E402
    FREE,
    INFLATED,
    OCCUPIED,
    build_costmap,
    plane_z,
    snap_to_traversable,
)
from ransac_seg.ground_plane import segment_ground  # noqa: E402
from ransac_seg.lane import fit_lane  # noqa: E402
from ransac_seg.planners import (  # noqa: E402
    astar,
    path_cost,
    path_length_m,
    select_goal,
)
from rclpy.serialization import serialize_message  # noqa: E402
from sensor_msgs.msg import PointField  # noqa: E402
from std_msgs.msg import ColorRGBA, Header  # noqa: E402
from visualization_msgs.msg import Marker  # noqa: E402

FRAME_ID = "lidar_front"
SENSOR_HEIGHT = 1.7
LEGACY_HEIGHT = 0.2
GRID_SIZE = 0.5
HALF_WIDTH = 4.5

# The planner tunables the nodes default to, repeated here so the bag shows the
# same route the live graph would plan.
GOAL_XY = (25.0, 0.0)
GOAL_RADIUS = 5.0
START_RADIUS = 2.0

ROAD_COLOUR = (60, 220, 90)
OBSTACLE_COLOUR = (230, 70, 70)
_POINT_STEP = 16

CELL_LAYERS = (
    ("ground", 0, FREE, (0.0, 1.0, 0.0, 0.5)),
    ("obstacles", 1, OCCUPIED, (1.0, 0.0, 0.0, 0.5)),
    ("inflation", 2, INFLATED, (1.0, 0.55, 0.0, 0.25)),
)


def _stamp(nanos):
    """Build a builtin_interfaces/Time from a nanosecond count."""
    return Time(sec=int(nanos // 10**9), nanosec=int(nanos % 10**9))


def _header(nanos):
    """Build a std_msgs/Header stamped at ``nanos`` in the sensor frame."""
    header = Header()
    header.frame_id = FRAME_ID
    header.stamp = _stamp(nanos)
    return header


_EMPTY_XYZ = np.zeros((0, 3), dtype=np.float64)


def _pack_xyz_word(points, fourth, fourth_dtype):
    """
    Lay out xyz plus one trailing 32-bit word as a 16-byte-per-point buffer.

    A structured dtype does in one numpy pass what the old ``struct.pack``
    loop did per point, and keeps the trailing word's bit pattern intact
    (packed rgb must not be reinterpreted as a float). The result is an
    ``array.array('B')`` because that is the only branch of the generated
    PointCloud2 ``data`` setter that does not re-validate every byte in
    Python — assigning raw ``bytes`` there costs more than the packing did.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("w", fourth_dtype)])
    packed = np.empty(pts.shape[0], dtype=dtype)
    packed["x"] = pts[:, 0]
    packed["y"] = pts[:, 1]
    packed["z"] = pts[:, 2]
    packed["w"] = fourth
    return array.array("B", packed.tobytes())


def make_scene(frame_index, n_frames, rng):
    """
    Build one LiDAR sweep along a sloped road corridor with obstacles.

    Returns ``(points, is_road)`` where ``is_road`` is the ground truth used
    only for scoring — the estimator never sees it.

    The drivable surface is a road-width corridor rather than the open field
    this scene used to be, because the lane fit rejects anything wider than its
    ``max_width``: a 28 m clearing has no road edges to find, so the boundary
    half of the pipeline had nothing to show.
    """
    # Slope sweeps 0 -> 12 degrees and back, as if cresting a rise.
    phase = frame_index / max(n_frames - 1, 1)
    tilt = math.radians(12.0 * math.sin(math.pi * phase))

    n_road = 9000
    xs = rng.uniform(1.0, 45.0, n_road)
    ys = rng.uniform(-HALF_WIDTH, HALF_WIDTH, n_road)
    zs = -SENSOR_HEIGHT + xs * math.tan(tilt) + rng.normal(0.0, 0.012, n_road)
    road = np.column_stack([xs, ys, zs])

    # Obstacles: two vehicle-sized blocks parked in the corridor, and a
    # guardrail down each side to give the boundary fit an edge to land on.
    blocks = []
    for cx, cy in ((16.0, -3.0), (30.0, 1.5)):
        n = 1200
        bx = rng.uniform(cx, cx + 4.5, n)
        by = rng.uniform(cy, cy + 1.9, n)
        bz = -SENSOR_HEIGHT + bx * math.tan(tilt) + rng.uniform(0.1, 1.6, n)
        blocks.append(np.column_stack([bx, by, bz]))

    for side in (-1.0, 1.0):
        n_rail = 1800
        rx = rng.uniform(1.0, 45.0, n_rail)
        ry = side * rng.uniform(HALF_WIDTH + 0.1, HALF_WIDTH + 0.5, n_rail)
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
    colours = np.where(
        road_mask,
        (ROAD_COLOUR[0] << 16) | (ROAD_COLOUR[1] << 8) | ROAD_COLOUR[2],
        (OBSTACLE_COLOUR[0] << 16) | (OBSTACLE_COLOUR[1] << 8) | OBSTACLE_COLOUR[2],
    ).astype(np.uint32)

    # The shared packer owns the header, flags and xyz field layout so this bag
    # and the live nodes cannot drift apart. It is handed an empty array rather
    # than the sweep because this cloud carries a 4th (rgb) word, so its 12-byte
    # buffer would only be built to be thrown away.
    msg = xyz_to_pointcloud2(_EMPTY_XYZ, FRAME_ID, _stamp(nanos))
    msg.fields = msg.fields + [
        PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1)
    ]
    msg.width = int(points.shape[0])
    msg.point_step = _POINT_STEP
    msg.row_step = _POINT_STEP * msg.width
    msg.data = _pack_xyz_word(points, colours, "<u4")
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


def run_pipeline(points):
    """
    Run the node pipeline on one sweep and return everything it produced.

    Calls the same functions the ROS nodes call, in the same order, so the
    recording cannot drift from the running system the way a bespoke
    visualisation would.
    """
    ground_mask, model = segment_ground(points)
    ground, obstacles = points[ground_mask], points[~ground_mask]

    grid = build_costmap(ground, obstacles, resolution=GRID_SIZE, plane=model)
    clusters = cluster_obstacles(obstacles)
    lane = fit_lane(ground)

    start = snap_to_traversable(grid, 0.0, 0.0, START_RADIUS)
    goal = select_goal(grid, GOAL_XY, search_radius=GOAL_RADIUS)
    path = []
    if start is not None and goal is not None:
        path, _ = astar(grid, start, goal)

    return {
        "ground_mask": ground_mask,
        "model": model,
        "grid": grid,
        "clusters": clusters,
        "lane": lane,
        "path": path,
    }


def make_cell_markers(nanos, grid):
    """Build one CUBE_LIST per costmap state, all on the fitted plane."""
    stamp = _stamp(nanos)
    return [
        markers.cube_list(
            FRAME_ID,
            stamp,
            namespace,
            marker_id,
            grid.cell_to_world_xyz(grid.cells_with_state(state)),
            grid.resolution,
            colour,
        )
        for namespace, marker_id, state, colour in CELL_LAYERS
    ]


def make_cluster_markers(nanos, clusters):
    """Build one box per clustered obstacle."""
    return markers.box_array(
        FRAME_ID,
        _stamp(nanos),
        "obstacle_boxes",
        0,
        [c.centroid for c in clusters],
        [c.extent for c in clusters],
        (1.0, 0.3, 0.0, 0.6),
    )


def make_lane_markers(nanos, lane, model):
    """Build the left/right/centre boundary strips lifted onto the plane."""
    stamp = _stamp(nanos)
    strips = (
        ("road_left", lane.left, (0.0, 0.4, 1.0, 0.9)),
        ("road_right", lane.right, (1.0, 0.2, 0.0, 0.9)),
        ("road_center", lane.centerline, (1.0, 0.9, 0.0, 0.7)),
    )
    out = []
    for namespace, xy, colour in strips:
        if not lane.valid:
            out.append(markers.delete_all(FRAME_ID, stamp, namespace))
            continue
        pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        z = np.broadcast_to(plane_z(model, pts[:, 0], pts[:, 1]), pts[:, 0].shape)
        out.append(
            markers.line_strip(
                FRAME_ID, stamp, namespace, 0, np.column_stack([pts, z]), colour
            )
        )
    return out


def make_path_marker(nanos, grid, path):
    """Build the planned route as a LINE_STRIP; empty routes clear the old one."""
    cells = np.asarray(path, dtype=np.int64).reshape(-1, 2)
    return markers.line_strip(
        FRAME_ID,
        _stamp(nanos),
        "astar_path",
        0,
        grid.cell_to_world_xyz(cells),
        (1.0, 0.25, 0.2, 0.95),
        width=0.15,
    )


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
    "/semantic_map/obstacles": "visualization_msgs/msg/MarkerArray",
    "/semantic_map/road_left": "visualization_msgs/msg/Marker",
    "/semantic_map/road_right": "visualization_msgs/msg/Marker",
    "/semantic_map/road_center": "visualization_msgs/msg/Marker",
    "/planning/astar_path": "visualization_msgs/msg/Marker",
}

LANE_TOPICS = (
    "/semantic_map/road_left",
    "/semantic_map/road_right",
    "/semantic_map/road_center",
)


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
    widths, costs, cluster_counts, planned = [], [], [], 0

    for i in range(args.frames):
        nanos = (i + 1) * period_ns
        points, truth = make_scene(i, args.frames, rng)

        result = run_pipeline(points)
        ground_mask, model = result["ground_mask"], result["model"]
        grid, lane, path = result["grid"], result["lane"], result["path"]
        legacy_mask = points[:, 2] < LEGACY_HEIGHT

        if model is None:
            fallbacks += 1
        ransac_scores.append(score(ground_mask, truth))
        legacy_scores.append(score(legacy_mask, truth))
        cluster_counts.append(len(result["clusters"]))
        if lane.valid:
            widths.append(float(np.median(lane.width)))
        if path:
            planned += 1
            costs.append(float(path_cost(grid, path)))

        tilt = model.tilt_deg if model else float("nan")
        inliers = model.inlier_ratio if model else float("nan")
        width = float(np.median(lane.width)) if lane.valid else float("nan")
        text = (
            f"frame {i + 1}/{args.frames}\n"
            f"RANSAC  tilt {tilt:5.1f} deg  inliers {inliers:5.1%}  "
            f"recall {ransac_scores[-1][0]:.3f}  precision {ransac_scores[-1][1]:.3f}\n"
            f"legacy z<{LEGACY_HEIGHT}      "
            f"recall {legacy_scores[-1][0]:.3f}  precision {legacy_scores[-1][1]:.3f}\n"
            f"lane width {width:.2f} m   obstacles {len(result['clusters'])}   "
            f"path {len(path)} cells "
            f"{path_length_m(grid, path) if path else 0.0:.1f} m"
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
        for marker in make_cell_markers(nanos, grid):
            writer.write("/semantic_map/squares", serialize_message(marker), nanos)
        writer.write(
            "/semantic_map/obstacles",
            serialize_message(make_cluster_markers(nanos, result["clusters"])),
            nanos,
        )
        for topic, marker in zip(LANE_TOPICS, make_lane_markers(nanos, lane, model)):
            writer.write(topic, serialize_message(marker), nanos)
        writer.write(
            "/planning/astar_path",
            serialize_message(make_path_marker(nanos, grid, path)),
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
    print(
        f"  lane fitted:     {len(widths)}/{args.frames} frames, "
        f"median width {np.median(widths) if widths else float('nan'):.2f} m"
    )
    print(
        f"  obstacles:       {np.mean(cluster_counts):.1f} clusters/frame "
        f"(min {min(cluster_counts)}, max {max(cluster_counts)})"
    )
    print(
        f"  path planned:    {planned}/{args.frames} frames, "
        f"median cost {np.median(costs) if costs else float('nan'):.1f}"
    )

    if args.preview:
        # Mid-run frame, where the slope is steepest and the contrast is clearest.
        written = render_preview(args.preview, args.frames // 2, args.frames, args.seed)
        print(f"  preview image:   {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
