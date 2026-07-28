#!/usr/bin/env python3
"""
End-to-end tests driving every ROS 2 node in the package with real PointCloud2s.

Skipped automatically when ROS 2 is not on the path, so the numpy-only test
suite still runs on a bare checkout.

The three cases beyond "it does not raise" are the ones that caught real bugs:
markers must sit on the fitted plane rather than at z=0.0, a frame with nothing
to draw must clear the previous one instead of publishing an empty ADD (which
viewers read as "no change"), and two consecutive frames must survive the
temporal plane filter, which only engages from the second frame onwards.
"""

import math

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
point_cloud2 = pytest.importorskip("sensor_msgs_py.point_cloud2")
std_msgs = pytest.importorskip("std_msgs.msg")
visualization_msgs = pytest.importorskip("visualization_msgs.msg")
qos = pytest.importorskip("rclpy.qos")
sensor_msgs = pytest.importorskip("sensor_msgs.msg")

from ransac_seg.astar import AStarPlanner  # noqa: E402
from ransac_seg.bfs import BFSPlanner  # noqa: E402
from ransac_seg.curve import LaneBoundaryNode  # noqa: E402
from ransac_seg.dfs import DFSBaseline  # noqa: E402
from ransac_seg.dijkstra import DijkstraPlanner  # noqa: E402
from ransac_seg.grid import plane_z  # noqa: E402
from ransac_seg.ransac_node import GroundSegmentationNode  # noqa: E402

Marker = visualization_msgs.Marker
PointCloud2 = sensor_msgs.PointCloud2
ReliabilityPolicy = qos.ReliabilityPolicy

ALL_NODES = [
    GroundSegmentationNode,
    LaneBoundaryNode,
    AStarPlanner,
    BFSPlanner,
    DFSBaseline,
    DijkstraPlanner,
]

# Nodes that rasterise a costmap and therefore have nothing to draw once the
# drivable surface is gone. LaneBoundaryNode is excluded because it fits the
# ground returns directly and never consults the grid.
COSTMAP_NODES = [n for n in ALL_NODES if n is not LaneBoundaryNode]

# Marker types whose geometry lives in ``points``; the rest (CUBE, TEXT) carry
# it in ``pose``, so an empty ``points`` list is correct for them.
POINT_LIST_TYPES = frozenset(
    {
        Marker.POINTS,
        Marker.LINE_STRIP,
        Marker.LINE_LIST,
        Marker.CUBE_LIST,
        Marker.SPHERE_LIST,
        Marker.TRIANGLE_LIST,
    }
)

TILT_DEG = 10.0
SENSOR_HEIGHT = 1.7
HALF_WIDTH = 3.5


@pytest.fixture(scope="module", autouse=True)
def ros_context():
    """Initialise and tear down rclpy once for the module."""
    rclpy.init()
    yield
    rclpy.shutdown()


def _to_msg(points, frame_id="lidar_front"):
    """Wrap an (N, 3) array in a PointCloud2 the nodes can decode."""
    header = std_msgs.Header()
    header.frame_id = frame_id
    return point_cloud2.create_cloud_xyz32(
        header, np.asarray(points, dtype=np.float32).tolist()
    )


def _road_surface(rng, n=9000):
    """Sample a road corridor pitched TILT_DEG upwards, HALF_WIDTH either side."""
    xs = rng.uniform(0.6, 40.0, n)
    ys = rng.uniform(-HALF_WIDTH, HALF_WIDTH, n)
    zs = -SENSOR_HEIGHT + xs * math.tan(math.radians(TILT_DEG))
    return np.column_stack([xs, ys, zs + rng.normal(0.0, 0.01, n)])


def make_road_cloud(seed=19):
    """
    Build a sloped road corridor flanked by guardrails.

    The corridor is deliberately road-width rather than an open field: the lane
    fit rejects anything wider than its ``max_width`` parameter, so a 30 m
    clearing would (correctly) publish no boundaries at all and test nothing.
    """
    rng = np.random.default_rng(seed)
    ground = _road_surface(rng)

    rails = []
    for side in (-1.0, 1.0):
        n = 2500
        rx = rng.uniform(0.6, 40.0, n)
        ry = side * rng.uniform(HALF_WIDTH + 0.1, HALF_WIDTH + 0.6, n)
        rz = -SENSOR_HEIGHT + rx * math.tan(math.radians(TILT_DEG))
        rails.append(np.column_stack([rx, ry, rz + rng.uniform(0.4, 1.2, n)]))

    return _to_msg(np.vstack([ground] + rails))


def make_blocked_cloud(seed=23):
    """
    Build the same road with an obstacle sheet standing over every cell of it.

    Occupied beats free in the costmap, so no traversable cell survives and
    every costmap consumer has to clear rather than draw.
    """
    rng = np.random.default_rng(seed)
    ground = _road_surface(rng)

    # A lattice finer than the 0.5 m cell, so every cell clears the
    # min_obstacle_points threshold instead of leaving gaps to plan through.
    gx, gy = np.meshgrid(np.arange(0.6, 40.0, 0.2), np.arange(-4.0, 4.0, 0.2))
    gx, gy = gx.ravel(), gy.ravel()
    gz = -SENSOR_HEIGHT + gx * math.tan(math.radians(TILT_DEG)) + 1.0
    return _to_msg(np.vstack([ground, np.column_stack([gx, gy, gz])]))


def make_empty_cloud():
    """Build a cloud with no returns at all."""
    return _to_msg(np.zeros((0, 3)))


def _flatten(published):
    """Return every Marker in a list of Marker and MarkerArray messages."""
    markers = []
    for msg in published:
        markers.extend(getattr(msg, "markers", None) or [msg])
    return markers


def drive(node_cls, msgs):
    """
    Run a node over a sequence of clouds, capturing markers and frames.

    Publishers are stubbed rather than spun so the test observes exactly what
    the node emitted, without needing a live graph or a subscriber.
    """
    node = node_cls()
    published, frames = [], []

    for attr in dir(node):
        pub = getattr(node, attr, None)
        if hasattr(pub, "publish") and hasattr(pub, "topic_name"):
            pub.publish = lambda msg, sink=published: sink.append(msg)

    handler = node.on_frame

    def capture(frame):
        frames.append(frame)
        return handler(frame)

    node.on_frame = capture
    try:
        for msg in msgs:
            node._on_cloud(msg)
    finally:
        node.destroy_node()
    return _flatten(published), frames


def assert_no_empty_add(markers, node_cls):
    """No node may publish an ADD marker whose point list is empty."""
    for marker in markers:
        if marker.action == Marker.ADD and marker.type in POINT_LIST_TYPES:
            assert marker.points, (
                f"{node_cls.__name__} published an empty ADD in ns "
                f"'{marker.ns}'; viewers keep the previous frame on screen"
            )


@pytest.mark.parametrize("node_cls", ALL_NODES)
def test_node_publishes_geometry_for_a_real_cloud(node_cls):
    """Each node consumes a sloped road cloud and draws something for it."""
    markers, frames = drive(node_cls, [make_road_cloud()])

    assert len(frames) == 1, f"{node_cls.__name__} did not process the frame"
    assert markers, f"{node_cls.__name__} published nothing"
    assert_no_empty_add(markers, node_cls)
    assert any(m.action == Marker.ADD and m.points for m in markers), (
        f"{node_cls.__name__} drew no geometry for a good frame"
    )


@pytest.mark.parametrize("node_cls", ALL_NODES)
def test_marker_geometry_sits_on_the_fitted_plane(node_cls):
    """
    Published geometry follows the fitted plane, not the old hardcoded z=0.0.

    At 10 degrees the surface is 7 m above z=0 by the far end of the sweep, so
    a regression here is not a rounding error — it puts every marker in the air.
    """
    markers, frames = drive(node_cls, [make_road_cloud()])
    plane = frames[0].plane
    assert plane is not None, "the fixture cloud must produce a plane fit"

    checked = 0
    for marker in markers:
        if marker.action != Marker.ADD or marker.type not in POINT_LIST_TYPES:
            continue
        xyz = np.array([[p.x, p.y, p.z] for p in marker.points])
        expected = plane_z(plane, xyz[:, 0], xyz[:, 1])
        assert np.max(np.abs(xyz[:, 2] - expected)) < 0.1, (
            f"{node_cls.__name__} ns '{marker.ns}' is off the fitted plane"
        )
        checked += xyz.shape[0]

    assert checked > 0, f"{node_cls.__name__} published no geometry to check"


@pytest.mark.parametrize("node_cls", COSTMAP_NODES)
def test_blocked_ground_clears_instead_of_publishing_empty(node_cls):
    """With the road walled off, costmap consumers clear their namespaces."""
    markers, _ = drive(node_cls, [make_blocked_cloud()])

    assert_no_empty_add(markers, node_cls)
    assert any(m.action == Marker.DELETEALL for m in markers), (
        f"{node_cls.__name__} left stale geometry on screen for a blocked frame"
    )


@pytest.mark.parametrize("node_cls", ALL_NODES)
def test_empty_cloud_is_handled(node_cls):
    """A cloud with no returns must not raise and must not reach on_frame."""
    markers, frames = drive(node_cls, [make_empty_cloud()])

    assert frames == [], f"{node_cls.__name__} processed a frame with no points"
    assert markers == [], f"{node_cls.__name__} published for an empty cloud"


@pytest.mark.parametrize("node_cls", ALL_NODES)
def test_two_frames_exercise_the_plane_filter(node_cls):
    """
    A second frame blends against the held plane instead of replacing it.

    The filter is a no-op on frame one — it takes the first estimate verbatim —
    so only a second frame reaches the gating and blending code at all.
    """
    markers, frames = drive(node_cls, [make_road_cloud(19), make_road_cloud(37)])

    assert len(frames) == 2, f"{node_cls.__name__} dropped a frame"
    assert frames[1].plane is not None
    assert abs(frames[1].plane.tilt_deg - TILT_DEG) < 1.0
    assert_no_empty_add(markers, node_cls)


@pytest.mark.parametrize("node_cls", ALL_NODES)
def test_cloud_subscription_accepts_a_best_effort_publisher(node_cls):
    """
    Sensor drivers publish BEST_EFFORT, and a RELIABLE subscription never matches.

    Created from a bare depth the subscription inherited rclpy's RELIABLE
    default, so pointing the stack at a real LiDAR driver (or at `ros2 bag
    play` of data recorded from one) produced a silently dead pipeline: DDS
    refuses the match and only a QoS-incompatibility line in the log says so.
    """
    node = node_cls()
    try:
        subs = [s for s in node.subscriptions if s.msg_type is PointCloud2]
        assert subs, f"{node_cls.__name__} subscribes to no cloud"
        for sub in subs:
            assert sub.qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT
    finally:
        node.destroy_node()


@pytest.mark.parametrize("node_cls", ALL_NODES)
def test_markers_carry_the_cloud_stamp_not_the_wall_clock(node_cls):
    """
    Every marker describes the geometry of one sweep, so it wears that sweep's time.

    Stamping with the node's own clock breaks the standard bag-replay setup:
    `replay_bag.py` republishes recorded clouds verbatim, so under
    ``use_sim_time`` the cloud carries the recorded time and the markers
    carried the wall clock (or 0), leaving the viewer to look up a TF at two
    unrelated times.
    """
    msg = make_road_cloud()
    msg.header.stamp.sec = 1000
    msg.header.stamp.nanosec = 250
    markers, frames = drive(node_cls, [msg])

    assert frames[0].stamp.sec == 1000
    assert frames[0].stamp.nanosec == 250
    for marker in markers:
        assert marker.header.stamp.sec == 1000, marker.ns
        assert marker.header.stamp.nanosec == 250, marker.ns


@pytest.mark.parametrize("node_cls", ALL_NODES)
def test_runtime_parameter_sets_take_effect_or_are_refused(node_cls):
    """
    A successful `ros2 param set` must change behaviour, not just the store.

    Every parameter was snapshotted in ``__init__`` with no on-set callback, so
    an operator retuning range or inflation on a live vehicle got a success
    acknowledgement and identical output. Live parameters now apply; the ones
    consumed once at start-up are refused with a reason.
    """
    node = node_cls()
    try:
        ok = node.set_parameters(
            [
                rclpy.parameter.Parameter("resolution", value=2.0),
                rclpy.parameter.Parameter("max_range", value=5.0),
            ]
        )
        assert all(r.successful for r in ok)
        assert node.params.resolution == pytest.approx(2.0)
        assert node.params.max_range == pytest.approx(5.0)

        blend = node.set_parameters(
            [rclpy.parameter.Parameter("plane_filter_alpha", value=0.9)]
        )
        assert all(r.successful for r in blend)
        assert node.plane_filter.alpha == pytest.approx(0.9)

        refused = node.set_parameters(
            [rclpy.parameter.Parameter("input_topic", value="/other/points")]
        )
        assert not refused[0].successful
        assert "restart" in refused[0].reason
        assert node.params.input_topic != "/other/points"
    finally:
        node.destroy_node()


def test_runtime_parameter_change_reaches_the_next_frame():
    """The applied value is the one the next callback actually rasterises with."""
    node = GroundSegmentationNode()
    try:
        node._on_cloud(make_road_cloud())
        node.set_parameters([rclpy.parameter.Parameter("resolution", value=2.0)])

        seen = []
        node.on_frame = seen.append
        node._on_cloud(make_road_cloud(37))
        assert seen[0].grid.resolution == pytest.approx(2.0)
    finally:
        node.destroy_node()


def test_an_empty_layer_does_not_wipe_the_layers_beside_it():
    """
    ransac_node draws three namespaces on one topic; clearing one must spare the rest.

    ``cube_list`` used to return a DELETEALL for an empty layer, and viewers
    act on DELETEALL globally, so an obstacle-free frame blanked the ground
    tiles published microseconds earlier in the same frame.
    """
    rng = np.random.default_rng(5)
    clear_road = _to_msg(_road_surface(rng))
    published, _ = drive(GroundSegmentationNode, [clear_road])

    squares = [m for m in published if m.ns in {"ground", "obstacles", "inflation"}]
    ground = [m for m in squares if m.ns == "ground"]
    assert ground and ground[0].action == Marker.ADD and ground[0].points
    assert not any(m.action == Marker.DELETEALL for m in squares), (
        "a DELETEALL on the shared squares topic erases the ground layer too"
    )
    assert any(m.action == Marker.DELETE and m.ns == "obstacles" for m in squares)
