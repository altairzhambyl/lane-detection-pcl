#!/usr/bin/env python3
"""
Tests for the shared Marker constructors.

Skipped when visualization_msgs is not importable, so the numpy-only suite
still runs on a bare checkout.
"""

import numpy as np
import pytest

visualization_msgs = pytest.importorskip("visualization_msgs.msg")
builtin_interfaces = pytest.importorskip("builtin_interfaces.msg")

from ransac_seg.markers import (  # noqa: E402
    add_points,
    box_array,
    cube_list,
    delete_all,
    line_strip,
    make_marker,
)

Marker = visualization_msgs.Marker
STAMP = builtin_interfaces.Time(sec=12, nanosec=34)


def test_make_marker_fills_identity_colour_and_scale():
    """Header, namespace, id, colour and scale all land where RViz reads them."""
    marker = make_marker(
        "lidar_front",
        STAMP,
        "road",
        7,
        Marker.CUBE_LIST,
        (0.1, 0.2, 0.3, 0.4),
        (1.0, 2.0, 3.0),
    )

    assert marker.header.frame_id == "lidar_front"
    assert marker.header.stamp == STAMP
    assert (marker.ns, marker.id) == ("road", 7)
    assert marker.type == Marker.CUBE_LIST
    assert marker.action == Marker.ADD
    assert (marker.color.r, marker.color.g) == (pytest.approx(0.1), pytest.approx(0.2))
    assert (marker.color.b, marker.color.a) == (pytest.approx(0.3), pytest.approx(0.4))
    assert marker.scale.x == pytest.approx(1.0)
    assert marker.scale.z == pytest.approx(3.0)
    assert marker.pose.orientation.w == pytest.approx(1.0)


def test_lifetime_omitted_when_zero_and_set_otherwise():
    """A zero lifetime must stay unset, which ROS reads as 'never expire'."""
    forever = make_marker(
        "f", STAMP, "n", 0, Marker.CUBE, (1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0)
    )
    assert (forever.lifetime.sec, forever.lifetime.nanosec) == (0, 0)

    timed = make_marker(
        "f",
        STAMP,
        "n",
        0,
        Marker.CUBE,
        (1.0, 1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        lifetime_sec=1.5,
    )
    assert timed.lifetime.sec == 1
    assert timed.lifetime.nanosec == pytest.approx(5e8, rel=1e-6)


def test_add_points_matches_the_input_rows():
    """Every input row becomes exactly one marker point, in order."""
    pts = np.arange(9, dtype=np.float64).reshape(3, 3)
    marker = add_points(
        make_marker(
            "f", STAMP, "n", 0, Marker.POINTS, (1.0, 1.0, 1.0, 1.0), (0.1, 0.1, 0.1)
        ),
        pts,
    )

    assert len(marker.points) == 3
    assert (marker.points[1].x, marker.points[1].y, marker.points[1].z) == (
        pytest.approx(3.0),
        pytest.approx(4.0),
        pytest.approx(5.0),
    )


def test_cube_list_sizes_tiles_and_counts_points():
    """Tile footprint comes from cell_size, thickness from height."""
    centres = np.zeros((5, 3))
    marker = cube_list(
        "f", STAMP, "road", 3, centres, 0.5, (0.0, 1.0, 0.0, 0.5), height=0.05
    )

    assert marker.type == Marker.CUBE_LIST
    assert len(marker.points) == 5
    assert marker.scale.x == pytest.approx(0.5)
    assert marker.scale.z == pytest.approx(0.05)


def test_cube_list_clears_only_its_own_marker_on_empty_input():
    """
    An empty layer clears itself with a DELETE, never a display-wide DELETEALL.

    ransac_node draws ground, obstacles and inflation on one topic. A DELETEALL
    for the empty obstacle layer wiped the ground layer published microseconds
    earlier in the same frame, because viewers act on DELETEALL globally.
    """
    marker = cube_list("f", STAMP, "road", 3, np.zeros((0, 3)), 0.5, (0, 1, 0, 1))

    assert marker.action == Marker.DELETE
    assert marker.ns == "road"
    assert marker.id == 3


def test_line_strip_needs_two_points():
    """A 0- or 1-point LINE_STRIP is invalid, so it becomes a clear instead."""
    for count in (0, 1):
        marker = line_strip(
            "f", STAMP, "path", 1, np.zeros((count, 3)), (1.0, 0.0, 0.0, 1.0)
        )
        assert marker.action == Marker.DELETE, count
        assert marker.id == 1, count

    good = line_strip(
        "f", STAMP, "path", 1, np.zeros((2, 3)), (1.0, 0.0, 0.0, 1.0), width=0.2
    )
    assert good.action == Marker.ADD
    assert good.type == Marker.LINE_STRIP
    assert len(good.points) == 2
    assert good.scale.x == pytest.approx(0.2)


def test_box_array_assigns_consecutive_ids():
    """Each box gets its own id so none of them overwrite each other."""
    centres = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    array = box_array(
        "f", STAMP, "boxes", 10, centres, (0.5, 0.5, 1.0), (1.0, 1.0, 0.0, 0.8)
    )
    boxes = array.markers[1:]

    assert [m.id for m in boxes] == [10, 11]
    assert all(m.type == Marker.CUBE for m in boxes)
    assert boxes[1].pose.position.x == pytest.approx(4.0)
    assert boxes[0].scale.z == pytest.approx(1.0)


def test_box_array_always_leads_with_a_clear():
    """
    Ids retired when the obstacle count drops must not survive as phantom boxes.

    The default marker lifetime is "never expire", so a busy frame's surplus
    ids stayed on screen forever once the scene emptied out. Only the empty
    case used to clear.
    """
    centres = np.array([[float(i), 0.0, 0.0] for i in range(5)])
    busy = box_array("f", STAMP, "b", 0, centres, (1, 1, 1), (1, 1, 1, 1))
    assert busy.markers[0].action == Marker.DELETEALL
    assert [m.id for m in busy.markers[1:]] == [0, 1, 2, 3, 4]

    quiet = box_array("f", STAMP, "b", 0, centres[:2], (1, 1, 1), (1, 1, 1, 1))
    assert quiet.markers[0].action == Marker.DELETEALL
    assert [m.id for m in quiet.markers[1:]] == [0, 1]


def test_box_array_accepts_per_box_extents_and_clears_when_empty():
    """Sizes may vary per box; an empty array still clears the namespace."""
    centres = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    extents = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    array = box_array("f", STAMP, "b", 0, centres, extents, (1.0, 1.0, 1.0, 1.0))
    assert array.markers[2].scale.x == pytest.approx(2.0)

    empty = box_array("f", STAMP, "b", 0, np.zeros((0, 3)), (1, 1, 1), (1, 1, 1, 1))
    assert len(empty.markers) == 1
    assert empty.markers[0].action == Marker.DELETEALL


def test_delete_all_targets_one_namespace():
    """The clear marker carries the namespace it is meant to wipe."""
    marker = delete_all("lidar_front", STAMP, "path")

    assert marker.action == Marker.DELETEALL
    assert marker.ns == "path"
    assert marker.header.frame_id == "lidar_front"
