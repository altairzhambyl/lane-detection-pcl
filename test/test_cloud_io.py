#!/usr/bin/env python3
"""
Tests for the vectorised PointCloud2 codec.

Skipped when ROS 2 message packages are not importable, so the numpy-only
suite still runs on a bare checkout.
"""

import numpy as np
import pytest

point_cloud2 = pytest.importorskip("sensor_msgs_py.point_cloud2")
std_msgs = pytest.importorskip("std_msgs.msg")
builtin_interfaces = pytest.importorskip("builtin_interfaces.msg")

from ransac_seg.cloud_io import pointcloud2_to_xyz, xyz_to_pointcloud2  # noqa: E402


def make_cloud(points, frame_id="lidar_front"):
    """Build a PointCloud2 from an (N, 3) array via the stock ROS helper."""
    header = std_msgs.Header()
    header.frame_id = frame_id
    return point_cloud2.create_cloud_xyz32(
        header, np.asarray(points, dtype=np.float32).reshape(-1, 3).tolist()
    )


def test_round_trip_is_exact_to_float32():
    """Packing then decoding returns the same points at float32 precision."""
    rng = np.random.default_rng(7)
    pts = rng.uniform(-30.0, 30.0, size=(500, 3))

    msg = xyz_to_pointcloud2(pts, "lidar_front", builtin_interfaces.Time())
    back = pointcloud2_to_xyz(msg)

    assert back.shape == (500, 3)
    assert back.dtype == np.float64
    np.testing.assert_allclose(back, pts.astype(np.float32), rtol=0.0, atol=0.0)


def test_pack_sets_the_layout_fields():
    """The packed cloud is unorganised, dense and 12 bytes per point."""
    msg = xyz_to_pointcloud2(np.zeros((4, 3)), "base_link", builtin_interfaces.Time())

    assert (msg.height, msg.width) == (1, 4)
    assert (msg.point_step, msg.row_step) == (12, 48)
    assert msg.is_dense and not msg.is_bigendian
    assert [f.name for f in msg.fields] == ["x", "y", "z"]
    assert len(msg.data) == 48
    assert msg.header.frame_id == "base_link"


def test_empty_cloud_gives_zero_by_three():
    """A zero-width cloud decodes to (0, 3), not to an unusable empty array."""
    for msg in (
        make_cloud(np.zeros((0, 3))),
        xyz_to_pointcloud2(np.zeros((0, 3)), "lidar_front", builtin_interfaces.Time()),
    ):
        out = pointcloud2_to_xyz(msg)
        assert out.shape == (0, 3)


def test_single_point_round_trips():
    """One point is not mistaken for an empty or a flat cloud."""
    msg = xyz_to_pointcloud2([[1.5, -2.25, 0.75]], "l", builtin_interfaces.Time())
    np.testing.assert_allclose(pointcloud2_to_xyz(msg), [[1.5, -2.25, 0.75]])


def test_nan_rows_dropped_only_when_requested():
    """skip_nans drops non-finite rows; without it they survive the decode."""
    pts = np.array(
        [[1.0, 2.0, 3.0], [np.nan, 0.0, 0.0], [4.0, 5.0, 6.0]], dtype=np.float32
    )
    msg = make_cloud(pts)

    kept = pointcloud2_to_xyz(msg, skip_nans=True)
    assert kept.shape == (2, 3)
    np.testing.assert_allclose(kept, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    raw = pointcloud2_to_xyz(msg, skip_nans=False)
    assert raw.shape == (3, 3)
    assert np.isnan(raw[1, 0])


def test_organized_cloud_is_flattened():
    """An organised (height > 1) cloud decodes to a flat (N, 3) array."""
    msg = make_cloud(np.arange(24, dtype=np.float32).reshape(8, 3))
    msg.height = 2
    msg.width = 4
    msg.row_step = msg.point_step * 4

    out = pointcloud2_to_xyz(msg)
    assert out.shape == (8, 3)
    np.testing.assert_allclose(out, np.arange(24).reshape(8, 3))
