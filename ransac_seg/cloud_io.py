#!/usr/bin/env python3
"""
Vectorised PointCloud2 <-> numpy conversion shared by every node.

Each node used to carry its own copy of a per-point Python loop over
``point_cloud2.read_points``; on a full sweep that loop dominates the
callback, while ``read_points_numpy`` hands back the same data as one
strided view. The publish direction had the same problem in reverse
(``struct.pack`` per point), so packing here goes through a single
``astype(np.float32).tobytes()``.
"""

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2

XYZ_FIELDS = ("x", "y", "z")

_POINT_STEP = 12  # three float32 components, tightly packed


def pointcloud2_to_xyz(msg, skip_nans=True):
    """
    Return an ``(N, 3)`` float64 C-contiguous array of xyz from a PointCloud2.

    Organised clouds are flattened. ``skip_nans`` drops rows holding any
    non-finite component, which is what the RANSAC fit expects.
    """
    if msg.width == 0 or msg.height == 0:
        return np.zeros((0, 3), dtype=np.float64)

    raw = point_cloud2.read_points_numpy(
        msg, field_names=XYZ_FIELDS, skip_nans=skip_nans
    )
    if raw.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    # float64 up front: the plane maths is done in double, and converting once
    # here is cheaper than letting numpy upcast inside every RANSAC iteration.
    return np.ascontiguousarray(raw.reshape(-1, 3), dtype=np.float64)


def xyz_to_pointcloud2(points, frame_id, stamp):
    """
    Pack an ``(N, 3)`` array into a PointCloud2 with float32 xyz fields.

    ``stamp`` is a ``builtin_interfaces.msg.Time``. The result is an
    unorganised (height 1), dense, little-endian cloud.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)

    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = int(pts.shape[0])
    msg.fields = [
        PointField(name=name, offset=4 * i, datatype=PointField.FLOAT32, count=1)
        for i, name in enumerate(XYZ_FIELDS)
    ]
    msg.is_bigendian = False
    msg.point_step = _POINT_STEP
    msg.row_step = _POINT_STEP * msg.width
    msg.is_dense = True
    msg.data = np.ascontiguousarray(pts, dtype=np.float32).tobytes()
    return msg
