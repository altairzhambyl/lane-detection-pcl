#!/usr/bin/env python3
"""
Marker construction helpers shared by every visualising node.

Each node hand-rolled its own Marker scaffolding — header, namespace, id,
colour, scale — which drifted apart field by field. Beyond the duplication
that produced a real display bug: a frame with nothing to draw published an
ADD marker holding zero points, which RViz and Foxglove treat as "no change",
so the previous frame's path stayed on screen indefinitely.

Clearing is therefore explicit here, and it is deliberately *narrow*. A
fixed-id constructor clears with a DELETE of its own ``(ns, id)`` rather than a
DELETEALL, because consumers implement DELETEALL as "wipe this display", not
"wipe this namespace": ``ransac_node`` draws three namespaces on one topic, and
a DELETEALL for the empty obstacle layer erased the ground layer published
microseconds earlier in the same frame. Only ``box_array``, which owns a whole
topic and publishes a variable number of ids, still leads with a DELETEALL —
that is how ids retired when the obstacle count drops get taken off the screen.
"""

import numpy as np
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


def make_marker(
    frame_id,
    stamp,
    ns,
    marker_id,
    marker_type,
    color,
    scale,
    lifetime_sec=0.0,
    action=Marker.ADD,
):
    """
    Build a Marker with header, identity, colour and scale filled in.

    ``color`` is ``(r, g, b, a)`` and ``scale`` is ``(x, y, z)``. A
    ``lifetime_sec`` of 0.0 means "forever" and leaves the field untouched.
    """
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = ns
    marker.id = int(marker_id)
    marker.type = marker_type
    marker.action = action
    marker.scale.x, marker.scale.y, marker.scale.z = (float(v) for v in scale)
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = (
        float(v) for v in color
    )
    marker.pose.orientation.w = 1.0
    if lifetime_sec > 0.0:
        whole = int(lifetime_sec)
        marker.lifetime = Duration(sec=whole, nanosec=int((lifetime_sec - whole) * 1e9))
    return marker


def add_points(marker, points_xyz):
    """Append an ``(M, 3)`` array of points to a marker and return it."""
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    marker.points.extend(
        Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts
    )
    return marker


def delete_all(frame_id, stamp, ns):
    """
    Build a marker that clears the consumer's whole display.

    Despite the namespace field, RViz2 and Foxglove act on DELETEALL globally,
    so only publish this on a topic that carries a single namespace. To clear
    one marker, use ``delete`` instead.
    """
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = ns
    marker.action = Marker.DELETEALL
    return marker


def delete(frame_id, stamp, ns, marker_id):
    """Build a marker that clears exactly one previously published ``(ns, id)``."""
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = ns
    marker.id = int(marker_id)
    marker.action = Marker.DELETE
    return marker


def cube_list(
    frame_id,
    stamp,
    ns,
    marker_id,
    centres_xyz,
    cell_size,
    color,
    height=0.05,
    lifetime_sec=0.0,
):
    """
    Build a CUBE_LIST of equally sized tiles centred on ``centres_xyz``.

    Returns a DELETE of this ``(ns, marker_id)`` when there are no centres, so
    an empty frame erases its own layer — and only its own layer — instead of
    leaving it on screen.
    """
    centres = np.asarray(centres_xyz, dtype=np.float64).reshape(-1, 3)
    if centres.shape[0] == 0:
        return delete(frame_id, stamp, ns, marker_id)

    marker = make_marker(
        frame_id,
        stamp,
        ns,
        marker_id,
        Marker.CUBE_LIST,
        color,
        (cell_size, cell_size, height),
        lifetime_sec=lifetime_sec,
    )
    return add_points(marker, centres)


def line_strip(
    frame_id, stamp, ns, marker_id, points_xyz, color, width=0.1, lifetime_sec=0.0
):
    """
    Build a LINE_STRIP through ``points_xyz``.

    Fewer than two points cannot form a line and an empty ADD marker would
    leave the previous path on screen, so this returns a DELETE of this
    ``(ns, marker_id)`` instead.
    """
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 2:
        return delete(frame_id, stamp, ns, marker_id)

    marker = make_marker(
        frame_id,
        stamp,
        ns,
        marker_id,
        Marker.LINE_STRIP,
        color,
        (width, 0.0, 0.0),
        lifetime_sec=lifetime_sec,
    )
    return add_points(marker, pts)


def box_array(frame_id, stamp, ns, first_id, centres, extents, color, lifetime_sec=0.0):
    """
    Build a MarkerArray of CUBE markers, one per centre, ids from ``first_id``.

    ``extents`` is either one ``(x, y, z)`` size for every box or an
    ``(M, 3)`` array of per-box sizes.

    The array always leads with a DELETEALL, because the id range is as long as
    the cluster list and the previous frame's list may have been longer. Only
    clearing on an empty list left the surplus ids on screen forever under the
    default ``marker_lifetime_sec`` of 0.0 — phantom boxes for obstacles that
    had already left the scene. The consequence is that this must own its
    topic: publish nothing else on it.
    """
    centre_arr = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
    array = MarkerArray()
    array.markers.append(delete_all(frame_id, stamp, ns))
    if centre_arr.shape[0] == 0:
        return array

    extent_arr = np.asarray(extents, dtype=np.float64).reshape(-1, 3)
    if extent_arr.shape[0] == 1:
        extent_arr = np.repeat(extent_arr, centre_arr.shape[0], axis=0)

    for i, (centre, extent) in enumerate(zip(centre_arr, extent_arr)):
        marker = make_marker(
            frame_id,
            stamp,
            ns,
            first_id + i,
            Marker.CUBE,
            color,
            extent,
            lifetime_sec=lifetime_sec,
        )
        marker.pose.position.x = float(centre[0])
        marker.pose.position.y = float(centre[1])
        marker.pose.position.z = float(centre[2])
        array.markers.append(marker)
    return array
