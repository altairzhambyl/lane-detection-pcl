#!/usr/bin/env python3
"""
ROS parameter declarations shared by every node in the package.

Topics, grid resolution and the whole RANSAC configuration used to be literals
buried in six callbacks, so retuning for a different sensor meant editing (and
diverging) six files. Declaring them here once means every node exposes the
same names, a launch file can override any of them, and the defaults below are
the single place where "today's behaviour" is written down.
"""

from dataclasses import dataclass, fields, replace

from rcl_interfaces.msg import ParameterDescriptor

DEFAULTS = {
    "input_topic": ("/sensor/lidar_front/points", "PointCloud2 topic to subscribe to"),
    "marker_topic": (
        "/semantic_map/squares",
        "Topic for this node's visualisation markers",
    ),
    "qos_depth": (10, "Queue depth for the cloud subscription and marker publisher"),
    "frame_id_override": (
        "",
        "Frame to stamp outgoing markers with; empty means use the incoming cloud frame",
    ),
    "resolution": (0.5, "Costmap cell size in metres"),
    "robot_radius": (
        0.6,
        "Cells within this distance of an obstacle are untraversable",
    ),
    "inflation_radius": (
        1.5,
        "Distance in metres over which obstacle proximity still raises cost",
    ),
    "cost_scaling": (3.0, "Exponential decay rate of the inflation cost falloff"),
    "min_ground_points": (
        1,
        "Ground returns needed in a cell before it counts as free",
    ),
    "min_obstacle_points": (
        3,
        "Obstacle returns needed in a cell before it counts as occupied",
    ),
    "ray_fill": (
        True,
        "Carve cells the beams crossed as free, closing the gaps between rings",
    ),
    "ray_fill_bins": (720, "Azimuth bins used when carving free space"),
    "ray_fill_min_cells": (
        2,
        "Ground cells an azimuth needs before it may carve a ray",
    ),
    "min_range": (0.5, "Drop returns closer than this; rejects the sensor's own mount"),
    "max_range": (60.0, "Drop returns beyond this range in metres"),
    "marker_lifetime_sec": (0.0, "Marker lifetime in seconds; 0 means never expire"),
    "distance_threshold": (
        0.20,
        "RANSAC inlier band around the plane in metres; 0.20 was chosen on "
        "SemanticKITTI 00 and 05, not on synthetic data",
    ),
    "max_iterations": (200, "Upper bound on RANSAC hypotheses per frame"),
    "min_iterations": (
        12,
        "Hypotheses always drawn before the adaptive budget applies",
    ),
    "confidence": (0.99, "Target probability of drawing one all-inlier sample"),
    "max_tilt_deg": (
        10.0, "Reject planes tilted further than this from +z"),
    "min_inlier_ratio": (
        0.15,
        "Below this inlier fraction the fit is treated as failed",
    ),
    "max_scoring_points": (20000, "Cloud subsample size used to score hypotheses"),
    "fallback_height": (0.2, "Fixed z threshold used when no plane is found"),
    "plane_filter_alpha": (
        0.3,
        "Blend weight of a new plane estimate against the held one",
    ),
    "plane_max_tilt_jump_deg": (
        5.0,
        "Reject a plane estimate tilting more than this from the held plane",
    ),
    "plane_max_offset_jump": (
        0.5,
        "Reject a plane estimate whose offset moves more than this in metres",
    ),
}

# Shared parameters that are consumed once, while the node is being built: the
# subscription's topic and queue depth and the marker publishers' topic and
# depth. Rewriting them on a live node cannot change anything, so the runtime
# callback refuses them instead of acknowledging a change that will not happen.
CONSTRUCTION_ONLY = frozenset({"input_topic", "marker_topic", "qos_depth"})

# Keys accepted by ground_plane.segment_ground (directly or forwarded to
# fit_plane_ransac). Anything else would raise a TypeError at the call site.
_RANSAC_KEYS = (
    "distance_threshold",
    "fallback_height",
    "max_iterations",
    "min_iterations",
    "confidence",
    "max_tilt_deg",
    "min_inlier_ratio",
    "max_scoring_points",
)


@dataclass(frozen=True)
class CommonParams:
    """Every tunable a perception node reads, resolved once per node start-up."""

    input_topic: str
    marker_topic: str
    qos_depth: int
    frame_id_override: str
    resolution: float
    robot_radius: float
    inflation_radius: float
    cost_scaling: float
    min_ground_points: int
    min_obstacle_points: int
    ray_fill: bool
    ray_fill_bins: int
    ray_fill_min_cells: int
    min_range: float
    max_range: float
    marker_lifetime_sec: float
    distance_threshold: float
    max_iterations: int
    min_iterations: int
    confidence: float
    max_tilt_deg: float
    min_inlier_ratio: float
    max_scoring_points: int
    fallback_height: float
    plane_filter_alpha: float
    plane_max_tilt_jump_deg: float
    plane_max_offset_jump: float


def declare(node, name, default, description=""):
    """Declare one extra parameter on ``node`` and return its resolved value."""
    if not node.has_parameter(name):
        node.declare_parameter(
            name, default, ParameterDescriptor(description=description)
        )
    return node.get_parameter(name).value


def declare_common_params(node):
    """Declare every shared parameter on ``node`` with its documented default."""
    for name, (default, description) in DEFAULTS.items():
        declare(node, name, default, description)


def coerce(name, value):
    """
    Coerce one parameter value to the type its declared default fixes.

    A launch file can hand an int where a float is wanted (and rclpy keeps it
    an int), which then propagates into the numpy maths as one. The declared
    default is the authority on the intended type.
    """
    wanted = type(DEFAULTS[name][0])
    return value if wanted is str else wanted(value)


def read_common_params(node):
    """Read the declared shared parameters off ``node`` into a CommonParams."""
    values = {
        field.name: coerce(field.name, node.get_parameter(field.name).value)
        for field in fields(CommonParams)
    }
    return CommonParams(**values)


def with_updates(params, updates):
    """Return a copy of ``params`` with the named shared parameters replaced."""
    return replace(params, **{name: coerce(name, v) for name, v in updates.items()})


def ransac_kwargs(params):
    """Return the subset of ``params`` that ``segment_ground`` accepts."""
    return {key: getattr(params, key) for key in _RANSAC_KEYS}
