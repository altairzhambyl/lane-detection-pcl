#!/usr/bin/env python3
"""
Base node carrying the subscribe/decode/segment/rasterise pipeline.

Every node in this package repeated the same forty lines: decode the cloud,
call ``segment_ground``, log (or forget to log) the fit quality, quantise the
ground returns. The copies drifted — only one of them logged the plane, none
of them gated by range, and none rejected returns from behind the vehicle,
which pooled rear-facing points into the road fit. Subclasses now implement
``on_frame`` and nothing else.
"""

from dataclasses import dataclass, replace

import numpy as np
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import PointCloud2

from ransac_seg.cloud_io import pointcloud2_to_xyz
from ransac_seg.grid import build_costmap
from ransac_seg.ground_plane import segment_ground
from ransac_seg.params import (
    CONSTRUCTION_ONLY,
    DEFAULTS,
    coerce,
    declare,
    declare_common_params,
    ransac_kwargs,
    read_common_params,
    with_updates,
)
from ransac_seg.plane_filter import PlaneFilter


def sensor_qos(depth):
    """
    Return the QoS a LiDAR subscription needs.

    Passing a bare depth to ``create_subscription`` gives rclpy's RELIABLE
    default, and DDS refuses to match a RELIABLE subscription with a
    BEST_EFFORT publisher — which is what every production LiDAR driver, and
    ``ros2 bag play`` of data recorded from one, offers. The node then sits
    there receiving nothing, with only a QoS-incompatibility line in the log to
    say why. Requesting BEST_EFFORT matches publishers of either kind.
    """
    return QoSProfile(
        depth=int(depth),
        history=HistoryPolicy.KEEP_LAST,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


@dataclass(frozen=True)
class Frame:
    """One decoded, segmented LiDAR frame plus the costmap built from it."""

    frame_id: str
    stamp: object
    points: np.ndarray
    ground_mask: np.ndarray
    ground_xyz: np.ndarray
    obstacle_xyz: np.ndarray
    plane: object
    raw_plane: object
    grid: object


class PerceptionNode(Node):
    """
    Shared subscribe/decode/segment/rasterise pipeline for every node.

    Subclasses set ``needs_costmap`` and implement ``on_frame(frame)``. They
    own their own publishers and should default the marker topic to
    ``self.params.marker_topic``.

    Contract that both consumers depend on: ``on_frame`` is called for every
    frame that decoded to at least one in-range point, but ``frame.grid`` is
    ``None`` whenever no usable costmap exists — either ``needs_costmap`` is
    False, or the frame yielded fewer ground returns than the
    ``min_ground_points_frame`` parameter. A ``None`` grid means "publish a
    clear marker", never "publish an empty ADD"; an empty ADD leaves the
    previous frame's geometry on screen.
    """

    needs_costmap = True

    def __init__(self, node_name, *, default_marker_topic="/semantic_map/squares"):
        """Declare parameters, build the plane filter and subscribe to the cloud."""
        super().__init__(node_name)
        declare_common_params(self)
        params = read_common_params(self)
        if params.marker_topic == DEFAULTS["marker_topic"][0]:
            # A subclass may prefer its own topic, but an explicit launch-file
            # override must still win, so only the untouched default gives way.
            params = replace(params, marker_topic=default_marker_topic)
        self.params = params
        self.min_ground_points_frame = int(
            declare(
                self,
                "min_ground_points_frame",
                30,
                "Ground returns needed in a frame before a costmap is built",
            )
        )
        self.plane_filter = PlaneFilter(
            alpha=self.params.plane_filter_alpha,
            max_tilt_jump_deg=self.params.plane_max_tilt_jump_deg,
            max_offset_jump=self.params.plane_max_offset_jump,
        )
        self.create_subscription(
            PointCloud2,
            self.params.input_topic,
            self._on_cloud,
            sensor_qos(self.params.qos_depth),
        )
        # Registered last, so it sees only genuine runtime sets: rclpy runs
        # on-set callbacks for declarations too, and every subclass declares
        # its own parameters after this constructor returns.
        self.add_on_set_parameters_callback(self._on_set_parameters)

    def _on_set_parameters(self, parameters):
        """
        Apply a runtime change to the shared parameters, or refuse it with a reason.

        Without this, ``ros2 param set`` reported success for all 24 shared
        parameters while the node kept running on the values it snapshotted in
        ``__init__`` — an operator retuning inflation or range on a live
        vehicle got an acknowledgement and no behaviour change. Anything this
        node genuinely cannot re-read (a topic already subscribed, a parameter
        a subclass snapshotted into an attribute of its own) is now rejected
        rather than silently ignored.
        """
        updates = {}
        for parameter in parameters:
            name = parameter.name
            if not self.has_parameter(name):
                # A declaration, not a set: the parameter does not exist yet.
                continue
            if name in CONSTRUCTION_ONLY:
                return SetParametersResult(
                    successful=False,
                    reason=f"'{name}' is read once at start-up; restart the node",
                )
            if name not in DEFAULTS and name != "min_ground_points_frame":
                return SetParametersResult(
                    successful=False,
                    reason=f"'{name}' cannot be changed while the node is running",
                )
            try:
                updates[name] = (
                    int(parameter.value)
                    if name == "min_ground_points_frame"
                    else coerce(name, parameter.value)
                )
            except (TypeError, ValueError) as error:
                return SetParametersResult(
                    successful=False, reason=f"'{name}': {error}"
                )

        # Nothing is mutated until every parameter in the request has passed,
        # so a rejected batch leaves the node exactly as it was.
        self.min_ground_points_frame = updates.pop(
            "min_ground_points_frame", self.min_ground_points_frame
        )
        if updates:
            self.params = with_updates(self.params, updates)
            self.plane_filter.alpha = self.params.plane_filter_alpha
            self.plane_filter.max_tilt_jump_deg = self.params.plane_max_tilt_jump_deg
            self.plane_filter.max_offset_jump = self.params.plane_max_offset_jump
        return SetParametersResult(successful=True)

    def on_frame(self, frame):
        """Handle one processed frame; implemented by every subclass."""
        raise NotImplementedError

    def _range_gate(self, points):
        """Keep only returns in the sensor's usable forward annulus."""
        radial = np.hypot(points[:, 0], points[:, 1])
        # x > 0 matters as much as the range bounds: rear returns are a
        # different piece of road and dragging them into one plane fit tilts it.
        keep = (
            (radial >= self.params.min_range)
            & (radial <= self.params.max_range)
            & (points[:, 0] > 0.0)
        )
        return points[keep]

    def _on_cloud(self, msg):
        """Decode, segment and rasterise one cloud, then hand it to ``on_frame``."""
        points = pointcloud2_to_xyz(msg)
        if points.shape[0] == 0:
            return
        points = self._range_gate(points)
        if points.shape[0] == 0:
            return

        ground_mask, raw_plane = segment_ground(points, **ransac_kwargs(self.params))
        if raw_plane is None:
            self.get_logger().warn(
                "no ground plane found; using fixed-height fallback",
                throttle_duration_sec=5.0,
            )
        else:
            self.get_logger().info(
                f"ground plane tilt={raw_plane.tilt_deg:.1f} deg "
                f"inliers={raw_plane.inlier_ratio:.0%} iters={raw_plane.iterations}",
                throttle_duration_sec=1.0,
            )

        plane = self.plane_filter.update(raw_plane)
        ground_xyz = points[ground_mask]
        obstacle_xyz = points[~ground_mask]

        grid = None
        if self.needs_costmap and ground_xyz.shape[0] >= self.min_ground_points_frame:
            grid = build_costmap(
                ground_xyz,
                obstacle_xyz,
                resolution=self.params.resolution,
                robot_radius=self.params.robot_radius,
                inflation_radius=self.params.inflation_radius,
                cost_scaling=self.params.cost_scaling,
                min_ground_points=self.params.min_ground_points,
                min_obstacle_points=self.params.min_obstacle_points,
                ray_fill=self.params.ray_fill,
                ray_fill_bins=self.params.ray_fill_bins,
                ray_fill_min_cells=self.params.ray_fill_min_cells,
                plane=plane,
            )

        frame_id = self.params.frame_id_override or msg.header.frame_id
        self.on_frame(
            Frame(
                frame_id=frame_id,
                # The cloud's own stamp, not the wall clock: markers describe
                # the geometry of *this* sweep. Under the standard bag-replay
                # setup (use_sim_time, clouds republished with their recorded
                # stamps) a wall-clock stamp asks the viewer to look up a TF at
                # a time unrelated to the cloud's, and the geometry is dropped
                # or extrapolated.
                stamp=msg.header.stamp,
                points=points,
                ground_mask=ground_mask,
                ground_xyz=ground_xyz,
                obstacle_xyz=obstacle_xyz,
                plane=plane,
                raw_plane=raw_plane,
                grid=grid,
            )
        )
