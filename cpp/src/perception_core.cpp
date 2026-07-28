#include "ransac_seg_cpp/perception_core.hpp"

#include <cmath>
#include <cstdio>
#include <utility>

#include "ransac_seg_cpp/cloud_io.hpp"
#include "ransac_seg_cpp/costmap.hpp"
#include "ransac_seg_cpp/ros_params.hpp"

namespace ransac_seg_cpp
{

rclcpp::QoS sensor_qos(int depth)
{
  rclcpp::QoS qos(rclcpp::KeepLast(static_cast<std::size_t>(depth < 1 ? 1 : depth)));
  qos.best_effort();
  qos.durability_volatile();
  return qos;
}

rclcpp::QoS marker_qos(int depth)
{
  // rclcpp's own default for this initialisation is RELIABLE + VOLATILE, which
  // is exactly what rclpy gives create_publisher(..., depth). Spelled out
  // rather than left implicit so the difference from sensor_qos is visible at
  // the one line where it is decided.
  rclcpp::QoS qos(rclcpp::KeepLast(static_cast<std::size_t>(depth < 1 ? 1 : depth)));
  qos.reliable();
  qos.durability_volatile();
  return qos;
}

PerceptionHub::PerceptionHub(
  const std::string & node_name, const std::string & default_marker_topic)
: rclcpp::Node(node_name), plane_filter_(0.3, 5.0, 0.5)
{
  declare_common_params(*this);
  params_ = read_common_params(*this);
  if (params_.marker_topic == default_params().marker_topic) {
    // A consumer set may prefer its own topic, but an explicit launch-file
    // override must still win, so only the untouched default gives way.
    params_.marker_topic = default_marker_topic;
  }
  min_ground_points_frame_ = static_cast<int>(
    declare(
      *this, "min_ground_points_frame", static_cast<int64_t>(30),
      "Ground returns needed in a frame before a costmap is built"));
  plane_filter_.alpha = params_.plane_filter_alpha;
  plane_filter_.max_tilt_jump_deg = params_.plane_max_tilt_jump_deg;
  plane_filter_.max_offset_jump = params_.plane_max_offset_jump;
}

void PerceptionHub::add_consumer(std::unique_ptr<FrameConsumer> consumer)
{
  any_needs_costmap_ = any_needs_costmap_ || consumer->needs_costmap();
  consumers_.push_back(std::move(consumer));
}

void PerceptionHub::start()
{
  for (const auto & consumer : consumers_) {
    consumer->configure(*this, params_);
  }
  subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    params_.input_topic, sensor_qos(params_.qos_depth),
    [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg) {on_cloud(msg);});
  // Registered LAST, so it sees only genuine runtime sets: rclcpp runs
  // on-set callbacks for declarations too, and every consumer declares its own
  // parameters in configure() above.
  parameter_callback_ = add_on_set_parameters_callback(
    [this](const std::vector<rclcpp::Parameter> & request) {
      return on_set_parameters(request);
    });
}

rcl_interfaces::msg::SetParametersResult PerceptionHub::on_set_parameters(
  const std::vector<rclcpp::Parameter> & request)
{
  const rcl_interfaces::msg::SetParametersResult result =
    validate_and_apply(request, *this, params_, min_ground_points_frame_, plane_filter_);
  if (result.successful) {
    for (const auto & consumer : consumers_) {
      consumer->on_params_changed(params_);
    }
  }
  return result;
}

std::vector<float> PerceptionHub::range_gate(const std::vector<float> & points) const
{
  std::vector<float> kept;
  kept.reserve(points.size());
  const double min_range = params_.min_range;
  const double max_range = params_.max_range;
  for (std::size_t i = 0; i + 2 < points.size(); i += 3) {
    const double x = points[i];
    const double y = points[i + 1];
    const double radial = std::hypot(x, y);
    // x > 0 matters as much as the range bounds: rear returns are a different
    // piece of road and dragging them into one plane fit tilts it.
    if (radial >= min_range && radial <= max_range && x > 0.0) {
      kept.push_back(points[i]);
      kept.push_back(points[i + 1]);
      kept.push_back(points[i + 2]);
    }
  }
  return kept;
}

void PerceptionHub::on_cloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg)
{
  // DECLARED DEVIATION from perception_node.py, which calls this unguarded.
  // The throw itself is parity — cloud_io refuses a cloud with no z field, a
  // non-float xyz, a big-endian payload, a field past point_step or a truncated
  // payload — but the blast radius is not. In the six-process Python topology
  // one malformed message kills one node; here the hub feeds every consumer, so
  // the same message used to take all six down with std::terminate and no log
  // line. A malformed sweep is a property of the message, not of the node, so
  // the frame is dropped and the node stays up to decode the next one.
  std::vector<float> decoded;
  try {
    decoded = pointcloud2_to_xyz(*msg);
  } catch (const std::invalid_argument & error) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 5000, "dropping undecodable PointCloud2: %s", error.what());
    return;
  }
  if (decoded.empty()) {
    return;
  }
  const std::vector<float> points = range_gate(decoded);
  if (points.empty()) {
    return;
  }
  const std::size_t count = points.size() / 3;

  const Segmentation segmentation =
    segment_ground(points.data(), count, ransac_params(params_), params_.fallback_height);
  if (!segmentation.model.has_value()) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "no ground plane found; using fixed-height fallback");
  } else {
    const PlaneModel & model = *segmentation.model;
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 1000, "ground plane tilt=%.1f deg inliers=%.0f%% iters=%d",
      model.tilt_deg(), model.inlier_ratio * 100.0, model.iterations);
  }

  const std::optional<PlaneModel> filtered = plane_filter_.update(segmentation.model);
  OptPlane plane;
  if (filtered.has_value()) {
    plane = Plane{filtered->normal, filtered->offset};
  }

  std::vector<float> ground_xyz;
  std::vector<float> obstacle_xyz;
  ground_xyz.reserve(points.size());
  obstacle_xyz.reserve(points.size());
  for (std::size_t i = 0; i < count; ++i) {
    std::vector<float> & sink = segmentation.ground[i] != 0 ? ground_xyz : obstacle_xyz;
    sink.insert(sink.end(), points.begin() + 3 * i, points.begin() + 3 * i + 3);
  }

  std::shared_ptr<const CostGrid> grid;
  const std::size_t n_ground = ground_xyz.size() / 3;
  if (any_needs_costmap_ &&
    n_ground >= static_cast<std::size_t>(std::max(0, min_ground_points_frame_)))
  {
    CostmapOptions options;
    options.resolution = params_.resolution;
    options.robot_radius = params_.robot_radius;
    options.inflation_radius = params_.inflation_radius;
    options.cost_scaling = params_.cost_scaling;
    options.min_ground_points = params_.min_ground_points;
    options.min_obstacle_points = params_.min_obstacle_points;
    options.ray_fill = params_.ray_fill;
    options.ray_fill_bins = params_.ray_fill_bins;
    options.ray_fill_min_cells = params_.ray_fill_min_cells;
    options.plane = plane;
    // Built ONCE for every consumer that wants one; the six Python processes
    // each paid for their own copy of this.
    grid = std::make_shared<const CostGrid>(
      build_costmap(
        ground_xyz.data(), n_ground, 3, obstacle_xyz.data(), obstacle_xyz.size() / 3, 3,
        options));
  }

  const Frame frame{
    params_.frame_id_override.empty() ? msg->header.frame_id : params_.frame_id_override,
    // The cloud's own stamp, not the wall clock: markers describe the geometry
    // of *this* sweep. Under bag replay with use_sim_time a wall-clock stamp
    // asks the viewer to look up a TF at a time unrelated to the cloud's, and
    // the geometry is dropped or extrapolated.
    msg->header.stamp,
    points, segmentation.ground, ground_xyz, obstacle_xyz, plane, segmentation.model,
    std::move(grid)};

  for (const auto & consumer : consumers_) {
    consumer->on_frame(frame);
  }
}

}  // namespace ransac_seg_cpp
