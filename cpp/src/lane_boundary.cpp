// Lane boundary consumer — port of ransac_seg/curve.py.
//
// The Python this replaces also drew the grid squares, on the same topic,
// namespace and id as ransac_node.py, so RViz showed whichever arrived last.
// Squares belong to the ground-segmentation consumer alone; this one publishes
// only what it computes — the two boundaries and the centreline they were
// measured against, lifted onto the fitted plane instead of z=0.0.

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <visualization_msgs/msg/marker.hpp>

#include "ransac_seg_cpp/consumers.hpp"
#include "ransac_seg_cpp/cost_grid.hpp"
#include "ransac_seg_cpp/lane.hpp"
#include "ransac_seg_cpp/markers.hpp"
#include "ransac_seg_cpp/ros_params.hpp"

namespace ransac_seg_cpp
{
namespace
{

constexpr const char * kLeftNs = "road_left";
constexpr const char * kRightNs = "road_right";
constexpr const char * kCenterNs = "road_center";

class LaneBoundary : public FrameConsumer
{
public:
  std::string name() const override {return "lane";}

  /// Boundary fitting reads the ground returns directly; rasterising a costmap
  /// this consumer never consults would cost milliseconds per frame for
  /// nothing. With the shared hub that saving only survives if EVERY consumer
  /// declines the costmap, which is the honest generalisation of the Python
  /// per-node flag.
  bool needs_costmap() const override {return false;}

  void configure(rclcpp::Node & node, const CommonParams & params) override
  {
    params_ = params;
    lane_.bin_length = declare(
      node, "lane.lane_bin_length", 2.0, "Arc length of one boundary sample bin");
    lane_.min_points_per_bin = static_cast<int>(
      declare(
        node, "lane.lane_min_points_per_bin", static_cast<int64_t>(8),
        "Returns a bin needs before its edges are trusted"));
    lane_.edge_percentile = declare(
      node, "lane.lane_edge_percentile", 2.0,
      "Percentile taken as the road edge; tolerates that fraction of outliers");
    lane_.centerline_degree = static_cast<int>(
      declare(
        node, "lane.lane_centerline_degree", static_cast<int64_t>(2),
        "Polynomial degree of the fitted centreline"));
    lane_.robust_iterations = static_cast<int>(
      declare(
        node, "lane.lane_robust_iterations", static_cast<int64_t>(3),
        "Median re-fit passes of the centreline"));
    lane_.max_gap = declare(
      node, "lane.lane_max_gap", 6.0, "Boundaries are truncated at gaps this long");
    lane_.smoothing_per_point = declare(
      node, "lane.lane_smoothing_per_point", 0.02, "Spline residual budget per occupied bin");
    lane_.n_samples = static_cast<int>(
      declare(
        node, "lane.lane_n_samples", static_cast<int64_t>(120),
        "Points published along each boundary"));
    lane_.min_width = declare(
      node, "lane.lane_min_width", 2.0, "Fits narrower than this are rejected");
    lane_.max_width = declare(
      node, "lane.lane_max_width", 12.0, "Fits wider than this are rejected");
    lane_.min_bin_fill = declare(
      node, "lane.lane_min_bin_fill", 0.5,
      "Bin population needed relative to the median, dropping partial end bins");

    logger_ = std::make_unique<rclcpp::Logger>(node.get_logger());
    clock_ = node.get_clock();
    const auto make = [&node, &params](const char * name, const char * topic,
        const char * description) {
        return node.create_publisher<visualization_msgs::msg::Marker>(
          declare(node, name, std::string(topic), description), marker_qos(params.qos_depth));
      };
    pub_left_ = make("lane.left_topic", "/semantic_map/road_left", "Topic for the left boundary");
    pub_right_ =
      make("lane.right_topic", "/semantic_map/road_right", "Topic for the right boundary");
    pub_center_ = make("lane.center_topic", "/semantic_map/road_center", "Topic for the centreline");
  }

  void on_params_changed(const CommonParams & params) override {params_ = params;}

  void on_frame(const Frame & frame) override
  {
    const LaneFit fit =
      fit_lane(frame.ground_xyz.data(), frame.ground_xyz.size() / 3, 3, lane_);
    if (!fit.valid) {
      RCLCPP_INFO_THROTTLE(
        *logger_, *clock_, 5000, "no lane boundary published: %s", fit.reason.c_str());
      clear(frame);
      return;
    }
    strip(frame, pub_left_, kLeftNs, fit.left, {0.0, 0.4, 1.0, 0.9});
    strip(frame, pub_right_, kRightNs, fit.right, {1.0, 0.2, 0.0, 0.9});
    strip(frame, pub_center_, kCenterNs, fit.centerline, {1.0, 0.9, 0.0, 0.7});
  }

private:
  using MarkerPublisher = rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr;

  /// Publish one boundary as a LINE_STRIP lifted onto the fitted plane.
  void strip(
    const Frame & frame, const MarkerPublisher & publisher, const char * ns,
    const std::vector<double> & xy, const Color & color)
  {
    const std::size_t count = xy.size() / 2;
    std::vector<double> xyz(count * 3);
    for (std::size_t i = 0; i < count; ++i) {
      xyz[3 * i] = xy[2 * i];
      xyz[3 * i + 1] = xy[2 * i + 1];
      xyz[3 * i + 2] = plane_z(frame.plane, xy[2 * i], xy[2 * i + 1]);
    }
    publisher->publish(
      line_strip(
        frame.frame_id, frame.stamp, ns, 0, xyz.data(), count, color, 0.1,
        params_.marker_lifetime_sec));
  }

  /// DECLARED DEVIATION from curve.py, which published a DELETEALL per
  /// namespace: an empty line_strip is a DELETE of exactly this (ns, id), which
  /// clears the same geometry without wiping whatever else a viewer is showing.
  void clear(const Frame & frame)
  {
    const std::pair<MarkerPublisher, const char *> outputs[] = {
      {pub_left_, kLeftNs}, {pub_right_, kRightNs}, {pub_center_, kCenterNs}};
    for (const auto & [publisher, ns] : outputs) {
      publisher->publish(
        line_strip(
          frame.frame_id, frame.stamp, ns, 0, nullptr, 0, {0.0, 0.0, 0.0, 0.0}, 0.1,
          params_.marker_lifetime_sec));
    }
  }

  CommonParams params_;
  LaneOptions lane_;
  std::unique_ptr<rclcpp::Logger> logger_;
  rclcpp::Clock::SharedPtr clock_;
  MarkerPublisher pub_left_;
  MarkerPublisher pub_right_;
  MarkerPublisher pub_center_;
};

}  // namespace

std::unique_ptr<FrameConsumer> make_lane_boundary()
{
  return std::make_unique<LaneBoundary>();
}

}  // namespace ransac_seg_cpp
