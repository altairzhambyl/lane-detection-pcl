// BFS consumer — port of ransac_seg/bfs.py.
//
// This is the baseline the weighted Dijkstra field is read against. BFS answers
// one question honestly and completely — which cells are connected to the
// vehicle, and how many 4-connected steps away — and it answers nothing about
// cost, so a cell squeezed against an obstacle's clearance margin looks exactly
// like one in the middle of the road. Publishing both fields side by side is
// what makes the cost model visible.

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <visualization_msgs/msg/marker_array.hpp>

#include "ransac_seg_cpp/consumers.hpp"
#include "ransac_seg_cpp/field_markers.hpp"
#include "ransac_seg_cpp/planners.hpp"
#include "ransac_seg_cpp/ros_params.hpp"

namespace ransac_seg_cpp
{
namespace
{

constexpr const char * kMarkerNs = "bfs_hops";

class BfsPlanner : public FrameConsumer
{
public:
  std::string name() const override {return "bfs";}

  void configure(rclcpp::Node & node, const CommonParams & params) override
  {
    params_ = params;
    start_search_radius_ = declare(
      node, "bfs.start_search_radius", 2.0,
      "Radius in metres searched for a traversable cell at the ego origin");
    connectivity_ = static_cast<int>(
      declare(node, "bfs.connectivity", static_cast<int64_t>(4), "Grid neighbourhood: 4 or 8"));
    // Fail here rather than throwing out of every callback: a bad launch value
    // should stop the node, not flood the log.
    validate_connectivity(connectivity_);
    field_bands_ = static_cast<int>(
      declare(
        node, "bfs.field_bands", static_cast<int64_t>(6),
        "Colour levels the published field is cut into"));
    logger_ = std::make_unique<rclcpp::Logger>(node.get_logger());
    clock_ = node.get_clock();
    publisher_ = node.create_publisher<visualization_msgs::msg::MarkerArray>(
      declare(
        node, "bfs.marker_topic", std::string("/planning/bfs_hops"),
        "Topic for this consumer's hop field"),
      marker_qos(params.qos_depth));
  }

  void on_params_changed(const CommonParams & params) override {params_ = params;}

  void on_frame(const Frame & frame) override
  {
    if (frame.grid == nullptr) {
      publisher_->publish(clear_field(frame.frame_id, frame.stamp, kMarkerNs));
      return;
    }
    const CostGrid & grid = *frame.grid;
    const std::optional<Cell> start = snap_to_traversable(grid, 0.0, 0.0, start_search_radius_);
    if (!start.has_value()) {
      RCLCPP_WARN_THROTTLE(
        *logger_, *clock_, 1000,
        "no traversable cell within %.1f m of the ego origin; not searching",
        start_search_radius_);
      publisher_->publish(clear_field(frame.frame_id, frame.stamp, kMarkerNs));
      return;
    }

    const std::vector<int32_t> hops = bfs(grid, *start, connectivity_);
    std::vector<double> values(hops.size());
    std::vector<uint8_t> reached(hops.size());
    std::size_t count = 0;
    int32_t worst = 0;
    for (std::size_t i = 0; i < hops.size(); ++i) {
      values[i] = hops[i];
      reached[i] = hops[i] >= 0 ? 1 : 0;
      if (reached[i] != 0) {
        ++count;
        worst = std::max(worst, hops[i]);
      }
    }
    publisher_->publish(
      banded_field(
        frame.frame_id, frame.stamp, kMarkerNs, grid, values, reached, field_bands_,
        params_.marker_lifetime_sec));
    if (count > 0) {
      RCLCPP_INFO_THROTTLE(
        *logger_, *clock_, 1000, "reachable cells=%zu max hops=%d", count, worst);
    }
  }

private:
  CommonParams params_;
  double start_search_radius_ = 2.0;
  int connectivity_ = 4;
  int field_bands_ = 6;
  std::unique_ptr<rclcpp::Logger> logger_;
  rclcpp::Clock::SharedPtr clock_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher_;
};

}  // namespace

std::unique_ptr<FrameConsumer> make_bfs_planner()
{
  return std::make_unique<BfsPlanner>();
}

}  // namespace ransac_seg_cpp
