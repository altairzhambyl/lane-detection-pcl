// DFS consumer — port of ransac_seg/dfs.py. A traversal-order baseline, and
// explicitly not a planner.
//
// Say this plainly, because the repository used to imply otherwise by shipping
// DFS alongside A* as if they were alternatives: depth-first search explores the
// exact same reachable set as breadth-first search. Connectivity is a property
// of the graph, not of the order you walk it. It produces no shortest path, no
// hop distance and no cost.
//
// What it does produce, and the only thing that differs from BFS, is the order
// of discovery — a long snaking chain instead of expanding rings. To keep that
// claim honest rather than decorative, this consumer re-runs BFS on the same
// grid and logs whether the two visited sets actually match, so the assertion is
// measured on live data instead of asserted in a README.

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

constexpr const char * kMarkerNs = "dfs_order";

class DfsBaseline : public FrameConsumer
{
public:
  std::string name() const override {return "dfs";}

  void configure(rclcpp::Node & node, const CommonParams & params) override
  {
    params_ = params;
    start_search_radius_ = declare(
      node, "dfs.start_search_radius", 2.0,
      "Radius in metres searched for a traversable cell at the ego origin");
    connectivity_ = static_cast<int>(
      declare(node, "dfs.connectivity", static_cast<int64_t>(4), "Grid neighbourhood: 4 or 8"));
    // Fail here rather than throwing out of every callback: a bad launch value
    // should stop the node, not flood the log.
    validate_connectivity(connectivity_);
    field_bands_ = static_cast<int>(
      declare(
        node, "dfs.field_bands", static_cast<int64_t>(6),
        "Colour levels the published field is cut into"));
    logger_ = std::make_unique<rclcpp::Logger>(node.get_logger());
    clock_ = node.get_clock();
    publisher_ = node.create_publisher<visualization_msgs::msg::MarkerArray>(
      declare(
        node, "dfs.marker_topic", std::string("/planning/dfs_order"),
        "Topic for this consumer's discovery-order field"),
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

    const std::vector<int32_t> order = dfs(grid, *start, connectivity_);
    std::vector<double> values(order.size());
    std::vector<uint8_t> visited(order.size());
    std::size_t visited_count = 0;
    for (std::size_t i = 0; i < order.size(); ++i) {
      values[i] = order[i];
      visited[i] = order[i] >= 0 ? 1 : 0;
      visited_count += visited[i];
    }
    publisher_->publish(
      banded_field(
        frame.frame_id, frame.stamp, kMarkerNs, grid, values, visited, field_bands_,
        params_.marker_lifetime_sec));

    // The audit this consumer's whole claim rests on, run on live data.
    const std::vector<int32_t> hops = bfs(grid, *start, connectivity_);
    std::size_t reached_count = 0;
    bool identical = hops.size() == visited.size();
    for (std::size_t i = 0; i < hops.size(); ++i) {
      const uint8_t reached = hops[i] >= 0 ? 1 : 0;
      reached_count += reached;
      identical = identical && reached == visited[i];
    }
    RCLCPP_INFO_THROTTLE(
      *logger_, *clock_, 1000,
      "dfs visited=%zu bfs visited=%zu sets identical=%s; dfs gives order only, "
      "no path and no cost",
      visited_count, reached_count, identical ? "true" : "false");
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

std::unique_ptr<FrameConsumer> make_dfs_baseline()
{
  return std::make_unique<DfsBaseline>();
}

}  // namespace ransac_seg_cpp
