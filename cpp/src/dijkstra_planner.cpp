// Dijkstra consumer — port of ransac_seg/dijkstra.py.
//
// The file that preceded the Python (`djikstra.py`, misspelled) ran a search
// over a set of ground cells with a uniform step cost of 1.0, threw the
// distances away and published only the cells it had reached — which is exactly
// what a flood fill gives you, so it did no work the BFS node did not already
// do. Here the distances ARE the product: every cell is coloured by what it
// costs to drive there over the weighted costmap. That field is the thing BFS
// structurally cannot produce, because BFS counts hops and ignores cost.

#include <cmath>
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

constexpr const char * kMarkerNs = "dijkstra_cost";

class DijkstraPlanner : public FrameConsumer
{
public:
  std::string name() const override {return "dijkstra";}

  void configure(rclcpp::Node & node, const CommonParams & params) override
  {
    params_ = params;
    start_search_radius_ = declare(
      node, "dijkstra.start_search_radius", 2.0,
      "Radius in metres searched for a traversable cell at the ego origin");
    connectivity_ = static_cast<int>(
      declare(
        node, "dijkstra.connectivity", static_cast<int64_t>(8), "Grid neighbourhood: 4 or 8"));
    // Fail here rather than throwing out of every callback: a bad launch value
    // should stop the node, not flood the log.
    validate_connectivity(connectivity_);
    field_bands_ = static_cast<int>(
      declare(
        node, "dijkstra.field_bands", static_cast<int64_t>(6),
        "Colour levels the published field is cut into"));
    logger_ = std::make_unique<rclcpp::Logger>(node.get_logger());
    clock_ = node.get_clock();
    publisher_ = node.create_publisher<visualization_msgs::msg::MarkerArray>(
      declare(
        node, "dijkstra.marker_topic", std::string("/planning/dijkstra_cost"),
        "Topic for this consumer's cost field"),
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

    const DijkstraField field = dijkstra(grid, *start, connectivity_);
    std::vector<double> values(field.cost.size());
    std::vector<uint8_t> reachable(field.cost.size());
    std::size_t reached = 0;
    double worst = 0.0;
    for (std::size_t i = 0; i < field.cost.size(); ++i) {
      values[i] = field.cost[i];
      reachable[i] = std::isfinite(values[i]) ? 1 : 0;
      if (reachable[i] != 0) {
        ++reached;
        worst = std::max(worst, values[i]);
      }
    }
    publisher_->publish(
      banded_field(
        frame.frame_id, frame.stamp, kMarkerNs, grid, values, reachable, field_bands_,
        params_.marker_lifetime_sec));
    if (reached > 0) {
      RCLCPP_INFO_THROTTLE(
        *logger_, *clock_, 1000, "reachable cells=%zu max cost=%.1f", reached, worst);
    }
  }

private:
  CommonParams params_;
  double start_search_radius_ = 2.0;
  int connectivity_ = 8;
  int field_bands_ = 6;
  std::unique_ptr<rclcpp::Logger> logger_;
  rclcpp::Clock::SharedPtr clock_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher_;
};

}  // namespace

std::unique_ptr<FrameConsumer> make_dijkstra_planner()
{
  return std::make_unique<DijkstraPlanner>();
}

}  // namespace ransac_seg_cpp
