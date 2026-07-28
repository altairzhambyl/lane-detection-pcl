// A* planner consumer — port of ransac_seg/astar.py.
//
// The version this replaces searched a bare set of cells holding ground
// returns, started from "whichever cell was nearest the origin this frame" and
// aimed at "whichever cell was furthest away this frame", so the goal jumped
// every time the sweep changed shape. It also used a Euclidean heuristic on a
// 4-connected graph and had no CLOSED set. Here the start is the ego origin,
// the goal is a parameter in metres snapped onto the map, and the search runs
// over the tri-state costmap so obstacles are obstacles rather than holes in
// the ground returns.

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <visualization_msgs/msg/marker.hpp>

#include "ransac_seg_cpp/consumers.hpp"
#include "ransac_seg_cpp/markers.hpp"
#include "ransac_seg_cpp/planners.hpp"
#include "ransac_seg_cpp/ros_params.hpp"

namespace ransac_seg_cpp
{
namespace
{

constexpr const char * kMarkerNs = "astar_path";
constexpr Color kPathColor{1.0, 0.25, 0.2, 0.95};

class AStarPlanner : public FrameConsumer
{
public:
  std::string name() const override {return "astar";}

  void configure(rclcpp::Node & node, const CommonParams & params) override
  {
    params_ = params;
    goal_x_ = declare(node, "astar.goal_x", 25.0, "Goal x in the sensor frame");
    goal_y_ = declare(node, "astar.goal_y", 0.0, "Goal y in the sensor frame");
    goal_search_radius_ = declare(
      node, "astar.goal_search_radius", 5.0,
      "Radius in metres searched for a traversable cell near the goal");
    start_search_radius_ = declare(
      node, "astar.start_search_radius", 2.0,
      "Radius in metres searched for a traversable cell at the ego origin");
    connectivity_ = static_cast<int>(
      declare(
        node, "astar.connectivity", static_cast<int64_t>(8), "Grid neighbourhood: 4 or 8"));
    // Fail here rather than throwing out of every callback: a bad launch value
    // should stop the node, not flood the log.
    validate_connectivity(connectivity_);
    path_width_ = declare(
      node, "astar.path_width", 0.15, "Published path line width in metres");
    // Which way "onwards" points when the requested goal is unusable. It is +x
    // for a forward-facing sensor, but a rear or side mount needs its own.
    const std::vector<double> axis = declare(
      node, "astar.goal_fallback_axis", std::vector<double>{1.0, 0.0},
      "Direction the fallback goal is chosen furthest along");
    if (axis.size() != 2) {
      throw std::invalid_argument("astar.goal_fallback_axis must hold exactly two values");
    }
    axis_x_ = axis[0];
    axis_y_ = axis[1];

    logger_ = std::make_unique<rclcpp::Logger>(node.get_logger());
    clock_ = node.get_clock();
    publisher_ = node.create_publisher<visualization_msgs::msg::Marker>(
      declare(
        node, "astar.marker_topic", std::string("/planning/astar_path"),
        "Topic for this consumer's path marker"),
      marker_qos(params.qos_depth));
  }

  void on_params_changed(const CommonParams & params) override {params_ = params;}

  void on_frame(const Frame & frame) override
  {
    if (frame.grid == nullptr) {
      clear(frame);
      return;
    }
    const CostGrid & grid = *frame.grid;
    const std::optional<Cell> start = snap_to_traversable(grid, 0.0, 0.0, start_search_radius_);
    if (!start.has_value()) {
      RCLCPP_WARN_THROTTLE(
        *logger_, *clock_, 1000,
        "no traversable cell within %.1f m of the ego origin; not planning",
        start_search_radius_);
      clear(frame);
      return;
    }

    const std::vector<Cell> path = plan(grid, *start);
    if (path.empty()) {
      RCLCPP_WARN_THROTTLE(*logger_, *clock_, 1000, "no route to any goal in this frame");
      clear(frame);
      return;
    }

    std::vector<std::array<int64_t, 2>> cells;
    cells.reserve(path.size());
    for (const Cell & cell : path) {
      cells.push_back({cell.first, cell.second});
    }
    const std::vector<std::array<double, 3>> world = grid.cell_to_world_xyz(cells);
    std::vector<double> flat;
    flat.reserve(world.size() * 3);
    for (const std::array<double, 3> & point : world) {
      flat.insert(flat.end(), point.begin(), point.end());
    }
    publisher_->publish(
      line_strip(
        frame.frame_id, frame.stamp, kMarkerNs, 0, flat.data(), world.size(), kPathColor,
        path_width_, params_.marker_lifetime_sec));
    RCLCPP_INFO_THROTTLE(
      *logger_, *clock_, 1000, "path cost=%.1f length=%.1f m over %zu cells",
      path_cost(grid, path), path_length_m(grid, path), path.size());
  }

private:
  /// The cell route to the requested goal, or to a provably reachable one.
  std::vector<Cell> plan(const CostGrid & grid, Cell start) const
  {
    const std::optional<Cell> goal =
      select_goal(grid, goal_x_, goal_y_, goal_search_radius_, axis_x_, axis_y_);
    if (goal.has_value()) {
      PathResult direct = astar(grid, start, *goal, connectivity_);
      if (!direct.path.empty()) {
        return direct.path;
      }
    }
    // The requested goal is either off the map or walled off from the ego
    // position. Rather than publish nothing, aim at the furthest cell that is
    // provably connected to the start — the Dijkstra field is what makes
    // "provably" true and keeps the choice stable between frames.
    const DijkstraField field = dijkstra(grid, start, connectivity_);
    const std::optional<Cell> fallback = furthest_reachable(field, grid, axis_x_, axis_y_);
    if (!fallback.has_value() || *fallback == start) {
      return {};
    }
    return astar(grid, start, *fallback, connectivity_).path;
  }

  /// Retire the published path.
  ///
  /// DECLARED DEVIATION from astar.py, which publishes a DELETEALL here, and
  /// the same substitution the ground and lane consumers document: RViz2 and
  /// Foxglove act on DELETEALL across the whole display, not just the sending
  /// namespace. An empty LINE_STRIP is a DELETE of exactly this (ns, id).
  /// /planning/astar_path carries a single (ns, id), so the two are equivalent
  /// on this topic today; they stop being equivalent the moment anything else
  /// shares it, which is the direction that favours DELETE.
  void clear(const Frame & frame)
  {
    publisher_->publish(
      line_strip(
        frame.frame_id, frame.stamp, kMarkerNs, 0, nullptr, 0, kPathColor, path_width_,
        params_.marker_lifetime_sec));
  }

  CommonParams params_;
  double goal_x_ = 25.0;
  double goal_y_ = 0.0;
  double goal_search_radius_ = 5.0;
  double start_search_radius_ = 2.0;
  int connectivity_ = 8;
  double path_width_ = 0.15;
  double axis_x_ = 1.0;
  double axis_y_ = 0.0;
  std::unique_ptr<rclcpp::Logger> logger_;
  rclcpp::Clock::SharedPtr clock_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr publisher_;
};

}  // namespace

std::unique_ptr<FrameConsumer> make_astar_planner()
{
  return std::make_unique<AStarPlanner>();
}

}  // namespace ransac_seg_cpp
