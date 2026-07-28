// Ground, obstacle and inflation visualiser — port of ransac_seg/ransac_node.py.
//
// The Python this replaces quantised ground returns into a set of cell keys and
// drew them at z=0.0 under a namespace curve.py also published on, so in RViz
// whichever node ran last silently erased the other, and it had no obstacle
// channel at all. Here the three costmap states are separate namespaces, the
// clustered obstacles get their own boxes on their own topic, and every marker
// sits on the fitted plane: on the validated 10 degree slope, pinning geometry
// to z=0.0 puts it metres off the surface at the far end of the sweep.

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "ransac_seg_cpp/cluster.hpp"
#include "ransac_seg_cpp/consumers.hpp"
#include "ransac_seg_cpp/markers.hpp"
#include "ransac_seg_cpp/ros_params.hpp"

namespace ransac_seg_cpp
{
namespace
{

constexpr const char * kGroundNs = "ground";
constexpr const char * kObstacleNs = "obstacles";
constexpr const char * kInflationNs = "inflation";
constexpr const char * kClusterNs = "obstacle_boxes";

/// Cell centres of one state, lifted onto the fitted plane, as a flat xyz array.
std::vector<double> layer_centres(const CostGrid & grid, int8_t state)
{
  const std::vector<std::array<int32_t, 2>> cells = grid.cells_with_state(state);
  std::vector<std::array<int64_t, 2>> widened;
  widened.reserve(cells.size());
  for (const std::array<int32_t, 2> & cell : cells) {
    widened.push_back({cell[0], cell[1]});
  }
  const std::vector<std::array<double, 3>> world = grid.cell_to_world_xyz(widened);
  std::vector<double> flat;
  flat.reserve(world.size() * 3);
  for (const std::array<double, 3> & point : world) {
    flat.insert(flat.end(), point.begin(), point.end());
  }
  return flat;
}

class GroundSegmentation : public FrameConsumer
{
public:
  std::string name() const override {return "ground";}
  bool needs_costmap() const override {return true;}

  void configure(rclcpp::Node & node, const CommonParams & params) override
  {
    params_ = params;
    marker_alpha_ = declare(
      node, "ground.marker_alpha", 0.5, "Opacity of the ground and obstacle tiles");
    inflation_alpha_ = declare(
      node, "ground.inflation_alpha", 0.25,
      "Opacity of the inflation tiles; kept low so they read as a margin");
    cluster_.voxel_size = declare(
      node, "ground.cluster_voxel_size", 0.2, "Voxel size used to link obstacles");
    cluster_.link_radius = declare(
      node, "ground.cluster_link_radius", 0.6,
      "Returns within this distance join the same obstacle");
    cluster_.min_points = static_cast<int>(
      declare(
        node, "ground.cluster_min_points", static_cast<int64_t>(12),
        "Smallest reportable obstacle"));
    cluster_.min_height = declare(
      node, "ground.cluster_min_height", 0.15,
      "Clusters shorter than this are ground scraps, not obstacles");
    cluster_.max_height = declare(
      node, "ground.cluster_max_height", 4.0,
      "Clusters taller than this are ceilings or overhangs");
    const std::string boxes_topic = declare(
      node, "ground.obstacle_marker_topic", std::string("/semantic_map/obstacles"),
      "Topic for the per-obstacle bounding boxes");

    // The tiles ride the SHARED marker topic, exactly as in Python: this
    // consumer's own default IS the package default, so an untouched launch
    // file puts them on /semantic_map/squares.
    pub_cells_ = node.create_publisher<visualization_msgs::msg::Marker>(
      params.marker_topic, marker_qos(params.qos_depth));
    // box_array leads with a DELETEALL, which consumers act on globally, so it
    // must own its topic and cannot share the tiles'.
    pub_boxes_ = node.create_publisher<visualization_msgs::msg::MarkerArray>(
      boxes_topic, marker_qos(params.qos_depth));
  }

  void on_params_changed(const CommonParams & params) override {params_ = params;}

  void on_frame(const Frame & frame) override
  {
    if (frame.grid == nullptr) {
      clear(frame);
      return;
    }
    const CostGrid & grid = *frame.grid;
    const struct
    {
      const char * ns;
      int id;
      int8_t state;
      Color color;
    } layers[] = {
      {kGroundNs, 0, FREE, {0.0, 1.0, 0.0, marker_alpha_}},
      {kObstacleNs, 1, OCCUPIED, {1.0, 0.0, 0.0, marker_alpha_}},
      {kInflationNs, 2, INFLATED, {1.0, 0.55, 0.0, inflation_alpha_}},
    };
    for (const auto & layer : layers) {
      const std::vector<double> centres = layer_centres(grid, layer.state);
      pub_cells_->publish(
        cube_list(
          frame.frame_id, frame.stamp, layer.ns, layer.id, centres.data(), centres.size() / 3,
          grid.resolution, layer.color, 0.05, params_.marker_lifetime_sec));
    }

    const std::vector<Cluster> clusters =
      cluster_obstacles(frame.obstacle_xyz.data(), frame.obstacle_xyz.size() / 3, 3, cluster_);
    std::vector<std::array<double, 3>> centres;
    std::vector<std::array<double, 3>> extents;
    centres.reserve(clusters.size());
    extents.reserve(clusters.size());
    for (const Cluster & cluster : clusters) {
      centres.push_back({cluster.centroid.x(), cluster.centroid.y(), cluster.centroid.z()});
      const Eigen::Vector3d extent = cluster.extent();
      extents.push_back({extent.x(), extent.y(), extent.z()});
    }
    pub_boxes_->publish(
      box_array(
        frame.frame_id, frame.stamp, kClusterNs, 0, centres, extents, {1.0, 0.3, 0.0, 0.6},
        params_.marker_lifetime_sec));
  }

private:
  /// Retire this consumer's geometry without touching anything else on the topic.
  ///
  /// DECLARED DEVIATION from ransac_node.py, which published a DELETEALL per
  /// namespace here: RViz2 and Foxglove act on DELETEALL globally, so three of
  /// them on a topic carrying three namespaces is three whole-display wipes.
  /// An empty cube_list is a DELETE of exactly this (ns, id), which clears the
  /// same geometry and nothing else.
  void clear(const Frame & frame)
  {
    const std::pair<const char *, int> layers[] = {
      {kGroundNs, 0}, {kObstacleNs, 1}, {kInflationNs, 2}};
    for (const auto & [ns, id] : layers) {
      pub_cells_->publish(
        cube_list(
          frame.frame_id, frame.stamp, ns, id, nullptr, 0, params_.resolution,
          {0.0, 0.0, 0.0, 0.0}, 0.05, params_.marker_lifetime_sec));
    }
    pub_boxes_->publish(
      box_array(
        frame.frame_id, frame.stamp, kClusterNs, 0, {}, {}, {1.0, 0.3, 0.0, 0.6},
        params_.marker_lifetime_sec));
  }

  CommonParams params_;
  double marker_alpha_ = 0.5;
  double inflation_alpha_ = 0.25;
  ClusterOptions cluster_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_cells_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_boxes_;
};

}  // namespace

std::unique_ptr<FrameConsumer> make_ground_segmentation()
{
  return std::make_unique<GroundSegmentation>();
}

}  // namespace ransac_seg_cpp
