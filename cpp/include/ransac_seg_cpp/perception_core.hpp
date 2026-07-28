// One decode/segment/rasterise pipeline, many consumers.
//
// The Python launch file starts six processes and each of them subscribes to
// the same PointCloud2 and independently runs pointcloud2_to_xyz -> range gate
// -> segment_ground -> build_costmap. segment_ground alone is p95 8.30 ms in
// this port, so five sixths of that work is duplication of a result that is
// bit-identical across the six copies. PerceptionHub does it once per sweep and
// hands the same const Frame & to every registered consumer.
//
// The consumer interface is deliberately narrow: a consumer declares its own
// parameters, owns its own publishers, and reads a frame. It never touches the
// subscription, the plane filter or the costmap, which is what stopped the six
// Python copies from drifting apart in the first place.

#ifndef RANSAC_SEG_CPP__PERCEPTION_CORE_HPP_
#define RANSAC_SEG_CPP__PERCEPTION_CORE_HPP_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "ransac_seg_cpp/cost_grid.hpp"
#include "ransac_seg_cpp/ground_plane.hpp"
#include "ransac_seg_cpp/params.hpp"
#include "ransac_seg_cpp/plane_filter.hpp"

namespace ransac_seg_cpp
{

/// The QoS a LiDAR subscription needs.
///
/// rclcpp's default is RELIABLE, and DDS refuses to match a RELIABLE
/// subscription with a BEST_EFFORT publisher — which is what every production
/// LiDAR driver, and `ros2 bag play` of data recorded from one, offers. The
/// node then sits there receiving nothing, with only a QoS-incompatibility line
/// in the log to say why. Requesting BEST_EFFORT matches publishers of either
/// kind.
rclcpp::QoS sensor_qos(int depth);

/// The QoS every marker publisher uses: KEEP_LAST(depth), RELIABLE, VOLATILE.
///
/// NOT sensor_qos. sensor_qos exists for the LiDAR SUBSCRIPTION, and the Python
/// uses it in exactly that one place too (perception_node.sensor_qos, called
/// only from create_subscription); every Python publisher is created with a
/// bare depth, which rclpy resolves to RELIABLE. Publishing markers BEST_EFFORT
/// instead does not merely risk a dropped frame — RViz2's marker display
/// requests RELIABLE by default, DDS refuses that match, and a stock RViz2
/// config receives nothing at all from every marker topic in the package.
rclcpp::QoS marker_qos(int depth);

/// One decoded, segmented LiDAR frame plus the costmap built from it.
///
/// The point arrays are references into buffers the hub owns for the duration
/// of the callback: a consumer may read them but must not retain them past
/// on_frame().
struct Frame
{
  std::string frame_id;
  builtin_interfaces::msg::Time stamp;
  const std::vector<float> & points;         ///< 3*N, range-gated
  const std::vector<uint8_t> & ground_mask;  ///< N, 1 = ground
  const std::vector<float> & ground_xyz;     ///< 3*Ng
  const std::vector<float> & obstacle_xyz;   ///< 3*No
  OptPlane plane;                            ///< filtered
  std::optional<PlaneModel> raw_plane;       ///< absent => the height cut ran
  /// Null means "publish a clear marker", never "publish an empty ADD": an ADD
  /// holding zero points reads as "no change" and leaves the previous frame's
  /// geometry on screen.
  std::shared_ptr<const CostGrid> grid;
};

/// One visualiser or planner reading the shared frame.
class FrameConsumer
{
public:
  virtual ~FrameConsumer() = default;

  /// Parameter prefix, e.g. "astar"; also what the hub logs it as.
  virtual std::string name() const = 0;

  /// False lets the hub skip the rasterise when NO consumer wants a costmap.
  virtual bool needs_costmap() const {return true;}

  /// Declare parameters and create publishers. Called once, before start().
  virtual void configure(rclcpp::Node & node, const CommonParams & params) = 0;

  virtual void on_frame(const Frame & frame) = 0;

  /// The shared parameters changed at runtime and were accepted.
  virtual void on_params_changed(const CommonParams & params) {(void)params;}
};

/// Node carrying the shared pipeline for a set of consumers.
class PerceptionHub : public rclcpp::Node
{
public:
  explicit PerceptionHub(
    const std::string & node_name,
    const std::string & default_marker_topic = "/semantic_map/squares");

  /// Register a consumer. Must be called before start().
  void add_consumer(std::unique_ptr<FrameConsumer> consumer);

  /// Configure the consumers, subscribe, and start accepting parameter sets.
  void start();

  const CommonParams & params() const {return params_;}

private:
  void on_cloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg);
  rcl_interfaces::msg::SetParametersResult on_set_parameters(
    const std::vector<rclcpp::Parameter> & request);
  std::vector<float> range_gate(const std::vector<float> & points) const;

  CommonParams params_;
  int min_ground_points_frame_ = 30;
  PlaneFilter plane_filter_;
  std::vector<std::unique_ptr<FrameConsumer>> consumers_;
  bool any_needs_costmap_ = false;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_;
};

}  // namespace ransac_seg_cpp

#endif  // RANSAC_SEG_CPP__PERCEPTION_CORE_HPP_
