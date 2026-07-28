// Marker constructors shared by every visualising node — port of
// ransac_seg/markers.py.
//
// Each node used to hand-roll its own Marker scaffolding (header, namespace,
// id, colour, scale), which drifted apart field by field. Beyond the
// duplication that produced a real display bug: a frame with nothing to draw
// published an ADD marker holding zero points, which RViz and Foxglove treat as
// "no change", so the previous frame's geometry stayed on screen indefinitely.
//
// Clearing is therefore explicit here, and deliberately NARROW. A fixed-id
// constructor clears with a DELETE of its own (ns, id) rather than a DELETEALL,
// because consumers implement DELETEALL as "wipe this display", not "wipe this
// namespace": the ground-segmentation node draws three namespaces on one topic,
// and a DELETEALL for the empty obstacle layer erased the ground layer
// published microseconds earlier in the same frame.

#ifndef RANSAC_SEG_CPP__MARKERS_HPP_
#define RANSAC_SEG_CPP__MARKERS_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace ransac_seg_cpp
{

/// (r, g, b, a), each in [0, 1].
using Color = std::array<double, 4>;

/// Build a Marker with header, identity, colour and scale filled in.
///
/// `scale` is (x, y, z). A `lifetime_sec` of 0.0 means "forever" and leaves the
/// lifetime field untouched — ROS reads an unset lifetime as never expiring.
visualization_msgs::msg::Marker make_marker(
  const std::string & frame_id, const builtin_interfaces::msg::Time & stamp,
  const std::string & ns, int marker_id, int32_t marker_type, const Color & color,
  const std::array<double, 3> & scale, double lifetime_sec = 0.0,
  int32_t action = visualization_msgs::msg::Marker::ADD);

/// Append `count` xyz triples (a `count * 3` array) to a marker's point list.
void add_points(visualization_msgs::msg::Marker & marker, const double * xyz, std::size_t count);

/// Clear exactly one previously published (ns, marker_id).
///
/// Named `delete_marker` because `delete` is a C++ keyword; the Python calls it
/// `delete`.
visualization_msgs::msg::Marker delete_marker(
  const std::string & frame_id, const builtin_interfaces::msg::Time & stamp,
  const std::string & ns, int marker_id);

/// Clear the consumer's whole display.
///
/// Despite the namespace field, RViz2 and Foxglove act on DELETEALL globally,
/// so only publish this on a topic that carries a single namespace. To clear
/// one marker use delete_marker instead.
visualization_msgs::msg::Marker delete_all(
  const std::string & frame_id, const builtin_interfaces::msg::Time & stamp,
  const std::string & ns);

/// A CUBE_LIST of equally sized tiles centred on `centres_xyz` (count * 3).
///
/// Returns a DELETE of this (ns, marker_id) when there are no centres, so an
/// empty frame erases its own layer — and only its own layer — instead of
/// leaving it on screen. Scale is (cell_size, cell_size, height).
visualization_msgs::msg::Marker cube_list(
  const std::string & frame_id, const builtin_interfaces::msg::Time & stamp,
  const std::string & ns, int marker_id, const double * centres_xyz, std::size_t count,
  double cell_size, const Color & color, double height = 0.05, double lifetime_sec = 0.0);

/// A LINE_STRIP through `points_xyz` (count * 3). Scale is (width, 0, 0).
///
/// Fewer than two points cannot form a line, and an empty ADD marker would
/// leave the previous path on screen, so that case returns a DELETE of this
/// (ns, marker_id).
visualization_msgs::msg::Marker line_strip(
  const std::string & frame_id, const builtin_interfaces::msg::Time & stamp,
  const std::string & ns, int marker_id, const double * points_xyz, std::size_t count,
  const Color & color, double width = 0.1, double lifetime_sec = 0.0);

/// A MarkerArray of CUBE markers, one per centre, ids from `first_id`.
///
/// `extents` is either a single (x, y, z) size broadcast to every box or one
/// size per centre; any other length throws std::invalid_argument. Centres are
/// marker POSES here, not marker points.
///
/// The array ALWAYS leads with a DELETEALL, even when there are boxes, because
/// the id range is as long as the cluster list and the previous frame's may
/// have been longer; clearing only on an empty list left the surplus ids on
/// screen forever under the default lifetime of 0.0 ("never expire"). The
/// consequence is that box_array MUST OWN ITS TOPIC — publish nothing else on
/// it, or the DELETEALL will wipe that too.
visualization_msgs::msg::MarkerArray box_array(
  const std::string & frame_id, const builtin_interfaces::msg::Time & stamp,
  const std::string & ns, int first_id, const std::vector<std::array<double, 3>> & centres,
  const std::vector<std::array<double, 3>> & extents, const Color & color,
  double lifetime_sec = 0.0);

}  // namespace ransac_seg_cpp

#endif  // RANSAC_SEG_CPP__MARKERS_HPP_
