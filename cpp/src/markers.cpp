#include "ransac_seg_cpp/markers.hpp"

#include <stdexcept>
#include <string>

#include <geometry_msgs/msg/point.hpp>

namespace ransac_seg_cpp
{

namespace
{

using visualization_msgs::msg::Marker;

/// Header, namespace and action — everything a bare clear marker needs.
Marker bare(
  const std::string & frame_id, const builtin_interfaces::msg::Time & stamp,
  const std::string & ns, int32_t action)
{
  Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = stamp;
  marker.ns = ns;
  marker.action = action;
  return marker;
}

}  // namespace

Marker make_marker(
  const std::string & frame_id, const builtin_interfaces::msg::Time & stamp,
  const std::string & ns, int marker_id, int32_t marker_type, const Color & color,
  const std::array<double, 3> & scale, double lifetime_sec, int32_t action)
{
  Marker marker = bare(frame_id, stamp, ns, action);
  marker.id = marker_id;
  marker.type = marker_type;
  marker.scale.x = scale[0];
  marker.scale.y = scale[1];
  marker.scale.z = scale[2];
  marker.color.r = static_cast<float>(color[0]);
  marker.color.g = static_cast<float>(color[1]);
  marker.color.b = static_cast<float>(color[2]);
  marker.color.a = static_cast<float>(color[3]);
  marker.pose.orientation.w = 1.0;
  // A lifetime of 0.0 means "forever" to ROS, and it is the default-constructed
  // value, so it is left untouched rather than written back as an explicit zero.
  if (lifetime_sec > 0.0) {
    const int32_t whole = static_cast<int32_t>(lifetime_sec);
    marker.lifetime.sec = whole;
    marker.lifetime.nanosec = static_cast<uint32_t>((lifetime_sec - whole) * 1e9);
  }
  return marker;
}

void add_points(Marker & marker, const double * xyz, std::size_t count)
{
  if (xyz == nullptr || count == 0) {
    return;
  }
  marker.points.reserve(marker.points.size() + count);
  for (std::size_t i = 0; i < count; ++i) {
    geometry_msgs::msg::Point point;
    point.x = xyz[i * 3];
    point.y = xyz[i * 3 + 1];
    point.z = xyz[i * 3 + 2];
    marker.points.push_back(point);
  }
}

Marker delete_marker(
  const std::string & frame_id, const builtin_interfaces::msg::Time & stamp,
  const std::string & ns, int marker_id)
{
  Marker marker = bare(frame_id, stamp, ns, Marker::DELETE);
  marker.id = marker_id;
  return marker;
}

Marker delete_all(
  const std::string & frame_id, const builtin_interfaces::msg::Time & stamp,
  const std::string & ns)
{
  return bare(frame_id, stamp, ns, Marker::DELETEALL);
}

Marker cube_list(
  const std::string & frame_id, const builtin_interfaces::msg::Time & stamp,
  const std::string & ns, int marker_id, const double * centres_xyz, std::size_t count,
  double cell_size, const Color & color, double height, double lifetime_sec)
{
  if (count == 0) {
    return delete_marker(frame_id, stamp, ns, marker_id);
  }
  Marker marker = make_marker(
    frame_id, stamp, ns, marker_id, Marker::CUBE_LIST, color,
    {cell_size, cell_size, height}, lifetime_sec);
  add_points(marker, centres_xyz, count);
  return marker;
}

Marker line_strip(
  const std::string & frame_id, const builtin_interfaces::msg::Time & stamp,
  const std::string & ns, int marker_id, const double * points_xyz, std::size_t count,
  const Color & color, double width, double lifetime_sec)
{
  if (count < 2) {
    return delete_marker(frame_id, stamp, ns, marker_id);
  }
  Marker marker = make_marker(
    frame_id, stamp, ns, marker_id, Marker::LINE_STRIP, color, {width, 0.0, 0.0},
    lifetime_sec);
  add_points(marker, points_xyz, count);
  return marker;
}

visualization_msgs::msg::MarkerArray box_array(
  const std::string & frame_id, const builtin_interfaces::msg::Time & stamp,
  const std::string & ns, int first_id, const std::vector<std::array<double, 3>> & centres,
  const std::vector<std::array<double, 3>> & extents, const Color & color,
  double lifetime_sec)
{
  visualization_msgs::msg::MarkerArray array;
  array.markers.push_back(delete_all(frame_id, stamp, ns));
  if (centres.empty()) {
    return array;
  }
  if (extents.size() != 1 && extents.size() != centres.size()) {
    throw std::invalid_argument(
            "box_array needs one extent or one per centre, got " +
            std::to_string(extents.size()) + " for " + std::to_string(centres.size()) +
            " centres");
  }

  array.markers.reserve(centres.size() + 1);
  for (std::size_t i = 0; i < centres.size(); ++i) {
    const std::array<double, 3> & extent = extents.size() == 1 ? extents[0] : extents[i];
    Marker marker = make_marker(
      frame_id, stamp, ns, first_id + static_cast<int>(i), Marker::CUBE, color, extent,
      lifetime_sec);
    // Centres are poses here, not marker.points: one CUBE per box.
    marker.pose.position.x = centres[i][0];
    marker.pose.position.y = centres[i][1];
    marker.pose.position.z = centres[i][2];
    array.markers.push_back(marker);
  }
  return array;
}

}  // namespace ransac_seg_cpp
