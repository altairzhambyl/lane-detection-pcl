// PointCloud2 <-> flat xyz conversion — C++ port of ransac_seg/cloud_io.py.
//
// The field offsets and datatypes are read out of msg.fields rather than
// assumed: real drivers pack xyzi, xyzir, and organised clouds, and the x/y/z
// members are not always the first twelve bytes of a point.
//
// DECLARED DEVIATION from cloud_io.py: the Python returns float64 because its
// RANSAC upcasts anyway and converting once there is cheaper than letting numpy
// upcast per iteration. This returns float32, per the project's storage rule —
// that is how the PointCloud2 payload and a KITTI .bin already sit in memory,
// fit_plane_ransac takes a `const float *`, and every maths path converts to
// double at the point of use. The values are bit-identical for FLOAT32 sources
// (the overwhelmingly common case) and rounded once for FLOAT64 ones.

#ifndef RANSAC_SEG_CPP__CLOUD_IO_HPP_
#define RANSAC_SEG_CPP__CLOUD_IO_HPP_

#include <cstddef>
#include <string>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace ransac_seg_cpp
{

/// Tightly packed x,y,z triples (3*N floats) read out of a PointCloud2.
///
/// Organised clouds are flattened. Returns empty when width or height is 0.
///
/// `skip_nans` is read_points_numpy's, with both of its conditions: a row is
/// dropped only when the message ALSO declares itself non-dense (is_dense
/// false), and only when one of x/y/z is NaN — +/-infinity is kept. A driver
/// that packs NaNs into a sweep while asserting is_dense therefore decodes with
/// them still in, exactly as the Python does. What keeps them out of the fit is
/// PerceptionHub::range_gate, which rejects a non-finite radius, and that is
/// true of both stacks.
///
/// Throws std::invalid_argument when x, y or z is missing, when one of them is
/// neither FLOAT32 nor FLOAT64, when a point would read past the payload, or
/// when the message is big-endian — no production driver emits big-endian and
/// silently mis-decoding it is worse than refusing the frame.
std::vector<float> pointcloud2_to_xyz(
  const sensor_msgs::msg::PointCloud2 & msg, bool skip_nans = true);

/// Pack `count` xyz triples into an unorganised, dense, little-endian cloud.
///
/// height 1, point_step 12, three FLOAT32 fields at offsets 0/4/8.
sensor_msgs::msg::PointCloud2 xyz_to_pointcloud2(
  const float * xyz, std::size_t count, const std::string & frame_id,
  const builtin_interfaces::msg::Time & stamp);

}  // namespace ransac_seg_cpp

#endif  // RANSAC_SEG_CPP__CLOUD_IO_HPP_
