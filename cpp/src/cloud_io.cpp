#include "ransac_seg_cpp/cloud_io.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include <sensor_msgs/msg/point_field.hpp>

namespace ransac_seg_cpp
{

namespace
{

constexpr uint32_t kPointStep = 12;  // three float32 components, tightly packed
constexpr const char * kXyzFields[3] = {"x", "y", "z"};

/// Offset and datatype of one named field.
struct FieldRef
{
  uint32_t offset;
  uint8_t datatype;
};

FieldRef find_field(const sensor_msgs::msg::PointCloud2 & msg, const char * name)
{
  for (const auto & field : msg.fields) {
    if (field.name == name) {
      if (field.datatype != sensor_msgs::msg::PointField::FLOAT32 &&
        field.datatype != sensor_msgs::msg::PointField::FLOAT64)
      {
        throw std::invalid_argument(
                std::string("PointCloud2 field '") + name +
                "' must be FLOAT32 or FLOAT64, got datatype " +
                std::to_string(static_cast<int>(field.datatype)));
      }
      return {field.offset, field.datatype};
    }
  }
  throw std::invalid_argument(std::string("PointCloud2 has no '") + name + "' field");
}

/// Read one float32/float64 component. The caller has already bounds-checked.
double read_component(const uint8_t * point, const FieldRef & ref)
{
  if (ref.datatype == sensor_msgs::msg::PointField::FLOAT32) {
    float value = 0.0f;
    std::memcpy(&value, point + ref.offset, sizeof(value));
    return static_cast<double>(value);
  }
  double value = 0.0;
  std::memcpy(&value, point + ref.offset, sizeof(value));
  return value;
}

std::size_t component_size(const FieldRef & ref)
{
  return ref.datatype == sensor_msgs::msg::PointField::FLOAT32 ? 4u : 8u;
}

}  // namespace

std::vector<float> pointcloud2_to_xyz(const sensor_msgs::msg::PointCloud2 & msg, bool skip_nans)
{
  std::vector<float> xyz;
  if (msg.width == 0 || msg.height == 0) {
    return xyz;
  }
  if (msg.is_bigendian) {
    throw std::invalid_argument(
            "PointCloud2 is big-endian; this decoder only handles little-endian payloads");
  }

  FieldRef refs[3];
  for (int c = 0; c < 3; ++c) {
    refs[c] = find_field(msg, kXyzFields[c]);
  }

  const std::size_t count = static_cast<std::size_t>(msg.width) * msg.height;
  const std::size_t step = msg.point_step;
  for (int c = 0; c < 3; ++c) {
    if (refs[c].offset + component_size(refs[c]) > step) {
      throw std::invalid_argument(
              std::string("PointCloud2 field '") + kXyzFields[c] + "' runs past point_step");
    }
  }
  if (step == 0 || msg.data.size() < count * step) {
    throw std::invalid_argument("PointCloud2 data is shorter than width * height * point_step");
  }

  // Both halves of this condition are read_points_numpy's, not a simplification
  // of it. It masks only when `skip_nans and not cloud.is_dense`, so a driver
  // asserting is_dense keeps every row it packed, NaNs included — and it masks
  // with ~np.isnan, so +/-inf survives either way. Filtering unconditionally on
  // isfinite fed the two stacks different point sets from the same message.
  const bool drop_nans = skip_nans && !msg.is_dense;

  xyz.reserve(count * 3);
  const uint8_t * base = msg.data.data();
  for (std::size_t i = 0; i < count; ++i) {
    const uint8_t * point = base + i * step;
    const double x = read_component(point, refs[0]);
    const double y = read_component(point, refs[1]);
    const double z = read_component(point, refs[2]);
    if (drop_nans && (std::isnan(x) || std::isnan(y) || std::isnan(z))) {
      continue;
    }
    xyz.push_back(static_cast<float>(x));
    xyz.push_back(static_cast<float>(y));
    xyz.push_back(static_cast<float>(z));
  }
  return xyz;
}

sensor_msgs::msg::PointCloud2 xyz_to_pointcloud2(
  const float * xyz, std::size_t count, const std::string & frame_id,
  const builtin_interfaces::msg::Time & stamp)
{
  sensor_msgs::msg::PointCloud2 msg;
  msg.header.frame_id = frame_id;
  msg.header.stamp = stamp;
  msg.height = 1;
  msg.width = static_cast<uint32_t>(count);

  msg.fields.resize(3);
  for (int c = 0; c < 3; ++c) {
    msg.fields[c].name = kXyzFields[c];
    msg.fields[c].offset = static_cast<uint32_t>(4 * c);
    msg.fields[c].datatype = sensor_msgs::msg::PointField::FLOAT32;
    msg.fields[c].count = 1;
  }

  msg.is_bigendian = false;
  msg.point_step = kPointStep;
  msg.row_step = kPointStep * msg.width;
  msg.is_dense = true;
  msg.data.resize(count * kPointStep);
  if (count != 0) {
    // One memcpy: the payload layout is already exactly the caller's array.
    std::memcpy(msg.data.data(), xyz, count * kPointStep);
  }
  return msg;
}

}  // namespace ransac_seg_cpp
