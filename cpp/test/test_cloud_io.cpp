// Mirrors test/test_cloud_io.py plus the layouts a real driver emits: xyzi,
// FLOAT64 xyz, and an organised cloud that has to flatten.

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ransac_seg_cpp/cloud_io.hpp"

namespace
{

using ransac_seg_cpp::pointcloud2_to_xyz;
using ransac_seg_cpp::xyz_to_pointcloud2;
using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

sensor_msgs::msg::PointField make_field(
  const std::string & name, uint32_t offset, uint8_t datatype)
{
  sensor_msgs::msg::PointField field;
  field.name = name;
  field.offset = offset;
  field.datatype = datatype;
  field.count = 1;
  return field;
}

/// An xyzi cloud: xyz are not the only fields, and i sits between them and the
/// end of the point, which is why offsets have to be read rather than assumed.
PointCloud2 make_xyzi(const std::vector<float> & xyzi, uint32_t width, uint32_t height = 1)
{
  PointCloud2 msg;
  msg.height = height;
  msg.width = width;
  msg.fields = {
    make_field("x", 0, PointField::FLOAT32),
    make_field("y", 4, PointField::FLOAT32),
    make_field("z", 8, PointField::FLOAT32),
    make_field("intensity", 12, PointField::FLOAT32)};
  msg.is_bigendian = false;
  msg.point_step = 16;
  msg.row_step = 16 * width;
  msg.is_dense = true;
  msg.data.resize(xyzi.size() * sizeof(float));
  std::memcpy(msg.data.data(), xyzi.data(), msg.data.size());
  return msg;
}

TEST(CloudIo, EmptyCloudsReturnNothing)
{
  PointCloud2 msg = make_xyzi({}, 0);
  EXPECT_TRUE(pointcloud2_to_xyz(msg).empty());
  msg.width = 4;
  msg.height = 0;
  EXPECT_TRUE(pointcloud2_to_xyz(msg).empty());
}

TEST(CloudIo, ReadsXyzOutOfAnXyziLayout)
{
  const PointCloud2 msg = make_xyzi({1, 2, 3, 99, 4, 5, 6, 98}, 2);
  const std::vector<float> xyz = pointcloud2_to_xyz(msg);
  ASSERT_EQ(xyz.size(), 6u);
  EXPECT_FLOAT_EQ(xyz[0], 1.0f);
  EXPECT_FLOAT_EQ(xyz[3], 4.0f);
  EXPECT_FLOAT_EQ(xyz[5], 6.0f);
}

TEST(CloudIo, ReadsFieldsInDeclaredOrderNotStorageOrder)
{
  // A driver that packs z, y, x must still come back as x, y, z.
  PointCloud2 msg = make_xyzi({1, 2, 3, 0}, 1);
  msg.fields = {
    make_field("z", 0, PointField::FLOAT32),
    make_field("y", 4, PointField::FLOAT32),
    make_field("x", 8, PointField::FLOAT32)};
  const std::vector<float> xyz = pointcloud2_to_xyz(msg);
  ASSERT_EQ(xyz.size(), 3u);
  EXPECT_FLOAT_EQ(xyz[0], 3.0f);
  EXPECT_FLOAT_EQ(xyz[2], 1.0f);
}

TEST(CloudIo, OrganisedCloudsFlatten)
{
  const PointCloud2 msg = make_xyzi({1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0, 10, 11, 12, 0}, 2, 2);
  const std::vector<float> xyz = pointcloud2_to_xyz(msg);
  EXPECT_EQ(xyz.size(), 12u);
  EXPECT_FLOAT_EQ(xyz[9], 10.0f);
}

TEST(CloudIo, SkipNansDropsWholeRows)
{
  // is_dense false, like every cloud sensor_msgs_py.point_cloud2.create_cloud
  // builds — which is what test_cloud_io.py's own fixture uses.
  const float nan = std::numeric_limits<float>::quiet_NaN();
  PointCloud2 msg = make_xyzi({1, 2, 3, 0, 4, nan, 6, 0, 7, 8, 9, 0}, 3);
  msg.is_dense = false;

  const std::vector<float> kept = pointcloud2_to_xyz(msg, true);
  ASSERT_EQ(kept.size(), 6u);
  EXPECT_FLOAT_EQ(kept[0], 1.0f);
  EXPECT_FLOAT_EQ(kept[3], 7.0f);

  const std::vector<float> all = pointcloud2_to_xyz(msg, false);
  EXPECT_EQ(all.size(), 9u);
  EXPECT_TRUE(std::isnan(all[4]));
}

TEST(CloudIo, ADenseCloudKeepsItsNansEvenWithSkipNans)
{
  // read_points_numpy masks only when `skip_nans and not cloud.is_dense`, so a
  // driver packing NaNs into an organised sweep while asserting is_dense hands
  // both stacks the same rows. Filtering here regardless used to hand the C++
  // two points where the Python got four.
  const float nan = std::numeric_limits<float>::quiet_NaN();
  PointCloud2 msg = make_xyzi({1, 2, 3, 0, nan, 5, 6, 0, 7, 8, 9, 0}, 3);
  msg.is_dense = true;

  const std::vector<float> kept = pointcloud2_to_xyz(msg, true);
  ASSERT_EQ(kept.size(), 9u);
  EXPECT_TRUE(std::isnan(kept[3]));
  EXPECT_FLOAT_EQ(kept[4], 5.0f);
}

TEST(CloudIo, InfinityIsNotANanAndSurvivesSkipNans)
{
  // The mask is ~np.isnan, not np.isfinite. An infinite return is a range
  // overflow the gate rejects downstream, not a hole in the sweep.
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  PointCloud2 msg = make_xyzi({1, 2, 3, 0, 4, inf, 6, 0, nan, 8, 9, 0}, 3);
  msg.is_dense = false;

  const std::vector<float> kept = pointcloud2_to_xyz(msg, true);
  ASSERT_EQ(kept.size(), 6u);
  EXPECT_FLOAT_EQ(kept[0], 1.0f);
  EXPECT_FLOAT_EQ(kept[3], 4.0f);
  EXPECT_TRUE(std::isinf(kept[4]));
}

TEST(CloudIo, SupportsFloat64Fields)
{
  PointCloud2 msg;
  msg.height = 1;
  msg.width = 1;
  msg.fields = {
    make_field("x", 0, PointField::FLOAT64),
    make_field("y", 8, PointField::FLOAT64),
    make_field("z", 16, PointField::FLOAT64)};
  msg.is_bigendian = false;
  msg.point_step = 24;
  msg.row_step = 24;
  msg.data.resize(24);
  const double values[3] = {1.5, -2.25, 3.125};
  std::memcpy(msg.data.data(), values, sizeof(values));

  const std::vector<float> xyz = pointcloud2_to_xyz(msg);
  ASSERT_EQ(xyz.size(), 3u);
  EXPECT_FLOAT_EQ(xyz[0], 1.5f);
  EXPECT_FLOAT_EQ(xyz[1], -2.25f);
  EXPECT_FLOAT_EQ(xyz[2], 3.125f);
}

TEST(CloudIo, RejectsMissingFieldsUnsupportedTypesAndBigEndian)
{
  PointCloud2 missing = make_xyzi({1, 2, 3, 0}, 1);
  missing.fields.erase(missing.fields.begin() + 2);  // drop z
  EXPECT_THROW(pointcloud2_to_xyz(missing), std::invalid_argument);
  try {
    pointcloud2_to_xyz(missing);
  } catch (const std::invalid_argument & error) {
    EXPECT_STREQ(error.what(), "PointCloud2 has no 'z' field");
  }

  PointCloud2 wrong_type = make_xyzi({1, 2, 3, 0}, 1);
  wrong_type.fields[1].datatype = PointField::INT32;
  EXPECT_THROW(pointcloud2_to_xyz(wrong_type), std::invalid_argument);

  PointCloud2 big = make_xyzi({1, 2, 3, 0}, 1);
  big.is_bigendian = true;
  EXPECT_THROW(pointcloud2_to_xyz(big), std::invalid_argument);

  PointCloud2 truncated = make_xyzi({1, 2, 3, 0}, 1);
  truncated.data.resize(8);
  EXPECT_THROW(pointcloud2_to_xyz(truncated), std::invalid_argument);
}

TEST(CloudIo, RoundTripsThroughXyzToPointcloud2)
{
  const std::vector<float> points{1.0f, 2.0f, 3.0f, -4.5f, 5.25f, 6.125f};
  builtin_interfaces::msg::Time stamp;
  stamp.sec = 7;
  stamp.nanosec = 8;

  const PointCloud2 msg = xyz_to_pointcloud2(points.data(), 2, "base_link", stamp);
  EXPECT_EQ(msg.header.frame_id, "base_link");
  EXPECT_EQ(msg.header.stamp.sec, 7);
  EXPECT_EQ(msg.header.stamp.nanosec, 8u);
  EXPECT_EQ(msg.height, 1u);
  EXPECT_EQ(msg.width, 2u);
  EXPECT_EQ(msg.point_step, 12u);
  EXPECT_EQ(msg.row_step, 24u);
  EXPECT_FALSE(msg.is_bigendian);
  EXPECT_TRUE(msg.is_dense);
  ASSERT_EQ(msg.fields.size(), 3u);
  EXPECT_EQ(msg.fields[0].name, "x");
  EXPECT_EQ(msg.fields[2].offset, 8u);
  EXPECT_EQ(msg.fields[2].datatype, PointField::FLOAT32);
  EXPECT_EQ(msg.data.size(), 24u);

  EXPECT_EQ(pointcloud2_to_xyz(msg), points);
}

TEST(CloudIo, PacksAnEmptyCloudWithoutTouchingTheNullPointer)
{
  builtin_interfaces::msg::Time stamp;
  const PointCloud2 msg = xyz_to_pointcloud2(nullptr, 0, "map", stamp);
  EXPECT_EQ(msg.width, 0u);
  EXPECT_TRUE(msg.data.empty());
  EXPECT_TRUE(pointcloud2_to_xyz(msg).empty());
}

}  // namespace
