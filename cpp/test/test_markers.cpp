// Port of test/test_markers.py. The interesting assertions are the clearing
// ones: they pin the display bugs the module exists to prevent.

#include <array>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "ransac_seg_cpp/markers.hpp"

namespace
{

using ransac_seg_cpp::Color;
using ransac_seg_cpp::add_points;
using ransac_seg_cpp::box_array;
using ransac_seg_cpp::cube_list;
using ransac_seg_cpp::delete_all;
using ransac_seg_cpp::delete_marker;
using ransac_seg_cpp::line_strip;
using ransac_seg_cpp::make_marker;
using visualization_msgs::msg::Marker;

builtin_interfaces::msg::Time stamp()
{
  builtin_interfaces::msg::Time t;
  t.sec = 12;
  t.nanosec = 34;
  return t;
}

TEST(Markers, MakeMarkerFillsIdentityColourAndScale)
{
  const Marker marker = make_marker(
    "lidar_front", stamp(), "road", 7, Marker::CUBE_LIST, Color{0.1, 0.2, 0.3, 0.4},
    {1.0, 2.0, 3.0});

  EXPECT_EQ(marker.header.frame_id, "lidar_front");
  EXPECT_EQ(marker.header.stamp.sec, 12);
  EXPECT_EQ(marker.header.stamp.nanosec, 34u);
  EXPECT_EQ(marker.ns, "road");
  EXPECT_EQ(marker.id, 7);
  EXPECT_EQ(marker.type, Marker::CUBE_LIST);
  EXPECT_EQ(marker.action, Marker::ADD);
  EXPECT_FLOAT_EQ(marker.color.r, 0.1f);
  EXPECT_FLOAT_EQ(marker.color.g, 0.2f);
  EXPECT_FLOAT_EQ(marker.color.b, 0.3f);
  EXPECT_FLOAT_EQ(marker.color.a, 0.4f);
  EXPECT_DOUBLE_EQ(marker.scale.x, 1.0);
  EXPECT_DOUBLE_EQ(marker.scale.y, 2.0);
  EXPECT_DOUBLE_EQ(marker.scale.z, 3.0);
  EXPECT_DOUBLE_EQ(marker.pose.orientation.w, 1.0);
}

TEST(Markers, LifetimeOmittedWhenZeroAndSetOtherwise)
{
  const Marker forever = make_marker(
    "f", stamp(), "n", 0, Marker::CUBE, Color{1.0, 1.0, 1.0, 1.0}, {1.0, 1.0, 1.0});
  // Unset is what ROS reads as "never expire".
  EXPECT_EQ(forever.lifetime.sec, 0);
  EXPECT_EQ(forever.lifetime.nanosec, 0u);

  const Marker timed = make_marker(
    "f", stamp(), "n", 0, Marker::CUBE, Color{1.0, 1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}, 1.5);
  EXPECT_EQ(timed.lifetime.sec, 1);
  EXPECT_NEAR(static_cast<double>(timed.lifetime.nanosec), 5e8, 1.0);
}

TEST(Markers, MakeMarkerHonoursAnExplicitAction)
{
  const Marker marker = make_marker(
    "f", stamp(), "n", 3, Marker::CUBE, Color{1.0, 1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}, 0.0,
    Marker::MODIFY);
  EXPECT_EQ(marker.action, Marker::MODIFY);
}

TEST(Markers, AddPointsMatchesTheInputRows)
{
  std::vector<double> pts(9);
  for (std::size_t i = 0; i < pts.size(); ++i) {
    pts[i] = static_cast<double>(i);
  }
  Marker marker = make_marker(
    "f", stamp(), "n", 0, Marker::POINTS, Color{1.0, 1.0, 1.0, 1.0}, {0.1, 0.1, 0.1});
  add_points(marker, pts.data(), 3);

  ASSERT_EQ(marker.points.size(), 3u);
  EXPECT_DOUBLE_EQ(marker.points[1].x, 3.0);
  EXPECT_DOUBLE_EQ(marker.points[1].y, 4.0);
  EXPECT_DOUBLE_EQ(marker.points[1].z, 5.0);
}

TEST(Markers, CubeListSizesTilesAndCountsPoints)
{
  const std::vector<double> centres(15, 0.0);
  const Marker marker = cube_list(
    "f", stamp(), "road", 3, centres.data(), 5, 0.5, Color{0.0, 1.0, 0.0, 0.5}, 0.05);

  EXPECT_EQ(marker.type, Marker::CUBE_LIST);
  EXPECT_EQ(marker.points.size(), 5u);
  EXPECT_DOUBLE_EQ(marker.scale.x, 0.5);
  EXPECT_DOUBLE_EQ(marker.scale.y, 0.5);
  EXPECT_DOUBLE_EQ(marker.scale.z, 0.05);
}

TEST(Markers, CubeListClearsOnlyItsOwnMarkerOnEmptyInput)
{
  // A DELETEALL here would wipe the sibling namespaces published on the same
  // topic microseconds earlier in the same frame.
  const Marker marker = cube_list(
    "f", stamp(), "road", 3, nullptr, 0, 0.5, Color{0.0, 1.0, 0.0, 1.0});

  EXPECT_EQ(marker.action, Marker::DELETE);
  EXPECT_EQ(marker.ns, "road");
  EXPECT_EQ(marker.id, 3);
}

TEST(Markers, LineStripNeedsTwoPoints)
{
  const std::vector<double> pts(6, 0.0);
  for (std::size_t count = 0; count < 2; ++count) {
    const Marker marker = line_strip(
      "f", stamp(), "path", 1, pts.data(), count, Color{1.0, 0.0, 0.0, 1.0});
    EXPECT_EQ(marker.action, Marker::DELETE) << "count " << count;
    EXPECT_EQ(marker.id, 1) << "count " << count;
  }

  const Marker good = line_strip(
    "f", stamp(), "path", 1, pts.data(), 2, Color{1.0, 0.0, 0.0, 1.0}, 0.2);
  EXPECT_EQ(good.action, Marker::ADD);
  EXPECT_EQ(good.type, Marker::LINE_STRIP);
  EXPECT_EQ(good.points.size(), 2u);
  EXPECT_DOUBLE_EQ(good.scale.x, 0.2);
  EXPECT_DOUBLE_EQ(good.scale.y, 0.0);
  EXPECT_DOUBLE_EQ(good.scale.z, 0.0);
}

TEST(Markers, BoxArrayAssignsConsecutiveIds)
{
  const std::vector<std::array<double, 3>> centres = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  const auto array = box_array(
    "f", stamp(), "boxes", 10, centres, {{0.5, 0.5, 1.0}}, Color{1.0, 1.0, 0.0, 0.8});

  ASSERT_EQ(array.markers.size(), 3u);
  EXPECT_EQ(array.markers[1].id, 10);
  EXPECT_EQ(array.markers[2].id, 11);
  EXPECT_EQ(array.markers[1].type, Marker::CUBE);
  EXPECT_DOUBLE_EQ(array.markers[2].pose.position.x, 4.0);
  EXPECT_DOUBLE_EQ(array.markers[1].scale.z, 1.0);
}

TEST(Markers, BoxArrayAlwaysLeadsWithAClear)
{
  // Ids retired when the obstacle count drops must not survive as phantoms:
  // the default lifetime is "never expire".
  std::vector<std::array<double, 3>> centres;
  for (int i = 0; i < 5; ++i) {
    centres.push_back({static_cast<double>(i), 0.0, 0.0});
  }
  const auto busy = box_array(
    "f", stamp(), "b", 0, centres, {{1.0, 1.0, 1.0}}, Color{1.0, 1.0, 1.0, 1.0});
  EXPECT_EQ(busy.markers[0].action, Marker::DELETEALL);
  ASSERT_EQ(busy.markers.size(), 6u);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(busy.markers[static_cast<std::size_t>(i) + 1].id, i);
  }

  const std::vector<std::array<double, 3>> two(centres.begin(), centres.begin() + 2);
  const auto quiet = box_array(
    "f", stamp(), "b", 0, two, {{1.0, 1.0, 1.0}}, Color{1.0, 1.0, 1.0, 1.0});
  EXPECT_EQ(quiet.markers[0].action, Marker::DELETEALL);
  EXPECT_EQ(quiet.markers.size(), 3u);
}

TEST(Markers, BoxArrayAcceptsPerBoxExtentsAndClearsWhenEmpty)
{
  const std::vector<std::array<double, 3>> centres = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  const std::vector<std::array<double, 3>> extents = {{1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}};
  const auto array = box_array(
    "f", stamp(), "b", 0, centres, extents, Color{1.0, 1.0, 1.0, 1.0});
  EXPECT_DOUBLE_EQ(array.markers[2].scale.x, 2.0);

  const auto empty = box_array(
    "f", stamp(), "b", 0, {}, {{1.0, 1.0, 1.0}}, Color{1.0, 1.0, 1.0, 1.0});
  ASSERT_EQ(empty.markers.size(), 1u);
  EXPECT_EQ(empty.markers[0].action, Marker::DELETEALL);
}

TEST(Markers, BoxArrayRejectsAMismatchedExtentList)
{
  const std::vector<std::array<double, 3>> centres = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  const std::vector<std::array<double, 3>> extents = {
    {1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}, {3.0, 3.0, 3.0}};
  EXPECT_THROW(
    box_array("f", stamp(), "b", 0, centres, extents, Color{1.0, 1.0, 1.0, 1.0}),
    std::invalid_argument);
}

TEST(Markers, DeleteAllTargetsOneNamespace)
{
  const Marker marker = delete_all("lidar_front", stamp(), "path");
  EXPECT_EQ(marker.action, Marker::DELETEALL);
  EXPECT_EQ(marker.ns, "path");
  EXPECT_EQ(marker.header.frame_id, "lidar_front");
}

TEST(Markers, DeleteMarkerTargetsOneId)
{
  const Marker marker = delete_marker("lidar_front", stamp(), "path", 4);
  EXPECT_EQ(marker.action, Marker::DELETE);
  EXPECT_EQ(marker.ns, "path");
  EXPECT_EQ(marker.id, 4);
}

}  // namespace
