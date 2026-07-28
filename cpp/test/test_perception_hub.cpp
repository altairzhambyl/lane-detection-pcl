// The hub's own failure paths, exercised over a real DDS round trip.
//
// PerceptionHub is where the port stops being six independent processes, and
// that is a blast-radius change as much as a speed one: anything that escapes
// on_cloud takes every consumer with it. cloud_io refuses five distinct
// malformed-cloud shapes by throwing, so the callback has to be the place that
// decides a bad message is not a fatal one. Driving it through an actual
// publisher rather than calling on_cloud directly is deliberate — the bug was
// that the exception unwound through rclcpp::spin, which only the real path
// reproduces.

#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include "ransac_seg_cpp/consumers.hpp"
#include "ransac_seg_cpp/perception_core.hpp"

namespace
{

using ransac_seg_cpp::PerceptionHub;
using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

PointField make_field(const std::string & name, uint32_t offset)
{
  PointField field;
  field.name = name;
  field.offset = offset;
  field.datatype = PointField::FLOAT32;
  field.count = 1;
  return field;
}

/// A cloud carrying only x and y — one of the shapes cloud_io refuses.
PointCloud2 cloud_without_z()
{
  PointCloud2 msg;
  msg.header.frame_id = "base_link";
  msg.height = 1;
  msg.width = 64;
  msg.fields = {make_field("x", 0), make_field("y", 4)};
  msg.is_bigendian = false;
  msg.point_step = 8;
  msg.row_step = 8 * msg.width;
  msg.is_dense = true;
  std::vector<float> xy(2 * msg.width, 1.0f);
  msg.data.resize(xy.size() * sizeof(float));
  std::memcpy(msg.data.data(), xy.data(), msg.data.size());
  return msg;
}

/// An ordinary flat ground sweep, so the frame after the bad one is observable.
PointCloud2 flat_ground()
{
  PointCloud2 msg;
  msg.header.frame_id = "base_link";
  msg.height = 1;
  msg.width = 900;
  msg.fields = {make_field("x", 0), make_field("y", 4), make_field("z", 8)};
  msg.is_bigendian = false;
  msg.point_step = 12;
  msg.row_step = 12 * msg.width;
  msg.is_dense = true;
  std::vector<float> xyz;
  xyz.reserve(3 * msg.width);
  for (uint32_t i = 0; i < msg.width; ++i) {
    xyz.push_back(3.0f + 0.03f * static_cast<float>(i % 300));
    xyz.push_back(-5.0f + 0.01f * static_cast<float>(i));
    xyz.push_back(-1.6f);
  }
  msg.data.resize(xyz.size() * sizeof(float));
  std::memcpy(msg.data.data(), xyz.data(), msg.data.size());
  return msg;
}

class Hub : public ::testing::Test
{
public:
  static void SetUpTestSuite()
  {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
  }
  static void TearDownTestSuite() {rclcpp::shutdown();}
};

TEST_F(Hub, AnUndecodableCloudDropsTheFrameInsteadOfTheProcess)
{
  auto node = std::make_shared<PerceptionHub>("hub_undecodable_test");
  node->add_consumer(ransac_seg_cpp::make_consumer("ground"));
  node->start();

  auto talker = std::make_shared<rclcpp::Node>("hub_undecodable_talker");
  auto publisher = talker->create_publisher<PointCloud2>(
    node->params().input_topic, ransac_seg_cpp::sensor_qos(node->params().qos_depth));

  // Discovery: BEST_EFFORT does not queue for a subscription that is not
  // matched yet, so publishing immediately would prove nothing.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (publisher->get_subscription_count() == 0 &&
    std::chrono::steady_clock::now() < deadline)
  {
    rclcpp::spin_some(node);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_GT(publisher->get_subscription_count(), 0u) << "hub never matched the publisher";

  PointCloud2 bad = cloud_without_z();
  PointCloud2 good = flat_ground();
  for (int round = 0; round < 3; ++round) {
    bad.header.stamp = talker->get_clock()->now();
    publisher->publish(bad);
    // Before the fix this threw std::invalid_argument out of the executor; the
    // shipped node had no handler above it either, so the process aborted.
    ASSERT_NO_THROW(
      rclcpp::spin_until_future_complete(
        node, std::promise<bool>().get_future(), std::chrono::milliseconds(150)));

    // And the very next well-formed sweep still gets through: the frame was
    // dropped, not the subscription.
    good.header.stamp = talker->get_clock()->now();
    publisher->publish(good);
    ASSERT_NO_THROW(
      rclcpp::spin_until_future_complete(
        node, std::promise<bool>().get_future(), std::chrono::milliseconds(150)));
  }
  EXPECT_TRUE(rclcpp::ok());
}

}  // namespace
