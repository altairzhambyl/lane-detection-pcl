// The rclcpp binding: declaration, reading, and the all-or-nothing runtime set.
//
// The behaviour worth pinning is the refusal path. `ros2 param set` used to
// report success for every shared parameter while the node kept running on its
// start-up snapshot, so an operator retuning inflation or range on a live
// vehicle got an acknowledgement and no behaviour change. Half of these cases
// exist to prove a rejected batch changes nothing at all.

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ransac_seg_cpp/perception_core.hpp"
#include "ransac_seg_cpp/ros_params.hpp"

namespace
{

using ransac_seg_cpp::CommonParams;
using ransac_seg_cpp::PlaneFilter;

class RosParams : public ::testing::Test
{
public:
  static void SetUpTestSuite()
  {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
  }
  static void TearDownTestSuite() {rclcpp::shutdown();}

protected:
  void SetUp() override
  {
    static int counter = 0;
    // A fresh node per case: parameters are declared once and these cases
    // deliberately declare conflicting values.
    node_ = std::make_shared<rclcpp::Node>("test_ros_params_" + std::to_string(counter++));
    ransac_seg_cpp::declare_common_params(*node_);
    params_ = ransac_seg_cpp::read_common_params(*node_);
  }

  rcl_interfaces::msg::SetParametersResult apply(const std::vector<rclcpp::Parameter> & request)
  {
    return ransac_seg_cpp::validate_and_apply(
      request, *node_, params_, min_ground_points_frame_, filter_);
  }

  std::shared_ptr<rclcpp::Node> node_;
  CommonParams params_;
  int min_ground_points_frame_ = 30;
  PlaneFilter filter_{0.3, 5.0, 0.5};
};

TEST_F(RosParams, DeclaresEveryDefault)
{
  for (const ransac_seg_cpp::ParamSpec & spec : ransac_seg_cpp::kDefaults) {
    EXPECT_TRUE(node_->has_parameter(spec.name)) << spec.name;
  }
  const CommonParams & defaults = ransac_seg_cpp::default_params();
  EXPECT_EQ(params_.input_topic, defaults.input_topic);
  EXPECT_DOUBLE_EQ(params_.resolution, defaults.resolution);
  EXPECT_EQ(params_.ray_fill_bins, defaults.ray_fill_bins);
  EXPECT_EQ(params_.ray_fill, defaults.ray_fill);
}

TEST_F(RosParams, ReadWidensAnIntegerIntoADoubleField)
{
  // A launch file routinely hands an int where a double is wanted and rclcpp
  // keeps it an int; the declared default is the authority on the type.
  auto node = std::make_shared<rclcpp::Node>(
    "test_ros_params_widen", rclcpp::NodeOptions().parameter_overrides(
      {rclcpp::Parameter("resolution", static_cast<int64_t>(1))}));
  ransac_seg_cpp::declare_common_params(*node);
  EXPECT_DOUBLE_EQ(ransac_seg_cpp::read_common_params(*node).resolution, 1.0);
}

TEST_F(RosParams, AppliesAGoodBatch)
{
  const auto result = apply(
    {rclcpp::Parameter("resolution", 0.25), rclcpp::Parameter("ray_fill", false)});
  EXPECT_TRUE(result.successful);
  EXPECT_DOUBLE_EQ(params_.resolution, 0.25);
  EXPECT_FALSE(params_.ray_fill);
}

TEST_F(RosParams, PushesPlaneFilterFieldsThrough)
{
  const auto result = apply({rclcpp::Parameter("plane_filter_alpha", 0.75)});
  ASSERT_TRUE(result.successful);
  EXPECT_DOUBLE_EQ(params_.plane_filter_alpha, 0.75);
  EXPECT_DOUBLE_EQ(filter_.alpha, 0.75);
}

TEST_F(RosParams, RefusesConstructionOnlyNames)
{
  const auto result = apply({rclcpp::Parameter("input_topic", std::string("/other"))});
  EXPECT_FALSE(result.successful);
  EXPECT_EQ(result.reason, "'input_topic' is read once at start-up; restart the node");
  EXPECT_EQ(params_.input_topic, ransac_seg_cpp::default_params().input_topic);
}

TEST_F(RosParams, RefusesNamesItCannotReRead)
{
  // A consumer's own parameter: declared on the node, snapshotted by the
  // consumer, and therefore not something a live set can change.
  ransac_seg_cpp::declare(
    *node_, "astar.connectivity", static_cast<int64_t>(8), "Grid neighbourhood");
  const auto result =
    apply({rclcpp::Parameter("astar.connectivity", static_cast<int64_t>(4))});
  EXPECT_FALSE(result.successful);
  EXPECT_EQ(result.reason, "'astar.connectivity' cannot be changed while the node is running");
}

TEST_F(RosParams, SkipsNamesTheNodeDoesNotHave)
{
  // rclcpp runs on-set callbacks for declarations too; a declaration is not a
  // set, and refusing it would make every consumer's configure() fail.
  const auto result = apply({rclcpp::Parameter("not_declared_yet", 1.0)});
  EXPECT_TRUE(result.successful);
}

TEST_F(RosParams, RefusesACoercionFailure)
{
  const auto result = apply({rclcpp::Parameter("resolution", std::string("wide"))});
  EXPECT_FALSE(result.successful);
  EXPECT_EQ(result.reason, "'resolution': expects a double, got a string");
}

TEST_F(RosParams, RejectedBatchChangesNothing)
{
  const CommonParams before = params_;
  const auto result = apply(
    {rclcpp::Parameter("resolution", 0.25),
      rclcpp::Parameter("min_ground_points_frame", static_cast<int64_t>(99)),
      rclcpp::Parameter("qos_depth", static_cast<int64_t>(50))});
  EXPECT_FALSE(result.successful);
  EXPECT_DOUBLE_EQ(params_.resolution, before.resolution);
  EXPECT_EQ(min_ground_points_frame_, 30);
  EXPECT_DOUBLE_EQ(filter_.alpha, 0.3);
}

TEST_F(RosParams, AppliesTheFrameMinimum)
{
  ransac_seg_cpp::declare(
    *node_, "min_ground_points_frame", static_cast<int64_t>(30), "Ground returns needed");
  const auto result =
    apply({rclcpp::Parameter("min_ground_points_frame", static_cast<int64_t>(7))});
  ASSERT_TRUE(result.successful);
  EXPECT_EQ(min_ground_points_frame_, 7);
}

TEST(SensorQos, IsBestEffortKeepLastVolatile)
{
  // DDS refuses to match a RELIABLE subscription with the BEST_EFFORT publisher
  // every production LiDAR driver offers, and the failure is silent.
  const rmw_qos_profile_t profile = ransac_seg_cpp::sensor_qos(7).get_rmw_qos_profile();
  EXPECT_EQ(profile.reliability, RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
  EXPECT_EQ(profile.history, RMW_QOS_POLICY_HISTORY_KEEP_LAST);
  EXPECT_EQ(profile.durability, RMW_QOS_POLICY_DURABILITY_VOLATILE);
  EXPECT_EQ(profile.depth, 7u);
}

TEST(MarkerQos, IsReliableLikeEveryPythonPublisher)
{
  // Not sensor_qos. Every Python node passes a bare depth to create_publisher,
  // which rclpy resolves to RELIABLE, and RViz2's marker display subscribes
  // RELIABLE by default — so publishing markers BEST_EFFORT does not lose the
  // occasional frame, it loses every frame on every marker topic to a stock
  // RViz2 config, with only a QoS-incompatibility line to say why.
  const rmw_qos_profile_t profile = ransac_seg_cpp::marker_qos(7).get_rmw_qos_profile();
  EXPECT_EQ(profile.reliability, RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  EXPECT_EQ(profile.history, RMW_QOS_POLICY_HISTORY_KEEP_LAST);
  EXPECT_EQ(profile.durability, RMW_QOS_POLICY_DURABILITY_VOLATILE);
  EXPECT_EQ(profile.depth, 7u);
}

TEST(MarkerQos, ClampsAZeroDepth)
{
  EXPECT_EQ(ransac_seg_cpp::marker_qos(0).get_rmw_qos_profile().depth, 1u);
}

TEST(SensorQos, ClampsAZeroDepth)
{
  // A KEEP_LAST depth of 0 is rejected by the middleware at subscription time.
  EXPECT_EQ(ransac_seg_cpp::sensor_qos(0).get_rmw_qos_profile().depth, 1u);
}

}  // namespace
