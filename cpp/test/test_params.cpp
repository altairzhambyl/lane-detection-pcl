// Pins the parameter table against ransac_seg/params.py: the same 27 keys in
// the same order with the same defaults, and the coercion a launch file relies
// on when it hands an int where a double is wanted.

#include <cstdint>
#include <string>

#include <gtest/gtest.h>

#include "ransac_seg_cpp/params.hpp"

namespace
{

using ransac_seg_cpp::CommonParams;
using ransac_seg_cpp::ParamType;
using ransac_seg_cpp::ParamValue;
using ransac_seg_cpp::apply_param;
using ransac_seg_cpp::default_params;
using ransac_seg_cpp::find_param;
using ransac_seg_cpp::is_construction_only;
using ransac_seg_cpp::kDefaults;
using ransac_seg_cpp::ransac_params;

TEST(Params, TableIsTheTwentySevenPythonKeysInOrder)
{
  const char * expected[] = {
    "input_topic", "marker_topic", "qos_depth", "frame_id_override", "resolution",
    "robot_radius", "inflation_radius", "cost_scaling", "min_ground_points",
    "min_obstacle_points", "ray_fill", "ray_fill_bins", "ray_fill_min_cells", "min_range",
    "max_range", "marker_lifetime_sec", "distance_threshold", "max_iterations",
    "min_iterations", "confidence", "max_tilt_deg", "min_inlier_ratio", "max_scoring_points",
    "fallback_height", "plane_filter_alpha", "plane_max_tilt_jump_deg", "plane_max_offset_jump"};
  ASSERT_EQ(kDefaults.size(), 27u);
  for (std::size_t i = 0; i < kDefaults.size(); ++i) {
    EXPECT_STREQ(kDefaults[i].name, expected[i]) << "at index " << i;
    EXPECT_NE(kDefaults[i].description, nullptr);
  }
}

TEST(Params, DefaultsMatchThePythonDict)
{
  const CommonParams & p = default_params();
  EXPECT_EQ(p.input_topic, "/sensor/lidar_front/points");
  EXPECT_EQ(p.marker_topic, "/semantic_map/squares");
  EXPECT_EQ(p.qos_depth, 10);
  EXPECT_EQ(p.frame_id_override, "");
  EXPECT_DOUBLE_EQ(p.resolution, 0.5);
  EXPECT_DOUBLE_EQ(p.robot_radius, 0.6);
  EXPECT_DOUBLE_EQ(p.inflation_radius, 1.5);
  EXPECT_DOUBLE_EQ(p.cost_scaling, 3.0);
  EXPECT_EQ(p.min_ground_points, 1);
  EXPECT_EQ(p.min_obstacle_points, 3);
  EXPECT_TRUE(p.ray_fill);
  EXPECT_EQ(p.ray_fill_bins, 720);
  EXPECT_EQ(p.ray_fill_min_cells, 2);
  EXPECT_DOUBLE_EQ(p.min_range, 0.5);
  EXPECT_DOUBLE_EQ(p.max_range, 60.0);
  EXPECT_DOUBLE_EQ(p.marker_lifetime_sec, 0.0);
  EXPECT_DOUBLE_EQ(p.distance_threshold, 0.20);
  EXPECT_EQ(p.max_iterations, 200);
  EXPECT_EQ(p.min_iterations, 12);
  EXPECT_DOUBLE_EQ(p.confidence, 0.99);
  EXPECT_DOUBLE_EQ(p.max_tilt_deg, 10.0);
  EXPECT_DOUBLE_EQ(p.min_inlier_ratio, 0.15);
  EXPECT_EQ(p.max_scoring_points, 20000);
  EXPECT_DOUBLE_EQ(p.fallback_height, 0.2);
  EXPECT_DOUBLE_EQ(p.plane_filter_alpha, 0.3);
  EXPECT_DOUBLE_EQ(p.plane_max_tilt_jump_deg, 5.0);
  EXPECT_DOUBLE_EQ(p.plane_max_offset_jump, 0.5);
}

TEST(Params, ConstructionOnlyIsTheThreeStartupKeys)
{
  EXPECT_TRUE(is_construction_only("input_topic"));
  EXPECT_TRUE(is_construction_only("marker_topic"));
  EXPECT_TRUE(is_construction_only("qos_depth"));
  EXPECT_FALSE(is_construction_only("resolution"));
  EXPECT_FALSE(is_construction_only("not_a_parameter"));
}

TEST(Params, FindParamKnowsTypesAndRejectsStrangers)
{
  ASSERT_NE(find_param("resolution"), nullptr);
  EXPECT_EQ(find_param("resolution")->type, ParamType::Double);
  EXPECT_EQ(find_param("ray_fill")->type, ParamType::Bool);
  EXPECT_EQ(find_param("max_iterations")->type, ParamType::Int);
  EXPECT_EQ(find_param("input_topic")->type, ParamType::String);
  EXPECT_EQ(find_param("nope"), nullptr);
}

TEST(Params, IntIsCoercedIntoADoubleField)
{
  // The case that actually happens: a launch file writes `resolution: 1` and
  // rclcpp keeps it an integer.
  CommonParams p;
  EXPECT_EQ(apply_param(p, "resolution", ParamValue{int64_t{1}}), "");
  EXPECT_DOUBLE_EQ(p.resolution, 1.0);
  EXPECT_EQ(apply_param(p, "min_range", ParamValue{true}), "");
  EXPECT_DOUBLE_EQ(p.min_range, 1.0);
}

TEST(Params, DoubleTruncatesTowardZeroIntoAnIntField)
{
  CommonParams p;
  EXPECT_EQ(apply_param(p, "max_iterations", ParamValue{321.9}), "");
  EXPECT_EQ(p.max_iterations, 321);  // Python int() truncates, it does not round
  EXPECT_EQ(apply_param(p, "min_iterations", ParamValue{-4.9}), "");
  EXPECT_EQ(p.min_iterations, -4);
  EXPECT_EQ(apply_param(p, "ray_fill_bins", ParamValue{true}), "");
  EXPECT_EQ(p.ray_fill_bins, 1);
}

TEST(Params, BoolTakesIntsAndDoubles)
{
  CommonParams p;
  EXPECT_EQ(apply_param(p, "ray_fill", ParamValue{int64_t{0}}), "");
  EXPECT_FALSE(p.ray_fill);
  EXPECT_EQ(apply_param(p, "ray_fill", ParamValue{int64_t{7}}), "");
  EXPECT_TRUE(p.ray_fill);
  EXPECT_EQ(apply_param(p, "ray_fill", ParamValue{0.0}), "");
  EXPECT_FALSE(p.ray_fill);
}

TEST(Params, StringsPassThroughUntouched)
{
  CommonParams p;
  EXPECT_EQ(apply_param(p, "frame_id_override", ParamValue{std::string("base_link")}), "");
  EXPECT_EQ(p.frame_id_override, "base_link");
}

TEST(Params, UncoercibleValuesAreRefusedWithAReasonAndLeaveTheStructAlone)
{
  CommonParams p;
  const double before = p.resolution;
  EXPECT_EQ(
    apply_param(p, "resolution", ParamValue{std::string("half")}),
    "expects a double, got a string");
  EXPECT_DOUBLE_EQ(p.resolution, before);

  EXPECT_EQ(
    apply_param(p, "max_iterations", ParamValue{std::string("200")}),
    "expects an integer, got a string");
  EXPECT_EQ(
    apply_param(p, "ray_fill", ParamValue{std::string("false")}),
    "expects a bool, got a string");
  EXPECT_EQ(apply_param(p, "input_topic", ParamValue{int64_t{3}}), "expects a string, got an integer");
  EXPECT_EQ(
    apply_param(p, "not_a_parameter", ParamValue{1.0}),
    "cannot be changed while the node is running");
}

TEST(Params, RansacParamsCarriesTheEightSegmentGroundFields)
{
  CommonParams p;
  p.distance_threshold = 0.33;
  p.max_iterations = 111;
  p.min_iterations = 7;
  p.confidence = 0.95;
  p.max_tilt_deg = 12.5;
  p.min_inlier_ratio = 0.25;
  p.max_scoring_points = 5000;
  p.fallback_height = 0.9;

  const auto r = ransac_params(p);
  EXPECT_DOUBLE_EQ(r.distance_threshold, 0.33);
  EXPECT_EQ(r.max_iterations, 111);
  EXPECT_EQ(r.min_iterations, 7);
  EXPECT_DOUBLE_EQ(r.confidence, 0.95);
  EXPECT_DOUBLE_EQ(r.max_tilt_deg, 12.5);
  EXPECT_DOUBLE_EQ(r.min_inlier_ratio, 0.25);
  EXPECT_EQ(r.max_scoring_points, 5000);
  // fallback_height is not part of RansacParams; segment_ground takes it
  // separately, so it must not have leaked into seed or refine.
  EXPECT_TRUE(r.refine);
  EXPECT_EQ(r.seed, 0u);
}

}  // namespace
