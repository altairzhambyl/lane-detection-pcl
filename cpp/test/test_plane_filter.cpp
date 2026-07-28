// Mirrors test/test_plane_filter.py: the miss budget, both innovation gates,
// and the opposed-normal path that cannot be averaged.

#include <cmath>
#include <optional>

#include <gtest/gtest.h>

#include "ransac_seg_cpp/plane_filter.hpp"

namespace
{

using ransac_seg_cpp::PlaneFilter;
using ransac_seg_cpp::PlaneModel;

constexpr double kPi = 3.14159265358979323846;

/// A plane tilted `tilt_deg` about the y axis, `offset` metres off the origin.
PlaneModel model_at(double tilt_deg, double offset, double inlier_ratio = 0.7)
{
  const double rad = tilt_deg * kPi / 180.0;
  PlaneModel model;
  model.normal = Eigen::Vector3d(std::sin(rad), 0.0, std::cos(rad));
  model.offset = offset;
  model.inlier_ratio = inlier_ratio;
  model.iterations = 42;
  model.refined = true;
  return model;
}

TEST(PlaneFilterTest, FirstGoodModelIsTakenVerbatim)
{
  PlaneFilter filter;
  EXPECT_FALSE(filter.last().has_value());

  const PlaneModel fresh = model_at(3.0, -1.7);
  const auto out = filter.update(fresh);
  ASSERT_TRUE(out.has_value());
  EXPECT_DOUBLE_EQ(out->offset, fresh.offset);
  EXPECT_TRUE(out->normal.isApprox(fresh.normal));
  ASSERT_TRUE(filter.last().has_value());
}

TEST(PlaneFilterTest, BlendPullsTowardsTheHeldPlane)
{
  PlaneFilter filter(0.3, 5.0, 0.5);
  filter.update(model_at(0.0, -1.7));
  const PlaneModel fresh = model_at(2.0, -1.5, 0.55);
  const auto out = filter.update(fresh);
  ASSERT_TRUE(out.has_value());

  // 0.7 * -1.7 + 0.3 * -1.5
  EXPECT_DOUBLE_EQ(out->offset, 0.7 * -1.7 + 0.3 * -1.5);
  EXPECT_NEAR(out->normal.norm(), 1.0, 1e-12);
  // Metadata comes from the fresh model, not the held one.
  EXPECT_DOUBLE_EQ(out->inlier_ratio, 0.55);
}

TEST(PlaneFilterTest, MissBudgetHoldsThenGivesUp)
{
  PlaneFilter filter(0.3, 5.0, 0.5, 3);
  const PlaneModel good = model_at(0.0, -1.7);
  ASSERT_TRUE(filter.update(good).has_value());

  // The first misses keep returning the held plane...
  EXPECT_TRUE(filter.update(std::nullopt).has_value());
  EXPECT_TRUE(filter.update(std::nullopt).has_value());
  // ...and the third exhausts the budget, so the caller must fall back.
  EXPECT_FALSE(filter.update(std::nullopt).has_value());
  EXPECT_FALSE(filter.last().has_value());

  // A good model after the reset is taken verbatim again.
  const auto restarted = filter.update(good);
  ASSERT_TRUE(restarted.has_value());
  EXPECT_DOUBLE_EQ(restarted->offset, good.offset);
}

TEST(PlaneFilterTest, AGoodFrameClearsTheMissCounter)
{
  PlaneFilter filter(0.3, 5.0, 0.5, 3);
  filter.update(model_at(0.0, -1.7));
  filter.update(std::nullopt);
  filter.update(std::nullopt);
  ASSERT_TRUE(filter.update(model_at(0.5, -1.7)).has_value());

  // Counter reset: two more misses must not trip the budget.
  EXPECT_TRUE(filter.update(std::nullopt).has_value());
  EXPECT_TRUE(filter.update(std::nullopt).has_value());
}

TEST(PlaneFilterTest, TiltGateRejectsAWallLikeFit)
{
  PlaneFilter filter(0.3, 5.0, 0.5, 100);
  const PlaneModel road = model_at(0.0, -1.7);
  filter.update(road);

  // 30 degrees is well past the 5 degree gate: a truck flank, not a road.
  const auto out = filter.update(model_at(30.0, -1.7));
  ASSERT_TRUE(out.has_value());
  EXPECT_TRUE(out->normal.isApprox(road.normal));  // held, not blended
  EXPECT_DOUBLE_EQ(out->offset, road.offset);

  // Just inside the gate is accepted and does move the estimate.
  const auto blended = filter.update(model_at(4.0, -1.7));
  ASSERT_TRUE(blended.has_value());
  EXPECT_FALSE(blended->normal.isApprox(road.normal));
}

TEST(PlaneFilterTest, OffsetGateRejectsAJumpInHeight)
{
  PlaneFilter filter(0.3, 5.0, 0.5, 100);
  const PlaneModel road = model_at(0.0, -1.7);
  filter.update(road);

  const auto out = filter.update(model_at(0.0, -1.7 + 0.9));
  ASSERT_TRUE(out.has_value());
  EXPECT_DOUBLE_EQ(out->offset, road.offset);

  // Exactly at the threshold is not a jump: the gate is a strict >.
  const auto edge = filter.update(model_at(0.0, -1.7 + 0.5));
  ASSERT_TRUE(edge.has_value());
  EXPECT_DOUBLE_EQ(edge->offset, 0.7 * -1.7 + 0.3 * (-1.7 + 0.5));
}

TEST(PlaneFilterTest, OpposedNormalsTakeTheFreshModelVerbatim)
{
  // alpha 0.5 with exactly opposed normals blends to the zero vector, which
  // cannot be normalised; the filter must trust the fresh measurement instead
  // of dividing by ~0. The tilt gate is opened wide so this path is reachable.
  PlaneFilter filter(0.5, 200.0, 10.0);
  PlaneModel first;
  first.normal = Eigen::Vector3d(0.0, 0.0, 1.0);
  first.offset = -1.0;
  filter.update(first);

  PlaneModel opposed;
  opposed.normal = Eigen::Vector3d(0.0, 0.0, -1.0);
  opposed.offset = -1.2;
  opposed.inlier_ratio = 0.9;
  const auto out = filter.update(opposed);
  ASSERT_TRUE(out.has_value());
  EXPECT_TRUE(out->normal.isApprox(opposed.normal));
  EXPECT_DOUBLE_EQ(out->offset, opposed.offset);
  EXPECT_DOUBLE_EQ(out->inlier_ratio, 0.9);
}

TEST(PlaneFilterTest, ResetForgetsEverything)
{
  PlaneFilter filter;
  filter.update(model_at(0.0, -1.7));
  ASSERT_TRUE(filter.last().has_value());
  filter.reset();
  EXPECT_FALSE(filter.last().has_value());
}

}  // namespace
