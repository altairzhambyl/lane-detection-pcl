// The C++ port has to agree with ransac_seg/ground_plane.py, not merely run.
// These mirror the cases in test/test_ground_plane.py that caught real defects:
// slope recovery, wall rejection, degenerate input, and the mask staying
// aligned with the caller's array when non-finite points are dropped.

#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "ransac_seg_cpp/ground_plane.hpp"

namespace
{

constexpr double kSensorHeight = 1.7;

/// Build a plane at `tilt_deg` plus clutter lifted clear of it.
std::vector<float> make_cloud(
  double tilt_deg, std::size_t n_ground, std::size_t n_obstacle, std::vector<uint8_t> & truth,
  uint64_t seed = 7)
{
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> along(3.0, 40.0);
  std::uniform_real_distribution<double> across(-12.0, 12.0);
  std::uniform_real_distribution<double> lift(0.6, 4.0);
  std::normal_distribution<double> noise(0.0, 0.01);

  const double slope = std::tan(tilt_deg * M_PI / 180.0);
  std::vector<float> xyz;
  xyz.reserve((n_ground + n_obstacle) * 3);
  truth.clear();

  for (std::size_t i = 0; i < n_ground; ++i) {
    const double x = along(rng), y = across(rng);
    xyz.push_back(static_cast<float>(x));
    xyz.push_back(static_cast<float>(y));
    xyz.push_back(static_cast<float>(-kSensorHeight + x * slope + noise(rng)));
    truth.push_back(1);
  }
  for (std::size_t i = 0; i < n_obstacle; ++i) {
    const double x = along(rng), y = across(rng);
    xyz.push_back(static_cast<float>(x));
    xyz.push_back(static_cast<float>(y));
    xyz.push_back(static_cast<float>(-kSensorHeight + x * slope + lift(rng)));
    truth.push_back(0);
  }
  return xyz;
}

double recall_of(const std::vector<uint8_t> & got, const std::vector<uint8_t> & truth)
{
  std::size_t hit = 0, total = 0;
  for (std::size_t i = 0; i < truth.size(); ++i) {
    if (truth[i]) {
      ++total;
      hit += got[i] ? 1 : 0;
    }
  }
  return total ? static_cast<double>(hit) / static_cast<double>(total) : 0.0;
}

}  // namespace

TEST(GroundPlane, RecoversASlopedSurface)
{
  std::vector<uint8_t> truth;
  const auto xyz = make_cloud(6.0, 6000, 2500, truth);
  const auto seg = ransac_seg_cpp::fit_plane_ransac(xyz.data(), truth.size());

  ASSERT_TRUE(seg.model.has_value());
  EXPECT_NEAR(seg.model->tilt_deg(), 6.0, 1.0);
  EXPECT_GT(recall_of(seg.ground, truth), 0.95);
}

TEST(GroundPlane, NormalIsAlwaysOrientedUp)
{
  std::vector<uint8_t> truth;
  const auto xyz = make_cloud(3.0, 4000, 1500, truth);
  const auto seg = ransac_seg_cpp::fit_plane_ransac(xyz.data(), truth.size());

  ASSERT_TRUE(seg.model.has_value());
  EXPECT_GT(seg.model->normal.z(), 0.0);
}

TEST(GroundPlane, TiltGateRejectsAWall)
{
  // A dominant vertical sheet fits a plane far better than any road does; only
  // the frame convention distinguishes them.
  std::mt19937_64 rng(3);
  std::uniform_real_distribution<double> along(-20.0, 20.0);
  std::uniform_real_distribution<double> up(0.0, 6.0);
  std::normal_distribution<double> jitter(0.0, 0.01);

  std::vector<float> xyz;
  for (int i = 0; i < 4000; ++i) {
    xyz.push_back(static_cast<float>(5.0 + jitter(rng)));
    xyz.push_back(static_cast<float>(along(rng)));
    xyz.push_back(static_cast<float>(up(rng)));
  }
  const auto seg = ransac_seg_cpp::fit_plane_ransac(xyz.data(), xyz.size() / 3);
  EXPECT_FALSE(seg.model.has_value());
}

TEST(GroundPlane, DegenerateInputIsSafe)
{
  const std::vector<float> two{0.f, 0.f, 0.f, 1.f, 0.f, 0.f};
  const auto seg = ransac_seg_cpp::fit_plane_ransac(two.data(), 2);
  EXPECT_FALSE(seg.model.has_value());
  EXPECT_EQ(seg.ground.size(), 2u);
  EXPECT_EQ(seg.ground[0], 0);

  const auto empty = ransac_seg_cpp::fit_plane_ransac(nullptr, 0);
  EXPECT_FALSE(empty.model.has_value());
  EXPECT_TRUE(empty.ground.empty());
}

TEST(GroundPlane, CollinearPointsDoNotDefineAPlane)
{
  std::vector<float> line;
  for (int i = 0; i < 500; ++i) {
    line.push_back(static_cast<float>(i) * 0.02f);
    line.push_back(0.f);
    line.push_back(0.f);
  }
  const auto seg = ransac_seg_cpp::fit_plane_ransac(line.data(), line.size() / 3);
  EXPECT_FALSE(seg.model.has_value());
}

TEST(GroundPlane, NonFiniteRowsKeepTheMaskAligned)
{
  std::vector<uint8_t> truth;
  auto xyz = make_cloud(4.0, 3000, 1200, truth);
  const std::size_t finite_count = truth.size();

  const float nan = std::nanf("");
  xyz.insert(xyz.end(), {nan, nan, nan});
  xyz.insert(xyz.end(), {std::numeric_limits<float>::infinity(), 0.f, 0.f});
  const std::size_t total = finite_count + 2;

  const auto seg = ransac_seg_cpp::fit_plane_ransac(xyz.data(), total);
  ASSERT_TRUE(seg.model.has_value());
  EXPECT_EQ(seg.ground.size(), total);
  EXPECT_EQ(seg.ground[total - 1], 0);
  EXPECT_EQ(seg.ground[total - 2], 0);
}

TEST(GroundPlane, SegmentGroundFallsBackWhenNoPlaneIsFound)
{
  // Uniform noise has no consensus set worth reporting, so the fixed-height
  // cut runs and the absent model is what tells the caller.
  std::mt19937_64 rng(11);
  std::uniform_real_distribution<double> box(-10.0, 10.0);
  std::vector<float> xyz;
  for (int i = 0; i < 3000; ++i) {
    xyz.push_back(static_cast<float>(box(rng)));
    xyz.push_back(static_cast<float>(box(rng)));
    xyz.push_back(static_cast<float>(box(rng)));
  }
  const std::size_t n = xyz.size() / 3;
  const auto seg = ransac_seg_cpp::segment_ground(xyz.data(), n, {}, 0.2);

  EXPECT_FALSE(seg.model.has_value());
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(seg.ground[i], xyz[3 * i + 2] < 0.2f ? 1 : 0);
  }
}

TEST(GroundPlane, IsDeterministicForAGivenSeed)
{
  std::vector<uint8_t> truth;
  const auto xyz = make_cloud(5.0, 4000, 1600, truth);
  const auto a = ransac_seg_cpp::fit_plane_ransac(xyz.data(), truth.size());
  const auto b = ransac_seg_cpp::fit_plane_ransac(xyz.data(), truth.size());

  ASSERT_TRUE(a.model.has_value());
  ASSERT_TRUE(b.model.has_value());
  EXPECT_EQ(a.ground, b.ground);
  EXPECT_DOUBLE_EQ(a.model->offset, b.model->offset);
}
