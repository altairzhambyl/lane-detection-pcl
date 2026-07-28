// Port of test/test_cluster.py plus the edge cases the Python covers only
// implicitly (the boundary of the height gate, the ordering contract of
// voxel_labels, an empty cloud with a zero voxel size).
//
// The synthetic blocks are built with std::mt19937_64 rather than numpy's
// PCG64: every assertion here is structural (how many clusters, how close the
// centroid, what order) and none depends on the particular sample, so
// reproducing numpy's bit stream would buy nothing.

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "ransac_seg_cpp/cluster.hpp"

using ransac_seg_cpp::Cluster;
using ransac_seg_cpp::ClusterOptions;
using ransac_seg_cpp::cluster_obstacles;
using ransac_seg_cpp::voxel_downsample;
using ransac_seg_cpp::voxel_labels;

namespace
{

/// `count` points uniformly inside an axis-aligned block, appended to `out`.
void block(
  std::mt19937_64 & rng, const double centre[3], const double size[3], int count,
  std::vector<float> & out)
{
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (int i = 0; i < count; ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      out.push_back(
        static_cast<float>(centre[axis] + (unit(rng) - 0.5) * size[axis]));
    }
  }
}

std::vector<Cluster> run(const std::vector<float> & xyz, const ClusterOptions & opts)
{
  return cluster_obstacles(xyz.data(), xyz.size() / 3, 3, opts);
}

}  // namespace

TEST(VoxelDownsample, CollapsesAndMapsBack)
{
  const std::vector<double> points{0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 5.00, 5.00, 5.00};
  const auto result = voxel_downsample(points.data(), 3, 0.2);

  EXPECT_EQ(result.reps.size(), 6u);
  EXPECT_EQ(result.inverse.size(), 3u);
  EXPECT_EQ(result.inverse[0], result.inverse[1]);
  EXPECT_NE(result.inverse[0], result.inverse[2]);
  const std::size_t v = static_cast<std::size_t>(result.inverse[0]);
  for (int axis = 0; axis < 3; ++axis) {
    EXPECT_NEAR(result.reps[3 * v + static_cast<std::size_t>(axis)], 0.03, 1e-12);
  }
}

TEST(VoxelDownsample, EmptyCloud)
{
  const auto result = voxel_downsample(nullptr, 0, 0.2);
  EXPECT_TRUE(result.reps.empty());
  EXPECT_TRUE(result.inverse.empty());
}

TEST(VoxelDownsample, EmptyCloudWithZeroVoxelSizeDoesNotThrow)
{
  // The emptiness check comes first in the Python too; a zero voxel size only
  // matters once there is a point to bin.
  EXPECT_NO_THROW(voxel_downsample(nullptr, 0, 0.0));
}

TEST(VoxelDownsample, NonPositiveVoxelSizeThrows)
{
  const std::vector<double> points{0.0, 0.0, 0.0};
  EXPECT_THROW(voxel_downsample(points.data(), 1, 0.0), std::invalid_argument);
  EXPECT_THROW(voxel_downsample(points.data(), 1, -0.2), std::invalid_argument);
}

TEST(VoxelDownsample, SurvivesAHugeExtent)
{
  // The lattice spans 5e12 cells per axis, so the ravelled index would overflow
  // an int64; the general lexicographic path must take over instead of wrapping.
  const std::vector<double> points{0.0, 0.0, 0.0, 1e12, 1e12, 1e12, 1e12, 1e12, 1e12};
  const auto result = voxel_downsample(points.data(), 3, 0.2);

  EXPECT_EQ(result.reps.size(), 6u);
  ASSERT_EQ(result.inverse.size(), 3u);
  EXPECT_EQ(result.inverse[0], 0);
  EXPECT_EQ(result.inverse[1], 1);
  EXPECT_EQ(result.inverse[2], 1);
}

TEST(VoxelLabels, NumbersComponentsByFirstVisit)
{
  // Nodes 0 and 2 link, node 1 stands alone: scipy numbers by first visit, so
  // the labels are 0, 1, 0 and never 0, 1, 1.
  const std::vector<double> reps{0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.5, 0.0, 0.0};
  const auto labels = voxel_labels(reps, 0.6);
  ASSERT_EQ(labels.size(), 3u);
  EXPECT_EQ(labels[0], 0);
  EXPECT_EQ(labels[1], 1);
  EXPECT_EQ(labels[2], 0);
}

TEST(VoxelLabels, RadiusIsInclusive)
{
  // scipy's query_pairs links at exactly the radius; a strict < would split
  // these two and change the linkage at the boundary.
  const std::vector<double> reps{0.0, 0.0, 0.0, 0.6, 0.0, 0.0};
  const auto labels = voxel_labels(reps, 0.6);
  ASSERT_EQ(labels.size(), 2u);
  EXPECT_EQ(labels[0], labels[1]);
}

TEST(VoxelLabels, NoPairsGivesIota)
{
  const std::vector<double> reps{0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 20.0, 0.0, 0.0};
  const auto labels = voxel_labels(reps, 0.6);
  ASSERT_EQ(labels.size(), 3u);
  EXPECT_EQ(labels[0], 0);
  EXPECT_EQ(labels[1], 1);
  EXPECT_EQ(labels[2], 2);
}

TEST(ClusterObstacles, ThreeSeparatedBlocks)
{
  std::mt19937_64 rng(7);
  const double size[3]{1.0, 1.0, 1.6};
  const double truth[3][3]{{2.0, 0.0, 1.0}, {12.0, 6.0, 1.0}, {20.0, -8.0, 1.0}};
  std::vector<float> points;
  for (const auto & centre : truth) {block(rng, centre, size, 400, points);}

  const auto clusters = run(points, ClusterOptions{});

  ASSERT_EQ(clusters.size(), 3u);
  int total = 0;
  for (const auto & cluster : clusters) {total += cluster.count;}
  EXPECT_EQ(total, 1200);
  for (const auto & want : truth) {
    const Eigen::Vector3d target{want[0], want[1], want[2]};
    double best = std::numeric_limits<double>::infinity();
    for (const auto & cluster : clusters) {
      best = std::min(best, (cluster.centroid - target).norm());
    }
    EXPECT_LT(best, 0.2);
  }
}

TEST(ClusterObstacles, CloseBlocksMerge)
{
  std::mt19937_64 rng(11);
  const double size[3]{1.0, 1.0, 1.6};
  const double left[3]{5.0, 0.0, 1.0};
  const double right[3]{5.0, 1.3, 1.0};
  std::vector<float> points;
  block(rng, left, size, 300, points);
  block(rng, right, size, 300, points);

  const auto clusters = run(points, ClusterOptions{});
  ASSERT_EQ(clusters.size(), 1u);
  EXPECT_EQ(clusters[0].count, 600);
}

TEST(ClusterObstacles, SpeckRejectedByMinPoints)
{
  std::mt19937_64 rng(3);
  const double centre[3]{4.0, 0.0, 1.0};
  const double size[3]{0.2, 0.2, 1.0};
  std::vector<float> points;
  block(rng, centre, size, 5, points);
  EXPECT_TRUE(run(points, ClusterOptions{}).empty());
}

TEST(ClusterObstacles, PancakeRejectedByMinHeight)
{
  std::mt19937_64 rng(5);
  const double centre[3]{6.0, 0.0, 0.02};
  const double size[3]{2.0, 2.0, 0.05};
  std::vector<float> points;
  block(rng, centre, size, 500, points);
  EXPECT_TRUE(run(points, ClusterOptions{}).empty());
}

TEST(ClusterObstacles, TallArtefactRejectedByMaxHeight)
{
  std::mt19937_64 rng(9);
  const double centre[3]{8.0, 0.0, 3.0};
  const double size[3]{0.5, 0.5, 6.0};
  std::vector<float> points;
  block(rng, centre, size, 800, points);
  EXPECT_TRUE(run(points, ClusterOptions{}).empty());
}

TEST(ClusterObstacles, HeightGateIsStrictOnBothSides)
{
  // Exactly min_height must pass: the Python rejects on `height < min_height`.
  // 0.25 is exact in float32, so the boundary is the boundary and not a
  // rounding artefact of the input cast.
  std::vector<float> points;
  for (int i = 0; i < 60; ++i) {
    const float x = 4.0f + 0.005f * static_cast<float>(i);
    points.insert(points.end(), {x, 0.0f, 0.0f});
    points.insert(points.end(), {x, 0.0f, 0.25f});
  }
  ClusterOptions at_boundary;
  at_boundary.min_height = 0.25;
  ASSERT_EQ(run(points, at_boundary).size(), 1u);
  EXPECT_NEAR(run(points, at_boundary)[0].height(), 0.25, 1e-9);

  ClusterOptions just_above;
  just_above.min_height = 0.2500001;
  EXPECT_TRUE(run(points, just_above).empty());

  ClusterOptions max_at_boundary;
  max_at_boundary.min_height = 0.1;
  max_at_boundary.max_height = 0.25;
  EXPECT_EQ(run(points, max_at_boundary).size(), 1u);
}

TEST(ClusterObstacles, EmptyInputReturnsEmptyList)
{
  EXPECT_TRUE(cluster_obstacles(nullptr, 0, 3, ClusterOptions{}).empty());
}

TEST(ClusterObstacles, IndicesIndexTheInputArray)
{
  std::mt19937_64 rng(13);
  const double size[3]{1.0, 1.0, 1.6};
  const double a[3]{3.0, 0.0, 1.0};
  const double b[3]{15.0, 5.0, 1.0};
  std::vector<float> points;
  block(rng, a, size, 300, points);
  block(rng, b, size, 300, points);

  for (const auto & cluster : run(points, ClusterOptions{})) {
    EXPECT_EQ(static_cast<int>(cluster.indices.size()), cluster.count);
    int32_t previous = -1;
    for (int32_t row : cluster.indices) {
      EXPECT_GT(row, previous) << "indices must be ascending original rows";
      previous = row;
      for (int axis = 0; axis < 3; ++axis) {
        const double v = points[3 * static_cast<std::size_t>(row) +
          static_cast<std::size_t>(axis)];
        EXPECT_GE(v, cluster.min_bound[axis] - 1e-9);
        EXPECT_LE(v, cluster.max_bound[axis] + 1e-9);
      }
    }
  }
}

TEST(ClusterObstacles, IndicesSurviveNonFiniteRows)
{
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> points(15, nan);
  std::mt19937_64 rng(23);
  const double centre[3]{3.0, 0.0, 1.0};
  const double size[3]{1.0, 1.0, 1.6};
  block(rng, centre, size, 300, points);

  const auto clusters = run(points, ClusterOptions{});
  ASSERT_EQ(clusters.size(), 1u);
  ASSERT_FALSE(clusters[0].indices.empty());
  EXPECT_GE(
    *std::min_element(clusters[0].indices.begin(), clusters[0].indices.end()), 5)
    << "a port that compacts indices instead of carrying original rows fails here";
  for (int32_t row : clusters[0].indices) {
    for (int axis = 0; axis < 3; ++axis) {
      EXPECT_TRUE(
        std::isfinite(
          points[3 * static_cast<std::size_t>(row) + static_cast<std::size_t>(axis)]));
    }
  }
}

TEST(ClusterObstacles, SortedByCountAndTruncated)
{
  std::mt19937_64 rng(17);
  const double size[3]{1.0, 1.0, 1.6};
  const double a[3]{3.0, 0.0, 1.0};
  const double b[3]{10.0, 0.0, 1.0};
  const double c[3]{17.0, 0.0, 1.0};
  std::vector<float> points;
  block(rng, a, size, 100, points);
  block(rng, b, size, 400, points);
  block(rng, c, size, 250, points);

  const auto clusters = run(points, ClusterOptions{});
  ASSERT_EQ(clusters.size(), 3u);
  EXPECT_EQ(clusters[0].count, 400);
  EXPECT_EQ(clusters[1].count, 250);
  EXPECT_EQ(clusters[2].count, 100);

  ClusterOptions capped;
  capped.max_clusters = 2;
  const auto truncated = run(points, capped);
  ASSERT_EQ(truncated.size(), 2u);
  EXPECT_EQ(truncated[0].count, 400);
  EXPECT_EQ(truncated[1].count, 250);
}

TEST(ClusterObstacles, TiesKeepAscendingLabelOrder)
{
  // Two equal-sized blocks; the nearer one is labelled first because voxel
  // labels are ordered by smallest member index, and std::stable_sort must not
  // reorder them.
  std::mt19937_64 rng(29);
  const double size[3]{1.0, 1.0, 1.6};
  const double near[3]{3.0, 0.0, 1.0};
  const double far[3]{20.0, 0.0, 1.0};
  std::vector<float> points;
  block(rng, near, size, 200, points);
  block(rng, far, size, 200, points);

  const auto clusters = run(points, ClusterOptions{});
  ASSERT_EQ(clusters.size(), 2u);
  EXPECT_EQ(clusters[0].count, clusters[1].count);
  EXPECT_LT(clusters[0].centroid.x(), clusters[1].centroid.x());
}

TEST(ClusterObstacles, LinkRadiusBelowLatticeDiagonalThrowsOnAnEmptyCloud)
{
  ClusterOptions opts;
  opts.voxel_size = 0.5;
  opts.link_radius = 0.6;
  // Validation runs before any point is inspected, so the empty cloud still
  // reports the contract violation rather than silently returning {}.
  try {
    cluster_obstacles(nullptr, 0, 3, opts);
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument & error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("fragment"), std::string::npos) << message;
    EXPECT_NE(message.find("0.866"), std::string::npos) << message;
  }
}

TEST(ClusterObstacles, AcceptsAStrideOfFour)
{
  // An (N, 4) KITTI cloud with an intensity column is accepted; only the first
  // three columns are read, and a non-finite intensity is not a reason to drop
  // the row.
  std::mt19937_64 rng(31);
  const double centre[3]{3.0, 0.0, 1.0};
  const double size[3]{1.0, 1.0, 1.6};
  std::vector<float> xyz;
  block(rng, centre, size, 300, xyz);
  std::vector<float> xyzi;
  for (std::size_t i = 0; i < xyz.size() / 3; ++i) {
    xyzi.insert(xyzi.end(), {xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]});
    xyzi.push_back(std::numeric_limits<float>::quiet_NaN());
  }
  const auto clusters = cluster_obstacles(xyzi.data(), 300, 4, ClusterOptions{});
  ASSERT_EQ(clusters.size(), 1u);
  EXPECT_EQ(clusters[0].count, 300);
}

TEST(ClusterProperties, ExtentAndHeight)
{
  Cluster cluster;
  cluster.min_bound = Eigen::Vector3d(-1.0, -2.0, 0.0);
  cluster.max_bound = Eigen::Vector3d(1.0, 2.0, 1.5);
  EXPECT_NEAR(cluster.extent().x(), 2.0, 1e-12);
  EXPECT_NEAR(cluster.extent().y(), 4.0, 1e-12);
  EXPECT_NEAR(cluster.extent().z(), 1.5, 1e-12);
  EXPECT_NEAR(cluster.height(), 1.5, 1e-12);
}

TEST(ClusterObstacles, AReturnPastTheInt64LatticeDoesNotWrapTheVoxelSpan)
{
  // 1e25 / voxel_size does not fit in an int64. The plain cast that used to
  // build the voxel key is undefined there and produced INT64_MIN, which then
  // signed-overflowed `hi - lo` — the very quantity the 2^62 ravel guard exists
  // to judge — and the neighbour keys `base.x + dx` in voxel_labels.
  std::vector<float> xyz;
  for (int i = 0; i < 120; ++i) {
    // Spread in z as well, so the group clears the min_height gate.
    xyz.insert(
      xyz.end(),
      {2.0f + 0.02f * static_cast<float>(i), 0.0f, 0.01f * static_cast<float>(i % 60)});
  }
  xyz.insert(xyz.end(), {1e25f, 0.0f, 0.0f});

  ClusterOptions opts;
  opts.min_points = 10;
  const auto clusters = cluster_obstacles(xyz.data(), xyz.size() / 3, 3, opts);
  // The near group is still one cluster; the lone far return is its own
  // component and falls under min_points.
  ASSERT_EQ(clusters.size(), 1u);
  EXPECT_EQ(clusters[0].count, 120);
  EXPECT_LT(clusters[0].max_bound.x(), 100.0);
}
