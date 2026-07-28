// Port of the rasteriser half of test/test_grid.py, plus unit pins on the
// helpers the Python keeps private (_xy, _extent, _dims).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "ransac_seg_cpp/cost_grid.hpp"
#include "ransac_seg_cpp/costmap.hpp"

namespace
{

using ransac_seg_cpp::CostGrid;
using ransac_seg_cpp::CostmapOptions;
using ransac_seg_cpp::Dims;
using ransac_seg_cpp::FREE;
using ransac_seg_cpp::INFLATED;
using ransac_seg_cpp::OCCUPIED;
using ransac_seg_cpp::UNKNOWN;
using ransac_seg_cpp::build_costmap;
using ransac_seg_cpp::carve_ray_free;
using ransac_seg_cpp::dims;
using ransac_seg_cpp::extent;
using ransac_seg_cpp::filter_xy;
using ransac_seg_cpp::snap_to_traversable;

/// A dense flat sheet of ground returns covering a square patch, xyz stride 3.
std::vector<float> ground_patch(double step = 0.25, double extent_m = 10.0)
{
  std::vector<double> axis;
  for (double v = -extent_m; v <= extent_m + 1e-12; v += step) {
    axis.push_back(v);
  }
  std::vector<float> out;
  out.reserve(axis.size() * axis.size() * 3);
  for (double x : axis) {
    for (double y : axis) {
      out.push_back(static_cast<float>(x));
      out.push_back(static_cast<float>(y));
      out.push_back(0.0f);
    }
  }
  return out;
}

/// Ground hits arranged in concentric rings, as a spinning LiDAR gives.
std::vector<float> ring_returns(
  const std::vector<double> & ranges, int n_azimuth = 720)
{
  std::vector<float> out;
  for (double r : ranges) {
    for (int k = 0; k < n_azimuth; ++k) {
      const double a = -M_PI + 2.0 * M_PI * k / n_azimuth;
      out.push_back(static_cast<float>(r * std::cos(a)));
      out.push_back(static_cast<float>(r * std::sin(a)));
      out.push_back(0.0f);
    }
  }
  return out;
}

std::size_t point_count(const std::vector<float> & xyz) {return xyz.size() / 3;}

CostGrid rasterise(
  const std::vector<float> & ground, const std::vector<float> & obstacles,
  const CostmapOptions & opts)
{
  return build_costmap(
    ground.data(), point_count(ground), 3, obstacles.data(), point_count(obstacles), 3, opts);
}

int count_state(const CostGrid & grid, int8_t value)
{
  return static_cast<int>(std::count(grid.state.begin(), grid.state.end(), value));
}

const std::vector<float> kNoPoints;

// --- helper pins -----------------------------------------------------------

TEST(FilterXy, DropsNonFiniteXyButKeepsNonFiniteZ)
{
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const std::vector<float> pts = {
    1.0f, 2.0f, nan,      // a NaN Z must NOT drop the point: only xy is rasterised
    nan, 2.0f, 0.0f,
    1.0f, inf, 0.0f,
    3.0f, 4.0f, 5.0f};

  const auto kept = filter_xy(pts.data(), 4, 3);
  ASSERT_EQ(kept.size(), 2u);
  EXPECT_DOUBLE_EQ(kept[0][0], 1.0);
  EXPECT_DOUBLE_EQ(kept[0][1], 2.0);
  EXPECT_DOUBLE_EQ(kept[1][0], 3.0);
}

TEST(FilterXy, HonoursAStrideOfFourForKittiSweeps)
{
  const std::vector<float> xyzi = {1.0f, 2.0f, 3.0f, 0.5f, 4.0f, 5.0f, 6.0f, 0.9f};
  const auto kept = filter_xy(xyzi.data(), 2, 4);
  ASSERT_EQ(kept.size(), 2u);
  EXPECT_DOUBLE_EQ(kept[1][0], 4.0);
  EXPECT_DOUBLE_EQ(kept[1][1], 5.0);
}

TEST(Extent, IsAxisMajorAndSkipsEmptyClouds)
{
  const std::vector<std::array<double, 2>> a = {{-1.0, 5.0}, {2.0, 7.0}};
  const std::vector<std::array<double, 2>> none;

  const auto both = extent(a, none);
  ASSERT_TRUE(both.has_value());
  // (xmin, xmax, ymin, ymax) — not (xmin, ymin, xmax, ymax).
  EXPECT_DOUBLE_EQ((*both)[0], -1.0);
  EXPECT_DOUBLE_EQ((*both)[1], 2.0);
  EXPECT_DOUBLE_EQ((*both)[2], 5.0);
  EXPECT_DOUBLE_EQ((*both)[3], 7.0);

  EXPECT_FALSE(extent(none, none).has_value());
}

TEST(Dims, PadsExactlyOneCellBelowAndLeft)
{
  const Dims d = dims({0.0, 2.0, 0.0, 2.0}, 0.5);
  EXPECT_DOUBLE_EQ(d.origin_x, -0.5);
  EXPECT_DOUBLE_EQ(d.origin_y, -0.5);
  EXPECT_EQ(d.nx, 6);
  EXPECT_EQ(d.ny, 6);

  // A single point still gets a cell either side of it.
  const Dims degenerate = dims({1.0, 1.0, 1.0, 1.0}, 0.5);
  EXPECT_DOUBLE_EQ(degenerate.origin_x, 0.5);
  EXPECT_EQ(degenerate.nx, 2);
}

// --- rasteriser ------------------------------------------------------------

TEST(Costmap, OriginIsTheLowerLeftCornerOfCellZero)
{
  CostmapOptions opts;
  opts.resolution = 0.5;
  const CostGrid grid = rasterise(ground_patch(), kNoPoints, opts);

  const auto centre = grid.cell_to_world(0, 0);
  EXPECT_NEAR(centre.first, grid.origin_x + 0.25, 1e-12);
  EXPECT_NEAR(centre.second, grid.origin_y + 0.25, 1e-12);
}

TEST(Costmap, ObstacleBeatsGroundInTheSameCell)
{
  std::vector<float> ground, obstacles;
  for (int i = 0; i < 20; ++i) {
    ground.insert(ground.end(), {1.1f, 1.1f, 0.0f});
    obstacles.insert(obstacles.end(), {1.1f, 1.1f, 1.0f});
  }
  CostmapOptions opts;
  opts.bounds = std::array<double, 4>{0.0, 2.0, 0.0, 2.0};
  const CostGrid grid = rasterise(ground, obstacles, opts);

  const auto cell = grid.world_to_cell(1.1, 1.1);
  EXPECT_EQ(grid.state[grid.idx(cell.first, cell.second)], OCCUPIED);
  EXPECT_FALSE(grid.is_traversable(cell.first, cell.second));
}

TEST(Costmap, UnknownIsDistinctFromFreeAndUntraversable)
{
  std::vector<float> ground;
  for (int i = 0; i < 5; ++i) {
    ground.insert(ground.end(), {0.25f, 0.25f, 0.0f});
  }
  CostmapOptions opts;
  opts.bounds = std::array<double, 4>{0.0, 5.0, 0.0, 5.0};
  const CostGrid grid = rasterise(ground, kNoPoints, opts);

  const auto seen = grid.world_to_cell(0.25, 0.25);
  const auto unseen = grid.world_to_cell(4.25, 4.25);
  EXPECT_EQ(grid.state[grid.idx(seen.first, seen.second)], FREE);
  EXPECT_EQ(grid.state[grid.idx(unseen.first, unseen.second)], UNKNOWN);
  EXPECT_TRUE(grid.is_traversable(seen.first, seen.second));
  EXPECT_FALSE(grid.is_traversable(unseen.first, unseen.second));
  EXPECT_FALSE(std::isfinite(grid.cost[grid.idx(unseen.first, unseen.second)]));
}

TEST(Costmap, SingleObstacleInflatesADiscOfTheRightRadius)
{
  const double resolution = 0.25;
  const double robot_radius = 1.0;
  std::vector<float> obstacles;
  for (int i = 0; i < 10; ++i) {
    obstacles.insert(obstacles.end(), {0.0f, 0.0f, 1.0f});
  }
  CostmapOptions opts;
  opts.resolution = resolution;
  opts.robot_radius = robot_radius;
  opts.bounds = std::array<double, 4>{-10.0, 10.0, -10.0, 10.0};
  const CostGrid grid = rasterise(ground_patch(0.1), obstacles, opts);

  const int inflated = count_state(grid, INFLATED);
  const int occupied = count_state(grid, OCCUPIED);
  // The disc is padded by half a cell diagonal, because the return that made a
  // cell OCCUPIED can sit that far from the centre the transform measures from.
  const double padded = robot_radius + 0.5 * std::sqrt(2.0) * resolution;
  const double analytic = M_PI * padded * padded / (resolution * resolution);

  EXPECT_GE(occupied, 1);
  EXPECT_GE(inflated + occupied, 0.75 * analytic);
  EXPECT_LE(inflated + occupied, 1.35 * analytic);
}

TEST(Costmap, EveryFiniteCostIsAtLeastOneAndOnlyOnFreeCells)
{
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> uni(-8.0, 8.0);
  std::vector<float> obstacles;
  for (int i = 0; i < 400; ++i) {
    obstacles.push_back(static_cast<float>(uni(rng)));
    obstacles.push_back(static_cast<float>(uni(rng)));
    obstacles.push_back(1.0f);
  }
  CostmapOptions opts;
  opts.resolution = 0.25;
  const CostGrid grid = rasterise(ground_patch(0.1), obstacles, opts);

  int finite = 0;
  for (std::size_t i = 0; i < grid.cost.size(); ++i) {
    if (!std::isfinite(grid.cost[i])) {
      continue;
    }
    ++finite;
    EXPECT_GE(grid.cost[i], 1.0f) << "A* admissibility floor";
    EXPECT_EQ(grid.state[i], FREE) << "blocked states are never finitely costed";
  }
  EXPECT_GT(finite, 0);
}

TEST(Costmap, CostDecaysWithDistanceFromTheObstacle)
{
  std::vector<float> obstacles;
  for (int i = 0; i < 10; ++i) {
    obstacles.insert(obstacles.end(), {0.0f, 0.0f, 1.0f});
  }
  CostmapOptions opts;
  opts.resolution = 0.25;
  opts.robot_radius = 0.5;
  opts.inflation_radius = 3.0;
  opts.bounds = std::array<double, 4>{-10.0, 10.0, -10.0, 10.0};
  const CostGrid grid = rasterise(ground_patch(0.1), obstacles, opts);

  const auto at = [&grid](double x, double y) {
      const auto c = grid.world_to_cell(x, y);
      return grid.cost[grid.idx(c.first, c.second)];
    };
  const float near = at(0.8, 0.0);
  const float far = at(2.5, 0.0);
  const float edge = at(8.0, 0.0);

  EXPECT_GT(far, 1.0f);
  EXPECT_LT(far, near);
  EXPECT_NEAR(edge, 1.0f, 1e-6);
}

TEST(Costmap, EmptyInputsGiveAOneCellUnknownGridAtTheRequestedResolution)
{
  CostmapOptions opts;
  opts.resolution = 0.37;
  const CostGrid grid = rasterise(kNoPoints, kNoPoints, opts);

  EXPECT_EQ(grid.nx, 1);
  EXPECT_EQ(grid.ny, 1);
  EXPECT_DOUBLE_EQ(grid.origin_x, 0.0);
  EXPECT_DOUBLE_EQ(grid.origin_y, 0.0);
  // The requested spacing is never coarsened on a blank frame.
  EXPECT_DOUBLE_EQ(grid.resolution, 0.37);
  EXPECT_EQ(grid.state[0], UNKNOWN);
  EXPECT_FALSE(std::isfinite(grid.cost[0]));
  EXPECT_TRUE(grid.cells_with_state(FREE).empty());
}

TEST(Costmap, AllObstacleInputLeavesNothingTraversable)
{
  std::mt19937 rng(2);
  std::uniform_real_distribution<double> uni(-5.0, 5.0);
  std::vector<float> obstacles;
  for (int i = 0; i < 600; ++i) {
    obstacles.push_back(static_cast<float>(uni(rng)));
    obstacles.push_back(static_cast<float>(uni(rng)));
    obstacles.push_back(static_cast<float>(uni(rng)));
  }
  const CostGrid grid = rasterise(kNoPoints, obstacles, CostmapOptions{});

  const auto mask = grid.traversable_mask();
  EXPECT_EQ(std::count(mask.begin(), mask.end(), uint8_t(1)), 0);
  EXPECT_EQ(count_state(grid, FREE), 0);
}

TEST(Costmap, SingleGroundPointIsFree)
{
  const std::vector<float> ground = {2.0f, 3.0f, 0.0f};
  const CostGrid grid = rasterise(ground, kNoPoints, CostmapOptions{});

  const auto cell = grid.world_to_cell(2.0, 3.0);
  EXPECT_TRUE(grid.is_traversable(cell.first, cell.second));
  const auto free_cells = grid.cells_with_state(FREE);
  ASSERT_EQ(free_cells.size(), 1u);
  EXPECT_EQ(free_cells[0][0], cell.first);
  EXPECT_EQ(free_cells[0][1], cell.second);
}

TEST(Costmap, MinPointThresholdsAreRespected)
{
  const std::vector<float> ground = {0.25f, 0.25f, 0.0f};
  const std::vector<float> obstacles = {1.25f, 0.25f, 1.0f, 1.26f, 0.26f, 1.0f};
  CostmapOptions opts;
  opts.bounds = std::array<double, 4>{0.0, 2.0, 0.0, 2.0};
  opts.min_ground_points = 2;
  opts.min_obstacle_points = 3;
  const CostGrid grid = rasterise(ground, obstacles, opts);

  const auto a = grid.world_to_cell(0.25, 0.25);
  const auto b = grid.world_to_cell(1.25, 0.25);
  EXPECT_EQ(grid.state[grid.idx(a.first, a.second)], UNKNOWN);
  EXPECT_EQ(grid.state[grid.idx(b.first, b.second)], UNKNOWN);
}

TEST(Costmap, NonPositiveThresholdsBehaveAsOne)
{
  const std::vector<float> ground = {0.25f, 0.25f, 0.0f};
  CostmapOptions opts;
  opts.bounds = std::array<double, 4>{0.0, 2.0, 0.0, 2.0};
  opts.min_ground_points = 0;
  const CostGrid grid = rasterise(ground, kNoPoints, opts);

  const auto cell = grid.world_to_cell(0.25, 0.25);
  EXPECT_EQ(grid.state[grid.idx(cell.first, cell.second)], FREE);
}

TEST(Costmap, PointsOutsideExplicitBoundsAreDropped)
{
  const std::vector<float> ground = {
    0.25f, 0.25f, 0.0f, 50.0f, 50.0f, 0.0f, -50.0f, -50.0f, 0.0f};
  CostmapOptions opts;
  opts.bounds = std::array<double, 4>{0.0, 2.0, 0.0, 2.0};
  const CostGrid grid = rasterise(ground, kNoPoints, opts);

  // Dropped, not wrapped or clamped onto a boundary cell.
  EXPECT_EQ(count_state(grid, FREE), 1);
}

TEST(Costmap, AbsurdExtentIsClampedToTheGround)
{
  std::vector<float> obstacles;
  for (int i = 0; i < 5; ++i) {
    obstacles.insert(obstacles.end(), {10000.0f, 10000.0f, 1.0f});
  }
  CostmapOptions opts;
  opts.resolution = 0.05;
  const CostGrid grid = rasterise(ground_patch(0.5, 5.0), obstacles, opts);

  EXPECT_LE(static_cast<int64_t>(grid.nx) * grid.ny, ransac_seg_cpp::kMaxCells);
  const auto mask = grid.traversable_mask();
  EXPECT_GT(std::count(mask.begin(), mask.end(), uint8_t(1)), 0);
}

TEST(Costmap, AnUnaffordableResolutionCoarsensRatherThanCrops)
{
  const std::vector<float> ground = ground_patch(0.5, 30.0);
  CostmapOptions opts;
  opts.resolution = 0.02;
  const CostGrid grid = rasterise(ground, kNoPoints, opts);

  EXPECT_LE(static_cast<int64_t>(grid.nx) * grid.ny, ransac_seg_cpp::kMaxCells);
  EXPECT_GT(grid.resolution, 0.02);
  // Coarsen, never crop: the farthest ground return is still on the map.
  const auto far = grid.world_to_cell(30.0, 30.0);
  EXPECT_TRUE(grid.in_bounds(far.first, far.second));
  EXPECT_GT(count_state(grid, FREE), 0);
}

TEST(Costmap, FreeCellsClearTheRobotRadiusFromTheActualReturns)
{
  const double resolution = 0.5, robot_radius = 0.6;
  CostmapOptions opts;
  opts.resolution = resolution;
  opts.robot_radius = robot_radius;
  opts.bounds = std::array<double, 4>{-10.0, 10.0, -10.0, 10.0};

  const CostGrid probe = rasterise(ground_patch(0.25), kNoPoints, opts);
  // Park the return in the far corner of a cell, the worst case for the
  // centre-to-centre approximation the transform makes.
  const auto centre = probe.cell_to_world(20, 20);
  const double ox = centre.first + 0.49 * resolution;
  const double oy = centre.second + 0.49 * resolution;
  std::vector<float> corner;
  for (int i = 0; i < 5; ++i) {
    corner.push_back(static_cast<float>(ox));
    corner.push_back(static_cast<float>(oy));
    corner.push_back(1.0f);
  }

  const CostGrid grid = rasterise(ground_patch(0.25), corner, opts);
  double closest = std::numeric_limits<double>::infinity();
  for (const auto & cell : grid.cells_with_state(FREE)) {
    const auto w = grid.cell_to_world(cell[0], cell[1]);
    closest = std::min(closest, std::hypot(w.first - ox, w.second - oy));
  }
  EXPECT_GE(closest, robot_radius);
}

// --- ray carving -----------------------------------------------------------

TEST(Costmap, RayCarvingBridgesTheGapsBetweenBeamRings)
{
  std::vector<double> ranges;
  for (double r = 4.0; r < 30.0; r += 3.0) {
    ranges.push_back(r);
  }
  const std::vector<float> ground = ring_returns(ranges);

  CostmapOptions without;
  without.ray_fill = false;
  CostmapOptions with_fill;
  with_fill.ray_fill = true;

  const CostGrid a = rasterise(ground, kNoPoints, without);
  const CostGrid b = rasterise(ground, kNoPoints, with_fill);

  EXPECT_GT(count_state(b, FREE), 3 * count_state(a, FREE));
  const auto cell = b.world_to_cell(5.5, 0.0);
  EXPECT_EQ(a.state[a.idx(cell.first, cell.second)], UNKNOWN);
  EXPECT_EQ(b.state[b.idx(cell.first, cell.second)], FREE);
}

TEST(Costmap, RayCarvingNeverOverridesAnObstacle)
{
  std::vector<double> ranges;
  for (double r = 4.0; r < 20.0; r += 3.0) {
    ranges.push_back(r);
  }
  const std::vector<float> ground = ring_returns(ranges);
  std::vector<float> obstacles;
  for (int i = 0; i < 40; ++i) {
    obstacles.push_back(static_cast<float>(8.0 + (-0.2 + 0.4 * i / 39.0)));
    obstacles.push_back(0.0f);
    obstacles.push_back(1.0f);
  }
  const CostGrid grid = rasterise(ground, obstacles, CostmapOptions{});

  const auto cell = grid.world_to_cell(8.0, 0.0);
  EXPECT_EQ(grid.state[grid.idx(cell.first, cell.second)], OCCUPIED);
}

TEST(Costmap, RayCarvingStopsAtTheFarthestReturn)
{
  std::vector<double> ranges;
  for (double r = 4.0; r < 16.0; r += 2.0) {
    ranges.push_back(r);
  }
  const CostGrid grid = rasterise(ring_returns(ranges), kNoPoints, CostmapOptions{});

  double farthest = 0.0;
  for (const auto & cell : grid.cells_with_state(FREE)) {
    const auto w = grid.cell_to_world(cell[0], cell[1]);
    farthest = std::max(farthest, std::hypot(w.first, w.second));
  }
  // One cell diagonal of slack: the farthest ring sits at 14 m.
  EXPECT_LE(farthest, 14.0 + grid.resolution * 1.5);
}

TEST(Costmap, ALoneReturnDoesNotCarveASpoke)
{
  const std::vector<float> ground = {
    25.0f, 0.0f, 0.0f, 3.0f, 8.0f, 0.0f, 3.2f, 8.1f, 0.0f};
  CostmapOptions opts;
  opts.ray_fill_min_cells = 2;
  const CostGrid grid = rasterise(ground, kNoPoints, opts);

  const auto cell = grid.world_to_cell(12.0, 0.0);
  EXPECT_EQ(grid.state[grid.idx(cell.first, cell.second)], UNKNOWN);
}

TEST(Costmap, RayCarvingReachesTheSensorBlindSpot)
{
  std::vector<double> ranges;
  for (double r = 6.0; r < 24.0; r += 2.0) {
    ranges.push_back(r);
  }
  const std::vector<float> ground = ring_returns(ranges);

  CostmapOptions without;
  without.ray_fill = false;
  const CostGrid a = rasterise(ground, kNoPoints, without);
  const CostGrid b = rasterise(ground, kNoPoints, CostmapOptions{});

  EXPECT_FALSE(snap_to_traversable(a, 0.0, 0.0, 2.0).has_value());
  EXPECT_TRUE(snap_to_traversable(b, 0.0, 0.0, 2.0).has_value());
}

TEST(CarveRayFree, OnlyPromotesUnknownCells)
{
  CostGrid grid(-5.0, -5.0, 1.0, 10, 10, std::nullopt);
  grid.state[grid.idx(9, 5)] = FREE;
  grid.state[grid.idx(8, 5)] = FREE;
  grid.state[grid.idx(7, 5)] = OCCUPIED;
  grid.state[grid.idx(6, 5)] = INFLATED;

  carve_ray_free(grid, 0.0, 0.0, 720, 2);

  EXPECT_EQ(grid.state[grid.idx(7, 5)], OCCUPIED) << "carving never downgrades a blocker";
  EXPECT_EQ(grid.state[grid.idx(6, 5)], INFLATED);
}

TEST(CarveRayFree, IsANoOpWithoutAnyFreeCell)
{
  CostGrid grid(-5.0, -5.0, 1.0, 10, 10, std::nullopt);
  grid.state[grid.idx(4, 4)] = OCCUPIED;
  const std::vector<int8_t> before = grid.state;

  carve_ray_free(grid, 0.0, 0.0, 720, 2);

  EXPECT_EQ(grid.state, before);
}

}  // namespace
