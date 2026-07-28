// The C++ grid has to agree with ransac_seg/grid.py, not merely run. These
// mirror the cases in test/test_grid.py plus the ones that are only a hazard in
// C++: floor-versus-truncate on negative coordinates, and the corner rule an
// 8-connected search needs.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "ransac_seg_cpp/cost_grid.hpp"

namespace
{

using ransac_seg_cpp::CostGrid;
using ransac_seg_cpp::FREE;
using ransac_seg_cpp::INFLATED;
using ransac_seg_cpp::Neighbor;
using ransac_seg_cpp::OCCUPIED;
using ransac_seg_cpp::OptPlane;
using ransac_seg_cpp::Plane;
using ransac_seg_cpp::UNKNOWN;
using ransac_seg_cpp::kSqrt2;
using ransac_seg_cpp::neighbors;
using ransac_seg_cpp::plane_z;
using ransac_seg_cpp::snap_to_traversable;

/// An all-FREE, unit-cost grid; tests then carve obstacles into it.
CostGrid open_grid(int nx, int ny, double origin_x = -1.0, double origin_y = -1.0,
  double resolution = 0.5)
{
  CostGrid grid(origin_x, origin_y, resolution, nx, ny, OptPlane{});
  std::fill(grid.state.begin(), grid.state.end(), static_cast<int8_t>(FREE));
  std::fill(grid.cost.begin(), grid.cost.end(), 1.0f);
  return grid;
}

void block(CostGrid & grid, int ix, int iy)
{
  grid.state[grid.idx(ix, iy)] = OCCUPIED;
  grid.cost[grid.idx(ix, iy)] = std::numeric_limits<float>::infinity();
}

TEST(CostGrid, ConstructorFillsUnknownAndInfiniteCost)
{
  const CostGrid grid(-1.0, -2.0, 0.5, 3, 4, OptPlane{});
  EXPECT_EQ(grid.state.size(), 12u);
  EXPECT_EQ(grid.cost.size(), 12u);
  for (std::size_t k = 0; k < grid.state.size(); ++k) {
    EXPECT_EQ(grid.state[k], UNKNOWN);
    EXPECT_FALSE(std::isfinite(grid.cost[k]));
  }
  EXPECT_FALSE(grid.in_bounds(-1, 0));
  EXPECT_FALSE(grid.in_bounds(3, 0));
  EXPECT_TRUE(grid.in_bounds(2, 3));
  // x-major, so consecutive iy are adjacent.
  EXPECT_EQ(grid.idx(2, 3), 11u);
}

TEST(CostGrid, OriginIsTheLowerLeftCornerOfCellZero)
{
  const CostGrid grid(-3.0, 7.0, 0.5, 4, 4, OptPlane{});
  const auto centre = grid.cell_to_world(0, 0);
  EXPECT_DOUBLE_EQ(centre.first, grid.origin_x + 0.25);
  EXPECT_DOUBLE_EQ(centre.second, grid.origin_y + 0.25);
}

TEST(CostGrid, WorldToCellRoundTripsThroughCellCentres)
{
  const CostGrid grid(-12.25, -6.75, 0.5, 40, 30, OptPlane{});
  for (const auto & cell : {std::pair<int, int>{0, 0}, {1, 3}, {39, 29}, {17, 11}}) {
    const auto world = grid.cell_to_world(cell.first, cell.second);
    EXPECT_EQ(grid.world_to_cell(world.first, world.second), cell);
  }

  // Vector form must agree with the scalar one cell for cell.
  std::vector<double> xs, ys;
  std::vector<int32_t> ix(3), iy(3);
  for (const auto & cell : {std::pair<int, int>{0, 0}, {2, 5}, {39, 29}}) {
    const auto world = grid.cell_to_world(cell.first, cell.second);
    xs.push_back(world.first);
    ys.push_back(world.second);
  }
  grid.world_to_cell(xs.data(), ys.data(), 3, ix.data(), iy.data());
  EXPECT_EQ(ix[0], 0);
  EXPECT_EQ(iy[0], 0);
  EXPECT_EQ(ix[1], 2);
  EXPECT_EQ(iy[1], 5);
  EXPECT_EQ(ix[2], 39);
  EXPECT_EQ(iy[2], 29);
}

TEST(CostGrid, NegativeCoordinatesFloorRatherThanTruncate)
{
  // This is THE C++ defect: with origin 0 and resolution 0.5, a point at
  // x = -0.25 sits at raw index -0.5. numpy floors to -1; C++ int() truncation
  // toward zero would give 0 and fold the whole strip [-0.5, 0) into cell 0.
  const CostGrid grid(0.0, 0.0, 0.5, 10, 10, OptPlane{});
  EXPECT_EQ(grid.world_to_cell(-0.25, -0.25).first, -1);
  EXPECT_EQ(grid.world_to_cell(-0.25, -0.25).second, -1);
  EXPECT_EQ(grid.world_to_cell(-0.5, -0.5).first, -1);
  EXPECT_EQ(grid.world_to_cell(-0.6, -0.6).first, -2);
  EXPECT_EQ(grid.world_to_cell(0.0, 0.0).first, 0);
  EXPECT_EQ(grid.world_to_cell(0.4, 0.4).first, 0);

  // The vector form must floor identically.
  const double xs[] = {-0.25, -0.6, 0.4};
  const double ys[] = {-0.25, -0.6, 0.4};
  int32_t ix[3] = {}, iy[3] = {};
  grid.world_to_cell(xs, ys, 3, ix, iy);
  EXPECT_EQ(ix[0], -1);
  EXPECT_EQ(ix[1], -2);
  EXPECT_EQ(ix[2], 0);
  EXPECT_EQ(iy[0], -1);

  // Round-trip still holds on a grid with a negative origin, which is the
  // normal case (origin_x = xmin - resolution).
  const CostGrid negative(-7.0, -4.5, 0.5, 8, 8, OptPlane{});
  for (int i = 0; i < 8; ++i) {
    const auto world = negative.cell_to_world(i, i);
    EXPECT_EQ(negative.world_to_cell(world.first, world.second), std::make_pair(i, i));
  }
}

TEST(CostGrid, CellToWorldExtrapolatesOutOfRangeIndices)
{
  const CostGrid grid(0.0, 0.0, 0.5, 4, 4, OptPlane{});
  const auto below = grid.cell_to_world(-1, -2);
  EXPECT_DOUBLE_EQ(below.first, -0.25);
  EXPECT_DOUBLE_EQ(below.second, -0.75);
  EXPECT_EQ(grid.world_to_cell(below.first, below.second), std::make_pair(-1, -2));
}

TEST(CostGrid, IsTraversableNeedsBothFreeAndFiniteCost)
{
  CostGrid grid = open_grid(3, 3);
  EXPECT_TRUE(grid.is_traversable(1, 1));

  // FREE but infinitely costed is not traversable.
  grid.cost[grid.idx(1, 1)] = std::numeric_limits<float>::infinity();
  EXPECT_EQ(grid.state[grid.idx(1, 1)], FREE);
  EXPECT_FALSE(grid.is_traversable(1, 1));

  // Finitely costed but not FREE is not traversable either.
  grid.cost[grid.idx(1, 1)] = 1.0f;
  grid.state[grid.idx(1, 1)] = INFLATED;
  EXPECT_FALSE(grid.is_traversable(1, 1));

  // Out of bounds is false, not an out-of-range read.
  EXPECT_FALSE(grid.is_traversable(-1, 0));
  EXPECT_FALSE(grid.is_traversable(0, 3));
}

TEST(CostGrid, TraversableMaskMatchesThePredicate)
{
  CostGrid grid = open_grid(3, 4);
  block(grid, 1, 2);
  const auto mask = grid.traversable_mask();
  ASSERT_EQ(mask.size(), 12u);
  for (int ix = 0; ix < grid.nx; ++ix) {
    for (int iy = 0; iy < grid.ny; ++iy) {
      EXPECT_EQ(mask[grid.idx(ix, iy)] != 0, grid.is_traversable(ix, iy));
    }
  }
}

TEST(CostGrid, CellsWithStateAreLexicographicallyAscendingAndMayBeEmpty)
{
  CostGrid grid = open_grid(3, 3);
  block(grid, 2, 0);
  block(grid, 0, 2);
  block(grid, 0, 1);

  const auto occupied = grid.cells_with_state(OCCUPIED);
  ASSERT_EQ(occupied.size(), 3u);
  EXPECT_EQ(occupied[0], (std::array<int32_t, 2>{0, 1}));
  EXPECT_EQ(occupied[1], (std::array<int32_t, 2>{0, 2}));
  EXPECT_EQ(occupied[2], (std::array<int32_t, 2>{2, 0}));

  EXPECT_TRUE(grid.cells_with_state(UNKNOWN).empty());
}

TEST(CostGrid, CellToWorldXyzHandlesEmptyInputAndLiftsOntoThePlane)
{
  Plane plane;
  plane.normal = Eigen::Vector3d(0.0, 0.0, 1.0);
  plane.offset = 1.7;  // z = -1.7 everywhere
  const CostGrid grid(0.0, 0.0, 0.5, 4, 4, OptPlane{plane});

  EXPECT_TRUE(grid.cell_to_world_xyz({}).empty());

  const auto xyz = grid.cell_to_world_xyz({{0, 0}, {1, 2}});
  ASSERT_EQ(xyz.size(), 2u);
  EXPECT_DOUBLE_EQ(xyz[0][0], 0.25);
  EXPECT_DOUBLE_EQ(xyz[0][1], 0.25);
  EXPECT_DOUBLE_EQ(xyz[0][2], -1.7);
  EXPECT_DOUBLE_EQ(xyz[1][0], 0.75);
  EXPECT_DOUBLE_EQ(xyz[1][1], 1.25);
  EXPECT_DOUBLE_EQ(xyz[1][2], -1.7);
}

TEST(CostGrid, PlaneZIsZeroWithoutAPlaneAndForNearHorizontalNormals)
{
  EXPECT_DOUBLE_EQ(plane_z(OptPlane{}, 12.0, -3.0), 0.0);

  Plane vertical;
  vertical.normal = Eigen::Vector3d(1.0, 0.0, 5e-7);  // |nz| < 1e-6
  vertical.offset = 2.0;
  EXPECT_DOUBLE_EQ(plane_z(OptPlane{vertical}, 12.0, -3.0), 0.0);

  Plane tilted;
  tilted.normal = Eigen::Vector3d(0.1, 0.2, 0.9);
  tilted.offset = 1.0;
  const double expected = -(0.1 * 2.0 + 0.2 * 3.0 + 1.0) / 0.9;
  EXPECT_DOUBLE_EQ(plane_z(OptPlane{tilted}, 2.0, 3.0), expected);

  // The vector overload must agree, including on the degenerate paths.
  const double xs[] = {2.0, 0.0};
  const double ys[] = {3.0, 0.0};
  double out[2] = {};
  plane_z(OptPlane{tilted}, xs, ys, 2, out);
  EXPECT_DOUBLE_EQ(out[0], expected);
  EXPECT_DOUBLE_EQ(out[1], -1.0 / 0.9);
  plane_z(OptPlane{vertical}, xs, ys, 2, out);
  EXPECT_DOUBLE_EQ(out[0], 0.0);
  EXPECT_DOUBLE_EQ(out[1], 0.0);
}

TEST(CostGrid, NeighborsRejectAnyConnectivityOtherThanFourOrEight)
{
  const CostGrid grid = open_grid(3, 3);
  std::vector<Neighbor> out;
  EXPECT_THROW(neighbors(grid, 1, 1, 6, out), std::invalid_argument);
  try {
    neighbors(grid, 1, 1, 6, out);
  } catch (const std::invalid_argument & error) {
    EXPECT_STREQ(error.what(), "connectivity must be 4 or 8, got 6");
  }
}

TEST(CostGrid, NeighborsUseMetricStepLengthsAndClearTheOutput)
{
  const CostGrid grid = open_grid(3, 3);
  std::vector<Neighbor> out{{99, 99, 99.0}};

  neighbors(grid, 1, 1, 4, out);
  ASSERT_EQ(out.size(), 4u);  // the stale entry was cleared
  for (const auto & nb : out) {
    EXPECT_DOUBLE_EQ(nb.step_m, grid.resolution);
  }
  EXPECT_EQ(out[0].ix, 2);
  EXPECT_EQ(out[0].iy, 1);

  neighbors(grid, 1, 1, 8, out);
  ASSERT_EQ(out.size(), 8u);
  for (std::size_t i = 4; i < out.size(); ++i) {
    EXPECT_DOUBLE_EQ(out[i].step_m, grid.resolution * kSqrt2);
  }

  // A corner cell only offers what is on the grid.
  neighbors(grid, 0, 0, 8, out);
  EXPECT_EQ(out.size(), 3u);
}

TEST(CostGrid, DiagonalStepsAreBlockedByTheSqueezedCorner)
{
  CostGrid grid = open_grid(3, 3);
  // Two diagonally touching blocked cells around the (1,1) -> (2,2) diagonal.
  block(grid, 2, 1);
  block(grid, 1, 2);

  std::vector<Neighbor> out;
  neighbors(grid, 1, 1, 8, out);
  ASSERT_TRUE(grid.is_traversable(2, 2));
  for (const auto & nb : out) {
    EXPECT_FALSE(nb.ix == 2 && nb.iy == 2) << "walked through the shared corner";
  }

  // One free squeeze cell is not enough; both are required.
  CostGrid half = open_grid(3, 3);
  block(half, 2, 1);
  neighbors(half, 1, 1, 8, out);
  for (const auto & nb : out) {
    EXPECT_FALSE(nb.ix == 2 && nb.iy == 2);
  }

  // An UNKNOWN (not merely out-of-bounds) squeeze cell blocks too.
  CostGrid unknown = open_grid(3, 3);
  unknown.state[unknown.idx(1, 2)] = UNKNOWN;
  unknown.cost[unknown.idx(1, 2)] = std::numeric_limits<float>::infinity();
  neighbors(unknown, 1, 1, 8, out);
  for (const auto & nb : out) {
    EXPECT_FALSE(nb.ix == 2 && nb.iy == 2);
  }

  // With both squeezes free the diagonal is offered again.
  const CostGrid clear = open_grid(3, 3);
  neighbors(clear, 1, 1, 8, out);
  bool found = false;
  for (const auto & nb : out) {
    found = found || (nb.ix == 2 && nb.iy == 2);
  }
  EXPECT_TRUE(found);
}

TEST(CostGrid, SnapTiesResolveToTheSmallerCell)
{
  // Origin 0, resolution 1: centres at 0.5, 1.5, ... The query at (1.0, 1.0)
  // is exactly equidistant from all four cells around it, so the tie-break
  // decides. Ascending (ix, iy) means (0, 0).
  CostGrid grid = open_grid(4, 4, 0.0, 0.0, 1.0);
  const auto snapped = snap_to_traversable(grid, 1.0, 1.0, 2.0);
  ASSERT_TRUE(snapped.has_value());
  EXPECT_EQ(*snapped, std::make_pair(0, 0));

  // Blocking (0,0) moves it to the next cell in the same tie, (0,1).
  block(grid, 0, 0);
  const auto next = snap_to_traversable(grid, 1.0, 1.0, 2.0);
  ASSERT_TRUE(next.has_value());
  EXPECT_EQ(*next, std::make_pair(0, 1));
}

TEST(CostGrid, SnapRadiusIsInclusiveAndRejectsTheDegenerateCases)
{
  const CostGrid grid = open_grid(4, 4, 0.0, 0.0, 1.0);

  // Exactly on a cell centre: radius 0 must still find it.
  const auto exact = snap_to_traversable(grid, 0.5, 0.5, 0.0);
  ASSERT_TRUE(exact.has_value());
  EXPECT_EQ(*exact, std::make_pair(0, 0));

  // dist == radius is accepted (the centre of (0,0) is 0.5 away from x=1.0).
  const auto inclusive = snap_to_traversable(grid, 1.0, 0.5, 0.5);
  ASSERT_TRUE(inclusive.has_value());
  EXPECT_EQ(*inclusive, std::make_pair(0, 0));

  EXPECT_FALSE(snap_to_traversable(grid, 0.5, 0.5, -0.1).has_value());
  EXPECT_FALSE(snap_to_traversable(CostGrid(), 0.0, 0.0, 5.0).has_value());

  // Far outside the map: the window is empty, so nullopt rather than a clamp.
  EXPECT_FALSE(snap_to_traversable(grid, 500.0, 500.0, 1.0).has_value());

  // Inside the window but nothing traversable.
  CostGrid blocked = open_grid(2, 2, 0.0, 0.0, 1.0);
  for (int ix = 0; ix < 2; ++ix) {
    for (int iy = 0; iy < 2; ++iy) {
      block(blocked, ix, iy);
    }
  }
  EXPECT_FALSE(snap_to_traversable(blocked, 0.5, 0.5, 5.0).has_value());
}

TEST(CostGrid, SnapWorksFromANegativeQueryPoint)
{
  // The classic floor bug shows up here too: world_to_cell of a negative
  // coordinate has to land the window on the right side of the grid.
  const CostGrid grid = open_grid(4, 4, -2.0, -2.0, 1.0);
  const auto snapped = snap_to_traversable(grid, -1.4, -1.4, 0.5);
  ASSERT_TRUE(snapped.has_value());
  EXPECT_EQ(*snapped, std::make_pair(0, 0));  // centre (-1.5, -1.5)
}

TEST(CostGrid, SnapFromAFarFieldGoalIsRefusedNotWrapped)
{
  // astar.goal_x / astar.goal_y are unvalidated ROS doubles. 1e12 / 0.5 does
  // not fit in an int, and the plain cast that used to compute the centre cell
  // is undefined there: it produced INT_MIN, whose `- span` then overflowed and
  // flipped the window's sign. The answer was accidentally right; now it is
  // right by construction.
  const CostGrid grid = open_grid(41, 41, -10.0, -10.0, 0.5);
  EXPECT_FALSE(snap_to_traversable(grid, 1e12, 1e12, 5.0).has_value());
  EXPECT_FALSE(snap_to_traversable(grid, -1e12, 1e12, 5.0).has_value());
}

TEST(CostGrid, SnapFromANonFiniteGoalReturnsNullopt)
{
  // grid.py raises ValueError on int(nan); raising out of a planner callback
  // would take the node down, and "no traversable cell" is a state the caller
  // already publishes for.
  const CostGrid grid = open_grid(41, 41, -10.0, -10.0, 0.5);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(snap_to_traversable(grid, nan, nan, 5.0).has_value());
  EXPECT_FALSE(snap_to_traversable(grid, inf, 0.0, 5.0).has_value());
  EXPECT_FALSE(snap_to_traversable(grid, 0.0, 0.0, inf).has_value());
}

TEST(CostGrid, WorldToCellSaturatesInsteadOfWrapping)
{
  const CostGrid grid = open_grid(4, 4, 0.0, 0.0, 1.0);
  EXPECT_EQ(grid.world_to_cell(1e12, -1e12).first, std::numeric_limits<int>::max());
  EXPECT_EQ(grid.world_to_cell(1e12, -1e12).second, std::numeric_limits<int>::min());
}

}  // namespace
