// Tests for the costmap searches: optimality, admissibility and failure modes.
//
// Port of test/test_planners.py. The Python tests build every grid through
// build_costmap from synthetic point sets rather than by hand-filling arrays;
// this file does the same through a local rasteriser (`rasterise` below) that
// reproduces build_costmap's quantisation, exact Euclidean clearance transform
// and cost decay. Ray carving is deliberately absent — none of the scenes here
// depend on it, and keeping the helper in this file means the planner tests do
// not link against the costmap track.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include "ransac_seg_cpp/planners.hpp"

using ransac_seg_cpp::Cell;
using ransac_seg_cpp::CostGrid;
using ransac_seg_cpp::DijkstraField;
using ransac_seg_cpp::FREE;
using ransac_seg_cpp::INFLATED;
using ransac_seg_cpp::OCCUPIED;
using ransac_seg_cpp::UNKNOWN;
using ransac_seg_cpp::astar;
using ransac_seg_cpp::band_color;
using ransac_seg_cpp::bfs;
using ransac_seg_cpp::dfs;
using ransac_seg_cpp::dijkstra;
using ransac_seg_cpp::edge_cost;
using ransac_seg_cpp::furthest_reachable;
using ransac_seg_cpp::manhattan;
using ransac_seg_cpp::normalised_bands;
using ransac_seg_cpp::octile;
using ransac_seg_cpp::path_cost;
using ransac_seg_cpp::path_length_m;
using ransac_seg_cpp::pull_onto_grid;
using ransac_seg_cpp::reconstruct_path;
using ransac_seg_cpp::select_goal;
using ransac_seg_cpp::snap_to_traversable;
using ransac_seg_cpp::step_length;
using ransac_seg_cpp::validate_connectivity;

namespace
{

constexpr double kResolution = 0.5;
constexpr double kInf = std::numeric_limits<double>::infinity();

struct Point
{
  double x;
  double y;
};

/// A dense sheet of returns covering a rectangle, as test_planners.py's sheet().
std::vector<Point> sheet(
  double x0, double x1, double y0, double y1, double step = 0.2)
{
  std::vector<Point> pts;
  for (double x = x0; x <= x1 + 1e-9; x += step) {
    for (double y = y0; y <= y1 + 1e-9; y += step) {
      pts.push_back({x, y});
    }
  }
  return pts;
}

void extend(std::vector<Point> & into, const std::vector<Point> & more)
{
  into.insert(into.end(), more.begin(), more.end());
}

/// build_costmap without ray carving: quantise, stamp, inflate, decay.
///
/// The clearance transform is brute force over the occupied cells, which is
/// exact (scipy's EDT is exact too) and fast enough for the scenes here.
CostGrid rasterise(
  const std::vector<Point> & ground, const std::vector<Point> & obstacles,
  double resolution = kResolution, double robot_radius = 0.6,
  double inflation_radius = 1.5, double cost_scaling = 3.0,
  int min_obstacle_points = 3)
{
  double xmin = kInf, xmax = -kInf, ymin = kInf, ymax = -kInf;
  for (const std::vector<Point> * set : {&ground, &obstacles}) {
    for (const Point & p : *set) {
      xmin = std::min(xmin, p.x);
      xmax = std::max(xmax, p.x);
      ymin = std::min(ymin, p.y);
      ymax = std::max(ymax, p.y);
    }
  }
  const double origin_x = xmin - resolution;
  const double origin_y = ymin - resolution;
  const int nx = std::max(1, static_cast<int>(std::ceil((xmax + resolution - origin_x) / resolution)));
  const int ny = std::max(1, static_cast<int>(std::ceil((ymax + resolution - origin_y) / resolution)));

  CostGrid grid(origin_x, origin_y, resolution, nx, ny, std::nullopt);
  std::vector<int> ground_counts(static_cast<std::size_t>(nx) * ny, 0);
  std::vector<int> obstacle_counts(ground_counts.size(), 0);
  auto tally = [&](const std::vector<Point> & pts, std::vector<int> & counts) {
      for (const Point & p : pts) {
        const auto cell = grid.world_to_cell(p.x, p.y);
        if (grid.in_bounds(cell.first, cell.second)) {
          counts[grid.idx(cell.first, cell.second)] += 1;
        }
      }
    };
  tally(ground, ground_counts);
  tally(obstacles, obstacle_counts);

  std::vector<Cell> occupied_cells;
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      const std::size_t k = grid.idx(ix, iy);
      if (ground_counts[k] >= 1) {
        grid.state[k] = FREE;
      }
      if (obstacle_counts[k] >= std::max(1, min_obstacle_points)) {
        grid.state[k] = OCCUPIED;  // an obstacle on inlier road still blocks
        occupied_cells.push_back({ix, iy});
      }
    }
  }

  std::vector<double> clearance(ground_counts.size(), kInf);
  if (!occupied_cells.empty()) {
    for (int ix = 0; ix < nx; ++ix) {
      for (int iy = 0; iy < ny; ++iy) {
        double best = kInf;
        for (const Cell & c : occupied_cells) {
          const double dx = static_cast<double>(ix - c.first) * resolution;
          const double dy = static_cast<double>(iy - c.second) * resolution;
          best = std::min(best, std::hypot(dx, dy));
        }
        // The return that made a cell OCCUPIED sits anywhere inside it, so the
        // centre-to-centre transform overstates clearance by half a diagonal.
        clearance[grid.idx(ix, iy)] = std::max(best - 0.5 * std::sqrt(2.0) * resolution, 0.0);
      }
    }
  }

  for (std::size_t k = 0; k < clearance.size(); ++k) {
    if (grid.state[k] == FREE && clearance[k] <= robot_radius) {
      grid.state[k] = INFLATED;
    }
  }
  for (std::size_t k = 0; k < clearance.size(); ++k) {
    if (grid.state[k] != FREE) {
      continue;
    }
    grid.cost[k] = 1.0f;
    if (clearance[k] < inflation_radius) {
      grid.cost[k] = static_cast<float>(
        1.0 + std::exp(-cost_scaling * (clearance[k] - robot_radius)));
    }
  }
  return grid;
}

/// A straight corridor bounded by obstacle walls at y = +/- 4.
CostGrid corridor_grid(double inflation_radius = 1.5, double cost_scaling = 3.0)
{
  const std::vector<Point> ground = sheet(0.0, 20.0, -4.0, 4.0);
  std::vector<Point> walls = sheet(0.0, 20.0, 3.9, 4.4);
  extend(walls, sheet(0.0, 20.0, -4.4, -3.9));
  return rasterise(ground, walls, kResolution, 0.6, inflation_radius, cost_scaling);
}

/// An open patch peppered with obstacle blobs; deterministic, not numpy's RNG.
CostGrid random_grid(uint32_t seed = 7)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> cx_dist(2.0, 13.0);
  std::uniform_real_distribution<double> cy_dist(-6.0, 6.0);
  std::normal_distribution<double> jitter(0.0, 0.2);

  const std::vector<Point> ground = sheet(0.0, 15.0, -7.0, 7.0);
  std::vector<Point> blobs;
  for (int b = 0; b < 12; ++b) {
    const double cx = cx_dist(rng);
    const double cy = cy_dist(rng);
    for (int i = 0; i < 40; ++i) {
      blobs.push_back({cx + jitter(rng), cy + jitter(rng)});
    }
  }
  return rasterise(ground, blobs);
}

/// A 12 m square of ground split corner to corner by a one-cell diagonal wall.
CostGrid diagonal_wall_grid()
{
  std::vector<Point> ground = sheet(0.0, 12.0, 0.0, 12.0, 0.5);
  std::vector<Point> wall;
  for (int k = 1; k <= 12; ++k) {
    for (int r = 0; r < 5; ++r) {
      wall.push_back({k + 0.5, k + 0.5});
    }
  }
  return rasterise(ground, wall, 1.0, 0.1, 1.5, 3.0);
}

/// Two swept sheets with an unobserved band between them, joined only at x > 11.
CostGrid unknown_band_grid()
{
  std::vector<Point> ground = sheet(0.0, 15.0, -6.0, -1.0);
  extend(ground, sheet(0.0, 15.0, 1.0, 6.0));
  extend(ground, sheet(11.0, 15.0, -1.0, 1.0));
  const std::vector<Point> obstacles = sheet(6.0, 7.0, -6.0, -3.0);
  // No ray carving on purpose: the scene is physically impossible for a real
  // sweep (ground at x > 11 on azimuths whose nearer span is unobserved), and
  // carving would erase the UNKNOWN band this test exists to isolate.
  return rasterise(ground, obstacles);
}

/// Ground with a wall sealing the corridor at x ~ 10.
CostGrid sealed_grid()
{
  const std::vector<Point> ground = sheet(0.0, 20.0, -4.0, 4.0);
  const std::vector<Point> wall = sheet(9.5, 10.5, -4.5, 4.5);
  return rasterise(ground, wall);
}

std::vector<Cell> reachable_cells(const std::vector<int32_t> & field, const CostGrid & grid)
{
  std::vector<Cell> cells;
  for (int ix = 0; ix < grid.nx; ++ix) {
    for (int iy = 0; iy < grid.ny; ++iy) {
      if (field[grid.idx(ix, iy)] >= 0) {
        cells.push_back({ix, iy});
      }
    }
  }
  return cells;
}

/// An independent hand-rolled 4-connected flood, as the Python reference does.
std::map<Cell, int> brute_force_flood(const CostGrid & grid, Cell start)
{
  std::map<Cell, int> hops{{start, 0}};
  std::vector<Cell> frontier{start};
  while (!frontier.empty()) {
    std::vector<Cell> next;
    for (const Cell & c : frontier) {
      const int dxs[4] = {1, -1, 0, 0};
      const int dys[4] = {0, 0, 1, -1};
      for (int s = 0; s < 4; ++s) {
        const Cell nb{c.first + dxs[s], c.second + dys[s]};
        if (!grid.is_traversable(nb.first, nb.second) || hops.count(nb)) {
          continue;
        }
        hops[nb] = hops[c] + 1;
        next.push_back(nb);
      }
    }
    frontier = next;
  }
  return hops;
}

}  // namespace

TEST(Heuristics, ScaleWithResolutionAndReportMetres)
{
  EXPECT_DOUBLE_EQ(octile({0, 0}, {3, 0}, 0.5), 1.5);
  EXPECT_NEAR(octile({0, 0}, {3, 3}, 0.5), 3.0 * std::sqrt(2.0) * 0.5, 1e-12);
  EXPECT_DOUBLE_EQ(manhattan({0, 0}, {3, 3}, 0.5), 3.0);
  EXPECT_DOUBLE_EQ(octile({2, 5}, {2, 5}, 0.5), 0.0);
  // Symmetric and sign-blind.
  EXPECT_DOUBLE_EQ(octile({7, 1}, {2, 4}, 0.25), octile({2, 4}, {7, 1}, 0.25));
}

TEST(ValidateConnectivity, RejectsAnythingButFourAndEight)
{
  EXPECT_NO_THROW(validate_connectivity(4));
  EXPECT_NO_THROW(validate_connectivity(8));
  EXPECT_THROW(validate_connectivity(6), std::invalid_argument);
  EXPECT_THROW(validate_connectivity(0), std::invalid_argument);
  try {
    validate_connectivity(6);
    FAIL() << "expected a throw";
  } catch (const std::invalid_argument & err) {
    EXPECT_STREQ(err.what(), "connectivity must be 4 or 8, got 6");
  }
}

TEST(StepLength, ThrowsForNonAdjacentAndSelfLoops)
{
  const CostGrid grid(0.0, 0.0, 0.5, 4, 4, std::nullopt);
  EXPECT_DOUBLE_EQ(step_length(grid, {1, 1}, {2, 1}), 0.5);
  EXPECT_DOUBLE_EQ(step_length(grid, {1, 1}, {2, 2}), 0.5 * std::sqrt(2.0));
  EXPECT_THROW(step_length(grid, {1, 1}, {3, 1}), std::invalid_argument);
  EXPECT_THROW(step_length(grid, {1, 1}, {1, 1}), std::invalid_argument);
}

TEST(EdgeCost, IsTheStepTimesTheMeanCellCost)
{
  CostGrid grid(0.0, 0.0, 0.5, 2, 2, std::nullopt);
  grid.state.assign(grid.state.size(), FREE);
  grid.cost[grid.idx(0, 0)] = 1.0f;
  grid.cost[grid.idx(1, 0)] = 3.0f;
  const double step = step_length(grid, {0, 0}, {1, 0});
  EXPECT_DOUBLE_EQ(edge_cost(grid, {0, 0}, {1, 0}, step), 0.5 * 0.5 * 4.0);
}

TEST(AStar, MatchesTheDijkstraCostAtTheGoal)
{
  const CostGrid grid = random_grid();
  const auto start = snap_to_traversable(grid, 0.5, 0.0, 2.0);
  const auto goal = snap_to_traversable(grid, 14.0, 0.0, 3.0);
  ASSERT_TRUE(start.has_value());
  ASSERT_TRUE(goal.has_value());

  const DijkstraField field = dijkstra(grid, *start, 8);
  const auto result = astar(grid, *start, *goal, 8);

  ASSERT_FALSE(result.path.empty());
  EXPECT_EQ(result.path.front(), *start);
  EXPECT_EQ(result.path.back(), *goal);
  const double reference = static_cast<double>(field.cost_at(goal->first, goal->second));
  EXPECT_NEAR(result.cost, reference, 1e-4 * reference);
  // The reconstructed Dijkstra route must cost the same as A*'s even when the
  // two pick different tie-broken routes of equal cost.
  EXPECT_NEAR(path_cost(grid, reconstruct_path(field, *goal)), result.cost, 1e-4 * result.cost);
}

TEST(AStar, NeverEntersABlockedOrUnobservedCell)
{
  const CostGrid grid = unknown_band_grid();
  const auto start = snap_to_traversable(grid, 0.5, -4.0, 2.0);
  const auto goal = snap_to_traversable(grid, 14.0, 4.0, 3.0);
  ASSERT_TRUE(start.has_value());
  ASSERT_TRUE(goal.has_value());

  const auto result = astar(grid, *start, *goal, 8);
  ASSERT_FALSE(result.path.empty()) << "expected a route around the obstacle";
  EXPECT_TRUE(std::isfinite(result.cost));

  double crossing_min_x = kInf;
  int crossings = 0;
  for (const Cell & c : result.path) {
    EXPECT_EQ(grid.state[grid.idx(c.first, c.second)], FREE);
    EXPECT_TRUE(std::isfinite(grid.cost[grid.idx(c.first, c.second)]));
    const auto world = grid.cell_to_world(c.first, c.second);
    if (std::fabs(world.second) < 0.5) {
      ++crossings;
      crossing_min_x = std::min(crossing_min_x, world.first);
    }
  }
  // It had to detour to the observed crossing rather than cut straight across
  // the unobserved band, which spans x < 11 at |y| < 0.5.
  EXPECT_GT(crossings, 0);
  EXPECT_GT(crossing_min_x, 10.5);
}

TEST(AStar, PrefersTheCheapCentreOverTheWallHuggingShortcut)
{
  const CostGrid grid = corridor_grid(3.0, 1.0);
  const auto start = snap_to_traversable(grid, 1.0, 3.0, 1.0);
  const auto goal = snap_to_traversable(grid, 19.0, 3.0, 1.0);
  ASSERT_TRUE(start.has_value());
  ASSERT_TRUE(goal.has_value());

  const auto result = astar(grid, *start, *goal, 8);
  ASSERT_FALSE(result.path.empty());
  EXPECT_TRUE(std::isfinite(result.cost));

  // Reference: the geometrically shortest route, straight along y = 3.
  std::vector<Cell> straight;
  for (int ix = std::min(start->first, goal->first); ix <= std::max(start->first, goal->first);
    ++ix)
  {
    straight.push_back({ix, start->second});
    ASSERT_TRUE(grid.is_traversable(ix, start->second));
  }

  EXPECT_GT(path_length_m(grid, result.path), path_length_m(grid, straight));
  EXPECT_LT(path_cost(grid, result.path), path_cost(grid, straight));
}

TEST(AStar, WalledOffGoalReturnsEmptyPathAndInfiniteCost)
{
  const CostGrid grid = sealed_grid();
  const auto start = snap_to_traversable(grid, 1.0, 0.0, 2.0);
  const auto goal = snap_to_traversable(grid, 18.0, 0.0, 2.0);
  ASSERT_TRUE(start.has_value());
  ASSERT_TRUE(goal.has_value());

  const auto result = astar(grid, *start, *goal, 8);
  EXPECT_TRUE(result.path.empty());
  EXPECT_EQ(result.cost, kInf);
}

TEST(AStar, StartEqualsGoalIsAZeroCostSingleton)
{
  const CostGrid grid = corridor_grid();
  const auto start = snap_to_traversable(grid, 1.0, 0.0, 2.0);
  ASSERT_TRUE(start.has_value());
  const auto result = astar(grid, *start, *start, 8);
  ASSERT_EQ(result.path.size(), 1u);
  EXPECT_EQ(result.path.front(), *start);
  EXPECT_DOUBLE_EQ(result.cost, 0.0);
}

TEST(AStar, HeuristicNeverExceedsTheTrueCost)
{
  const CostGrid grid = random_grid(11);
  std::vector<Cell> traversable;
  for (int ix = 0; ix < grid.nx; ++ix) {
    for (int iy = 0; iy < grid.ny; ++iy) {
      if (grid.is_traversable(ix, iy)) {
        traversable.push_back({ix, iy});
      }
    }
  }
  ASSERT_FALSE(traversable.empty());

  std::mt19937 rng(3);
  for (int connectivity : {8, 4}) {
    std::uniform_int_distribution<std::size_t> pick(0, traversable.size() - 1);
    for (int s = 0; s < 20; ++s) {
      const Cell start = traversable[pick(rng)];
      const DijkstraField field = dijkstra(grid, start, connectivity);
      std::vector<Cell> reachable;
      for (int ix = 0; ix < grid.nx; ++ix) {
        for (int iy = 0; iy < grid.ny; ++iy) {
          if (std::isfinite(field.cost_at(ix, iy))) {
            reachable.push_back({ix, iy});
          }
        }
      }
      ASSERT_FALSE(reachable.empty());
      std::uniform_int_distribution<std::size_t> pick_goal(0, reachable.size() - 1);
      for (int g = 0; g < 10; ++g) {
        const Cell goal = reachable[pick_goal(rng)];
        const double true_cost = static_cast<double>(field.cost_at(goal.first, goal.second));
        const double estimate = (connectivity == 8) ?
          octile(start, goal, grid.resolution) :
          manhattan(start, goal, grid.resolution);
        // The field is stored as float32, so a route whose true cost equals the
        // heuristic exactly can read back a few ulps low.
        EXPECT_LE(estimate, true_cost + std::max(1e-9, 1e-6 * true_cost));
      }
    }
  }
}

TEST(AStar, DoesNotDriveThroughADiagonalWall)
{
  const CostGrid grid = diagonal_wall_grid();
  const Cell start{1, 10};
  const Cell goal{10, 1};
  ASSERT_TRUE(grid.is_traversable(start.first, start.second));
  ASSERT_TRUE(grid.is_traversable(goal.first, goal.second));

  for (int connectivity : {4, 8}) {
    const auto result = astar(grid, start, goal, connectivity);
    ASSERT_FALSE(result.path.empty()) << "connectivity " << connectivity << " found no route";
    for (std::size_t i = 0; i + 1 < result.path.size(); ++i) {
      const int dx = result.path[i + 1].first - result.path[i].first;
      const int dy = result.path[i + 1].second - result.path[i].second;
      if (dx != 0 && dy != 0) {
        EXPECT_TRUE(grid.is_traversable(result.path[i].first + dx, result.path[i].second));
        EXPECT_TRUE(grid.is_traversable(result.path[i].first, result.path[i].second + dy));
      }
    }
    // Any legal route walks around the wall's near end, which costs strictly
    // more than the 13.58 the corner-cutting shortcut used to report.
    int min_ix = grid.nx;
    for (const Cell & c : result.path) {
      min_ix = std::min(min_ix, c.first);
    }
    EXPECT_LE(min_ix, 1);
    EXPECT_GT(result.cost, 15.0);
  }
}

TEST(Bfs, MatchesABruteForceFlood)
{
  const CostGrid grid = random_grid(5);
  const auto start = snap_to_traversable(grid, 0.5, 0.0, 2.0);
  ASSERT_TRUE(start.has_value());

  const std::vector<int32_t> hops = bfs(grid, *start, 4);
  const std::map<Cell, int> reference = brute_force_flood(grid, *start);

  const std::vector<Cell> visited = reachable_cells(hops, grid);
  EXPECT_EQ(visited.size(), reference.size());
  for (const auto & entry : reference) {
    EXPECT_EQ(hops[grid.idx(entry.first.first, entry.first.second)], entry.second);
  }
}

TEST(Dfs, VisitsExactlyTheCellsBfsDoesInADifferentOrder)
{
  const CostGrid grid = random_grid(13);
  const auto start = snap_to_traversable(grid, 0.5, 0.0, 2.0);
  ASSERT_TRUE(start.has_value());

  const std::vector<int32_t> hops = bfs(grid, *start, 4);
  const std::vector<int32_t> order = dfs(grid, *start, 4);

  EXPECT_EQ(reachable_cells(hops, grid), reachable_cells(order, grid));

  std::vector<int32_t> hop_values;
  std::vector<int32_t> order_values;
  for (std::size_t k = 0; k < hops.size(); ++k) {
    if (hops[k] >= 0) {hop_values.push_back(hops[k]);}
    if (order[k] >= 0) {order_values.push_back(order[k]);}
  }
  // Ordering that set differently is the only thing DFS adds.
  EXPECT_NE(hop_values, order_values);
}

TEST(Dfs, IsIterativeOnAGridThatWouldBlowTheRecursionLimit)
{
  const std::vector<Point> ground = sheet(0.0, 149.0, 0.0, 149.0, 0.25);
  const CostGrid grid = rasterise(ground, {});
  ASSERT_GE(std::min(grid.nx, grid.ny), 300);

  const auto start = snap_to_traversable(grid, 1.0, 1.0, 2.0);
  ASSERT_TRUE(start.has_value());
  const std::vector<int32_t> order = dfs(grid, *start, 4);

  std::size_t visited = 0;
  for (int32_t v : order) {
    if (v >= 0) {++visited;}
  }
  std::size_t traversable = 0;
  for (uint8_t m : grid.traversable_mask()) {
    traversable += m;
  }
  EXPECT_EQ(visited, traversable);
}

TEST(Searches, RejectAnUnsupportedConnectivity)
{
  const CostGrid grid = corridor_grid();
  const auto free_start = snap_to_traversable(grid, 1.0, 0.0, 2.0);
  ASSERT_TRUE(free_start.has_value());
  const auto blocked_cells = grid.cells_with_state(OCCUPIED);
  ASSERT_FALSE(blocked_cells.empty());
  const Cell blocked{blocked_cells.front()[0], blocked_cells.front()[1]};

  // Validation runs before the traversability check, so a blocked start still
  // throws rather than short-circuiting to an empty result.
  for (const Cell & start : {*free_start, blocked}) {
    EXPECT_THROW(astar(grid, start, start, 6), std::invalid_argument);
    EXPECT_THROW(dijkstra(grid, start, 6), std::invalid_argument);
    EXPECT_THROW(bfs(grid, start, 6), std::invalid_argument);
    EXPECT_THROW(dfs(grid, start, 6), std::invalid_argument);
  }
}

TEST(Searches, FromAnUntraversableStartAreEmptyNotAnException)
{
  const CostGrid grid = corridor_grid();
  const auto blocked_cells = grid.cells_with_state(OCCUPIED);
  ASSERT_FALSE(blocked_cells.empty());
  const Cell blocked{blocked_cells.front()[0], blocked_cells.front()[1]};

  const auto result = astar(grid, blocked, blocked, 8);
  EXPECT_TRUE(result.path.empty());
  EXPECT_EQ(result.cost, kInf);

  const DijkstraField field = dijkstra(grid, blocked);
  for (float c : field.cost) {
    EXPECT_FALSE(std::isfinite(c));
  }
  for (int32_t p : field.parent) {
    EXPECT_EQ(p, -1);
  }
  const std::vector<int32_t> hops = bfs(grid, blocked);
  const std::vector<int32_t> order = dfs(grid, blocked);
  EXPECT_EQ(*std::max_element(hops.begin(), hops.end()), -1);
  EXPECT_EQ(*std::max_element(order.begin(), order.end()), -1);
}

TEST(ReconstructPath, ReturnsEmptyForAnUnreachableGoal)
{
  const CostGrid grid = sealed_grid();
  const auto start = snap_to_traversable(grid, 1.0, 0.0, 2.0);
  const auto goal = snap_to_traversable(grid, 18.0, 0.0, 2.0);
  ASSERT_TRUE(start.has_value());
  ASSERT_TRUE(goal.has_value());

  const DijkstraField field = dijkstra(grid, *start);
  EXPECT_TRUE(reconstruct_path(field, *goal).empty());
  EXPECT_EQ(reconstruct_path(field, *start), std::vector<Cell>{*start});
  // Out of bounds is a missing goal, not an out-of-range read.
  EXPECT_TRUE(reconstruct_path(field, {-1, 0}).empty());
  EXPECT_TRUE(reconstruct_path(field, {grid.nx, 0}).empty());
}

TEST(PathTotals, AreZeroForShortPathsAndThrowOnAGap)
{
  const CostGrid grid = corridor_grid();
  EXPECT_DOUBLE_EQ(path_cost(grid, {}), 0.0);
  EXPECT_DOUBLE_EQ(path_length_m(grid, {}), 0.0);
  EXPECT_DOUBLE_EQ(path_cost(grid, {{4, 4}}), 0.0);
  EXPECT_DOUBLE_EQ(path_length_m(grid, {{4, 4}}), 0.0);

  const std::vector<Cell> gapped{{4, 4}, {9, 4}};
  EXPECT_THROW(path_cost(grid, gapped), std::invalid_argument);
  EXPECT_THROW(path_length_m(grid, gapped), std::invalid_argument);
}

TEST(PathLength, IsPureGeometry)
{
  const CostGrid grid = corridor_grid();
  const std::vector<Cell> path{{4, 4}, {5, 4}, {6, 5}};
  EXPECT_DOUBLE_EQ(path_length_m(grid, path), kResolution * (1.0 + std::sqrt(2.0)));
}

TEST(PullOntoGrid, ReturnsNullOptOnMapAndLandsInsideTheMapOtherwise)
{
  const CostGrid grid = corridor_grid();
  EXPECT_FALSE(pull_onto_grid(grid, 5.0, 0.0, 1.0, 0.0).has_value());
  EXPECT_FALSE(pull_onto_grid(grid, 200.0, 0.0, 0.0, 0.0).has_value());

  const auto landing = pull_onto_grid(grid, 200.0, 0.0, 1.0, 0.0);
  ASSERT_TRUE(landing.has_value());
  const double hi_x = grid.origin_x + grid.nx * grid.resolution;
  EXPECT_LT(landing->first, hi_x);
  EXPECT_GT(landing->first, hi_x - grid.resolution);
  EXPECT_DOUBLE_EQ(landing->second, 0.0);
}

TEST(SelectGoal, ReturnsNullOptWhenBoxedIn)
{
  const std::vector<Point> patch = sheet(0.0, 10.0, -3.0, 3.0);
  const CostGrid grid = rasterise(patch, patch, kResolution, 0.6, 1.5, 3.0, 1);
  for (uint8_t m : grid.traversable_mask()) {
    ASSERT_EQ(m, 0u);
  }
  EXPECT_FALSE(select_goal(grid, 5.0, 0.0, 5.0).has_value());
}

TEST(SelectGoal, PullsAnOffMapRequestBackAlongTheAxis)
{
  const CostGrid grid = corridor_grid();
  const auto cell = select_goal(grid, 200.0, 0.0, 2.0, 1.0, 0.0);
  ASSERT_TRUE(cell.has_value());
  EXPECT_GT(grid.cell_to_world(cell->first, cell->second).first, 18.0);
}

TEST(SelectGoal, PrefersTheDirectSnapWhenTheRequestIsOnMap)
{
  const CostGrid grid = corridor_grid();
  const auto cell = select_goal(grid, 5.0, 0.0, 2.0, 1.0, 0.0);
  ASSERT_TRUE(cell.has_value());
  EXPECT_EQ(*cell, *snap_to_traversable(grid, 5.0, 0.0, 2.0));
}

TEST(FurthestReachable, OnlyOffersConnectedCells)
{
  const CostGrid grid = sealed_grid();
  const auto start = snap_to_traversable(grid, 1.0, 0.0, 2.0);
  ASSERT_TRUE(start.has_value());
  const DijkstraField field = dijkstra(grid, *start);

  const auto cell = furthest_reachable(field, grid, 1.0, 0.0);
  ASSERT_TRUE(cell.has_value());
  EXPECT_TRUE(std::isfinite(field.cost_at(cell->first, cell->second)));
  EXPECT_LT(grid.cell_to_world(cell->first, cell->second).first, 9.5);

  EXPECT_THROW(furthest_reachable(field, grid, 0.0, 0.0), std::invalid_argument);
}

TEST(FurthestReachable, BreaksTiesToTheSmallestCell)
{
  const CostGrid grid = corridor_grid();
  const auto start = snap_to_traversable(grid, 1.0, 0.0, 2.0);
  ASSERT_TRUE(start.has_value());
  const DijkstraField field = dijkstra(grid, *start);

  const auto cell = furthest_reachable(field, grid, 1.0, 0.0);
  ASSERT_TRUE(cell.has_value());
  // Every reachable cell in the winning column projects identically onto +x, so
  // the answer must be the lowest iy in the furthest reachable column, and it
  // must repeat exactly frame after frame.
  int expected_iy = grid.ny;
  for (int iy = 0; iy < grid.ny; ++iy) {
    if (std::isfinite(field.cost_at(cell->first, iy))) {
      expected_iy = std::min(expected_iy, iy);
    }
  }
  EXPECT_EQ(cell->second, expected_iy);
  EXPECT_EQ(furthest_reachable(field, grid, 2.0, 0.0), cell);
}

TEST(FurthestReachable, IsNullOptWhenNothingIsReachable)
{
  const CostGrid grid = corridor_grid();
  const auto blocked_cells = grid.cells_with_state(OCCUPIED);
  ASSERT_FALSE(blocked_cells.empty());
  const DijkstraField field =
    dijkstra(grid, Cell{blocked_cells.front()[0], blocked_cells.front()[1]});
  EXPECT_FALSE(furthest_reachable(field, grid, 1.0, 0.0).has_value());
}

TEST(NormalisedBands, CoversTheRangeAndMarksInvalidCells)
{
  const std::vector<double> values{0.0, 1.0, kInf, 3.0};
  const std::vector<uint8_t> valid{1, 1, 1, 1};
  const std::vector<int32_t> index = normalised_bands(values, valid, 4);

  EXPECT_EQ(index[2], -1);  // valid but infinite
  EXPECT_EQ(index[0], 0);
  EXPECT_EQ(index[3], 3);   // the clamp keeps the maximum out of band 4
  for (int32_t b : index) {
    EXPECT_LE(b, 3);
    EXPECT_GE(b, -1);
  }

  const std::vector<int32_t> flat = normalised_bands({5.0, 5.0, 5.0, 5.0}, valid, 4);
  for (int32_t b : flat) {
    EXPECT_EQ(b, 0);
  }

  const std::vector<int32_t> none = normalised_bands(values, {0, 0, 0, 0}, 4);
  for (int32_t b : none) {
    EXPECT_EQ(b, -1);
  }
  EXPECT_THROW(normalised_bands(values, {1, 1}, 4), std::invalid_argument);
}

TEST(BandColor, RampsBlueToRedAndSurvivesASingleBand)
{
  const auto low = band_color(0, 4);
  EXPECT_DOUBLE_EQ(low[0], 0.0);
  EXPECT_DOUBLE_EQ(low[2], 1.0);
  EXPECT_DOUBLE_EQ(low[3], 0.7);

  const auto high = band_color(3, 4);
  EXPECT_DOUBLE_EQ(high[0], 1.0);
  EXPECT_DOUBLE_EQ(high[2], 0.0);

  const auto mid = band_color(1, 3, 0.25);
  EXPECT_DOUBLE_EQ(mid[0], 0.5);
  EXPECT_DOUBLE_EQ(mid[1], 0.55);
  EXPECT_DOUBLE_EQ(mid[3], 0.25);

  // bands == 1 must not divide by zero, and out-of-range indices clamp.
  EXPECT_DOUBLE_EQ(band_color(0, 1)[0], 0.0);
  EXPECT_DOUBLE_EQ(band_color(9, 1)[0], 1.0);
  EXPECT_DOUBLE_EQ(band_color(-3, 4)[0], 0.0);
}
