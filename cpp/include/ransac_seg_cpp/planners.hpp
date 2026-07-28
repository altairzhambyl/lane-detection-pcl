// Graph searches over the tri-state cost grid, as pure functions.
//
// C++ port of ransac_seg/planners.py. The four planner nodes each used to carry
// a private copy of a search over a bare set of cells that happened to hold a
// ground return. That representation cannot express "occupied" or "unobserved"
// — both are simply absent — so an obstacle was only visible through the hole
// its LiDAR shadow left in the ground returns, and every step cost exactly one,
// which meant no route ever preferred the middle of a corridor over scraping
// past a wall.
//
// Everything here reads a CostGrid instead and shares one cost model:
//
//     edge(a, b) = step_length(a, b) * 0.5 * (cost[a] + cost[b])
//
// with step_length exactly as `neighbors` reports it. Both the nodes and the
// documentation quote that formula, so it lives in exactly one function and
// every caller goes through it.
//
// The two field-quantising helpers at the bottom live here rather than in
// markers for the same reason: they are pure array maths that three nodes need
// in order to colour a search result, and they must stay testable without ROS.
//
// No ROS header is included here, so the planner tests build on a bare checkout
// without a sourced ROS installation.

#ifndef RANSAC_SEG_CPP__PLANNERS_HPP_
#define RANSAC_SEG_CPP__PLANNERS_HPP_

#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ransac_seg_cpp/cost_grid.hpp"

namespace ransac_seg_cpp
{

/// One grid cell as (ix, iy). std::pair so it orders lexicographically, which
/// is what every tie-break in this file relies on.
using Cell = std::pair<int, int>;

/// Octile distance in METRES between two cells; the tight heuristic for 8.
double octile(Cell a, Cell b, double resolution);

/// 4-connected grid distance in METRES between two cells; the heuristic for 4.
double manhattan(Cell a, Cell b, double resolution);

/// Throw std::invalid_argument unless the connectivity is one `neighbors` walks.
void validate_connectivity(int connectivity);

/// Metric length of one step between two adjacent cells.
///
/// Throws std::invalid_argument when the cells are more than one apart or are
/// the same cell: a self-loop is not a step, and an "invalid path" is an error
/// rather than a segment silently worth zero.
double step_length(const CostGrid & grid, Cell a, Cell b);

/// The single shared cost model. Computed in double even though grid.cost is
/// float32, so a path total here matches the double accumulator inside the
/// searches bit for bit.
double edge_cost(const CostGrid & grid, Cell a, Cell b, double step);

/// A path from start to goal inclusive plus its total cost; ({}, inf) on failure.
struct PathResult
{
  std::vector<Cell> path;
  double cost;
};

/// Minimise accumulated cell cost from start to goal, or return ({}, inf).
///
/// Admissibility: build_costmap guarantees every finite cell cost is >= 1.0, so
/// the cheapest possible edge is its own step length in metres. The cheapest
/// possible route to the goal therefore costs at least the minimum walkable
/// distance to it, which on an 8-connected grid is exactly octile(cell, goal)
/// and on a 4-connected grid exactly manhattan(...). Multiplying by the cost
/// floor of 1.0 leaves both heuristics admissible AND consistent, so the first
/// pop of a cell is final and the CLOSED set can discard every later pop of it.
///
/// Using a Euclidean heuristic on a 4-connected grid — what this package did
/// before — is still admissible but looser by up to sqrt(2), which is what made
/// A* degenerate towards Dijkstra on open ground.
PathResult astar(const CostGrid & grid, Cell start, Cell goal, int connectivity = 8);

/// Cost and parent fields over the whole grid from one source.
///
/// `cost` holds +inf for every cell the source cannot reach and `parent` holds
/// -1 there. The source is its own parent, which is what terminates
/// reconstruct_path and what distinguishes "this is the source" from "this is
/// unreachable".
struct DijkstraField
{
  int nx = 0;
  int ny = 0;
  std::vector<float> cost;      ///< nx*ny, x-major, +inf where unreachable
  std::vector<int32_t> parent;  ///< 2*nx*ny, x-major (px, py) pairs, -1 if none

  float cost_at(int ix, int iy) const;
  Cell parent_at(int ix, int iy) const;
};

/// Dijkstra from one source with lazy deletion (no decrease-key).
///
/// An untraversable start returns the untouched field — all-inf cost, all -1
/// parent — never an exception.
DijkstraField dijkstra(const CostGrid & grid, Cell start, int connectivity = 8);

/// Hop count to every reachable cell; -1 where unreachable. nx*ny, x-major.
///
/// Hops, not metres and not cost: this is the unweighted reachability baseline
/// the weighted Dijkstra field is compared against. Note the default
/// connectivity is 4, unlike astar and dijkstra.
std::vector<int32_t> bfs(const CostGrid & grid, Cell start, int connectivity = 4);

/// Depth-first discovery order per cell; -1 where unvisited. nx*ny, x-major.
///
/// The stack is explicit. A 200x200 costmap is 40,000 cells and a recursive
/// depth-first search would overflow the C stack long before covering it, which
/// is why this never recurses.
std::vector<int32_t> dfs(const CostGrid & grid, Cell start, int connectivity = 4);

/// Cell path from a Dijkstra parent field, or {} when the goal is unreachable.
///
/// The walk terminates on the source, which is stored as its own parent. Never
/// returns a partial path.
std::vector<Cell> reconstruct_path(const DijkstraField & field, Cell goal);

/// Summed edge cost of a cell path; 0.0 for a path of one or none.
double path_cost(const CostGrid & grid, const std::vector<Cell> & path);

/// Geometric length of a cell path in metres; cost-blind.
double path_length_m(const CostGrid & grid, const std::vector<Cell> & path);

/// First point on the map when walking back along -axis, or nullopt.
///
/// Returns nullopt when the request already lies on the map — a snap failure
/// there is a real "nothing traversable here", not an off-map goal — and when
/// the backward ray never enters the map at all.
std::optional<std::pair<double, double>> pull_onto_grid(
  const CostGrid & grid, double x, double y, double axis_x, double axis_y);

/// Snap a requested goal onto a traversable cell, or fall back along an axis.
///
/// Geometry only: the returned cell is traversable but may sit in a different
/// connected component than the planner's start, which is the caller's problem
/// to notice and resolve with furthest_reachable.
///
/// A goal parameter tuned for a long bag routinely lands past the edge of a
/// single frame's map, so when the request is off-map it is pulled back along
/// the fallback axis to the map boundary before snapping. An on-map request
/// with nothing traversable within search_radius returns nullopt.
std::optional<Cell> select_goal(
  const CostGrid & grid, double x, double y, double search_radius = 5.0,
  double axis_x = 1.0, double axis_y = 0.0);

/// Reachable cell furthest along `axis`, or nullopt when nothing is reachable.
///
/// Reading the reachable set off a Dijkstra cost field rather than off the raw
/// traversability mask is what makes this fallback goal safe: every cell with a
/// finite cost is provably connected to the source, so the planner cannot pick
/// a goal on the far side of a wall and then fail every frame. Ties resolve to
/// the smaller (ix, iy) so a static scene keeps producing the same goal frame
/// after frame instead of flickering between two cells.
///
/// Throws std::invalid_argument when the axis has no length.
std::optional<Cell> furthest_reachable(
  const DijkstraField & field, const CostGrid & grid, double axis_x, double axis_y);

/// Quantise a scalar field into `bands` equal-width levels over `valid`.
///
/// Returns a band index per valid finite cell and -1 elsewhere; an infinite
/// value inside `valid` still comes back -1. Publishing one uniformly coloured
/// CUBE_LIST per band keeps the marker path vectorised; a per-cell colour would
/// mean one ColorRGBA per cell, and a 200x200 costmap is 40,000 of them every
/// frame.
///
/// Throws std::invalid_argument when `valid` is not the same length as `values`.
std::vector<int32_t> normalised_bands(
  const std::vector<double> & values, const std::vector<uint8_t> & valid, int bands);

/// (r, g, b, a) blue-to-red ramp colour for a band index.
std::array<double, 4> band_color(int band, int bands, double alpha = 0.7);

}  // namespace ransac_seg_cpp

#endif  // RANSAC_SEG_CPP__PLANNERS_HPP_
