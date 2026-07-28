// Rasteriser half of ransac_seg/grid.py: ground and non-ground returns in,
// tri-state inflated CostGrid out.
//
// Keeping UNKNOWN distinct from FREE is the point of the whole type — a plain
// occupied-set map cannot tell "no data" from "road", and a planner reading one
// routed straight through the LiDAR shadow behind a wall. Obstacles override
// the ground they stand on, and every FREE cell is inflated away from obstacle
// RETURNS (not merely from obstacle cell centres) by the robot radius.
//
// Costs are the planner's contract: every finite cost is >= 1.0, so a metric
// step-length heuristic stays admissible for A*.

#ifndef RANSAC_SEG_CPP__COSTMAP_HPP_
#define RANSAC_SEG_CPP__COSTMAP_HPP_

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

#include "ransac_seg_cpp/cost_grid.hpp"

namespace ransac_seg_cpp
{

/// Keyword arguments of grid.build_costmap, same names and same defaults.
struct CostmapOptions
{
  /// Requested cell size. Read the HONOURED one back off `CostGrid::resolution`:
  /// it coarsens when the request would blow past kMaxCells.
  double resolution = 0.5;
  /// Explicit map extent as (xmin, xmax, ymin, ymax) — NOT (xmin, ymin, ...).
  /// The order matches `extent()` below and grid.py's `_extent`.
  std::optional<std::array<double, 4>> bounds;
  double robot_radius = 0.6;
  double inflation_radius = 1.5;
  double cost_scaling = 3.0;
  int min_ground_points = 1;
  int min_obstacle_points = 3;
  OptPlane plane;
  bool ray_fill = true;
  int ray_fill_bins = 720;
  int ray_fill_min_cells = 2;
  double sensor_x = 0.0;
  double sensor_y = 0.0;
};

/// Rasterise ground and non-ground returns into a tri-state cost grid.
///
/// `ground_xyz` / `obstacle_xyz` are `n * stride` float arrays; `stride` is the
/// number of components per point (3 for xyz, 4 for a KITTI sweep carrying
/// intensity) and only the first two are read. Either cloud may be empty; with
/// both empty the result is a 1x1 UNKNOWN grid at the REQUESTED resolution
/// rather than an exception, so a callback on a blank frame still has something
/// to publish a clear marker against.
CostGrid build_costmap(
  const float * ground_xyz, std::size_t n_ground, int ground_stride,
  const float * obstacle_xyz, std::size_t n_obstacle, int obstacle_stride,
  const CostmapOptions & opts);

// --- internals, exposed only so the tests can pin them ---------------------

/// grid.py's `_xy`: strided xy extraction, dropping a point only when its x OR
/// y is non-finite.
///
/// A non-finite Z does NOT drop the point. This deliberately differs from the
/// ground-plane filter, which requires all three columns finite: that one is
/// fitting a plane in 3D, this one only ever rasterises xy.
std::vector<std::array<double, 2>> filter_xy(const float * pts, std::size_t n, int stride);

/// grid.py's `_extent`: (xmin, xmax, ymin, ymax) over the non-empty clouds,
/// nullopt when both are empty. Note the axis-major order.
std::optional<std::array<double, 4>> extent(
  const std::vector<std::array<double, 2>> & a, const std::vector<std::array<double, 2>> & b);

/// Origin corner and cell counts of a padded bounding box.
struct Dims
{
  double origin_x;
  double origin_y;
  int nx;
  int ny;
};

/// grid.py's `_dims`: exactly one cell of padding below/left of the extent.
Dims dims(const std::array<double, 4> & bounds, double resolution);

/// Mark cells the sensor must have seen over as FREE, in place.
///
/// A spinning LiDAR samples in rings, so the spacing between consecutive ground
/// returns grows with range. Rasterising returns alone therefore leaves the map
/// striped with UNKNOWN arcs no planner can cross: measured on a real 77k-point
/// sweep, reachability died at 13.5 m of a 60 m map. Those arcs are not
/// unobserved — a beam that reached ground at 40 m along some azimuth proves
/// everything nearer along that azimuth was empty at ground height, or the beam
/// would have been blocked first. So per azimuth bin, take the farthest ground
/// return and carve everything closer to it FREE.
///
/// It cannot see negative obstacles: a trench between the sensor and a far
/// return is carved free like everything else. That is why this only ever ADDS
/// FREE (UNKNOWN -> FREE, never anything else) and why obstacle returns are
/// stamped afterwards, overriding it.
void carve_ray_free(CostGrid & grid, double sensor_x, double sensor_y, int bins, int min_cells);

}  // namespace ransac_seg_cpp

#endif  // RANSAC_SEG_CPP__COSTMAP_HPP_
