#include "ransac_seg_cpp/costmap.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "ransac_seg_cpp/edt.hpp"

namespace ransac_seg_cpp
{

namespace
{

constexpr double kInf = std::numeric_limits<double>::infinity();

/// Cell count of a padded span, clamped so a nonsense extent (bad units, a
/// return at 10^9 m) cannot overflow the int the grid stores. Python's ints are
/// arbitrary precision, so it never needed the guard; either way the value only
/// has to be large enough to trip the kMaxCells coarsening path below.
int span_cells(double lo_pad, double hi_pad, double resolution)
{
  const double raw = std::ceil((hi_pad - lo_pad) / resolution);
  if (!(raw > 1.0)) {
    return 1;  // also catches NaN
  }
  const double capped = static_cast<double>(std::numeric_limits<int>::max());
  return raw >= capped ? std::numeric_limits<int>::max() : static_cast<int>(raw);
}

/// Per-cell hit counts of the in-bounds quantised points.
///
/// Out-of-bounds points are DROPPED, never clamped or wrapped: a return outside
/// explicit bounds must not pile onto the boundary cell and free it.
void accumulate(
  const std::vector<std::array<double, 2>> & xy, const Dims & d, double resolution,
  std::vector<int32_t> & counts)
{
  for (const auto & p : xy) {
    const int64_t ix = static_cast<int64_t>(std::floor((p[0] - d.origin_x) / resolution));
    const int64_t iy = static_cast<int64_t>(std::floor((p[1] - d.origin_y) / resolution));
    if (ix < 0 || ix >= d.nx || iy < 0 || iy >= d.ny) {
      continue;
    }
    ++counts[static_cast<std::size_t>(ix) * d.ny + static_cast<std::size_t>(iy)];
  }
}

}  // namespace

std::vector<std::array<double, 2>> filter_xy(const float * pts, std::size_t n, int stride)
{
  std::vector<std::array<double, 2>> out;
  if (pts == nullptr || n == 0 || stride < 2) {
    return out;
  }
  out.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double x = static_cast<double>(pts[i * static_cast<std::size_t>(stride)]);
    const double y = static_cast<double>(pts[i * static_cast<std::size_t>(stride) + 1]);
    if (std::isfinite(x) && std::isfinite(y)) {
      out.push_back({x, y});
    }
  }
  return out;
}

std::optional<std::array<double, 4>> extent(
  const std::vector<std::array<double, 2>> & a, const std::vector<std::array<double, 2>> & b)
{
  if (a.empty() && b.empty()) {
    return std::nullopt;
  }
  double xmin = kInf, xmax = -kInf, ymin = kInf, ymax = -kInf;
  for (const auto * cloud : {&a, &b}) {
    for (const auto & p : *cloud) {
      xmin = std::min(xmin, p[0]);
      xmax = std::max(xmax, p[0]);
      ymin = std::min(ymin, p[1]);
      ymax = std::max(ymax, p[1]);
    }
  }
  return std::array<double, 4>{xmin, xmax, ymin, ymax};
}

Dims dims(const std::array<double, 4> & bounds, double resolution)
{
  const double xmin = bounds[0], xmax = bounds[1], ymin = bounds[2], ymax = bounds[3];
  Dims d;
  d.origin_x = xmin - resolution;
  d.origin_y = ymin - resolution;
  // The associativity is copied from grid.py — ((xmax + resolution) - origin_x)
  // — so the float rounding of a cap-boundary map matches cell for cell.
  d.nx = span_cells(d.origin_x, xmax + resolution, resolution);
  d.ny = span_cells(d.origin_y, ymax + resolution, resolution);
  return d;
}

void carve_ray_free(CostGrid & grid, double sensor_x, double sensor_y, int bins, int min_cells)
{
  const std::size_t cells = static_cast<std::size_t>(grid.nx) * static_cast<std::size_t>(grid.ny);
  bool any_free = false;
  for (std::size_t i = 0; i < cells; ++i) {
    if (grid.state[i] == FREE) {
      any_free = true;
      break;
    }
  }
  if (!any_free) {
    return;
  }

  const int nbins = std::max(bins, 1);
  std::vector<double> reach(static_cast<std::size_t>(nbins), 0.0);
  std::vector<int64_t> votes(static_cast<std::size_t>(nbins), 0);
  std::vector<double> radius(cells);
  std::vector<int32_t> bin_index(cells);

  const double res = grid.resolution;
  for (int ix = 0; ix < grid.nx; ++ix) {
    const double dx = grid.origin_x + (ix + 0.5) * res - sensor_x;
    for (int iy = 0; iy < grid.ny; ++iy) {
      const double dy = grid.origin_y + (iy + 0.5) * res - sensor_y;
      const std::size_t i = grid.idx(ix, iy);
      const double r = std::hypot(dx, dy);
      const double azimuth = std::atan2(dy, dx);
      // The argument is >= 0, so the truncation is a floor; the min folds an
      // azimuth of exactly +pi into the last bin instead of one past the end.
      const int64_t b = std::min(
        static_cast<int64_t>((azimuth + M_PI) / (2.0 * M_PI) * nbins),
        static_cast<int64_t>(nbins) - 1);
      radius[i] = r;
      bin_index[i] = static_cast<int32_t>(b);
      if (grid.state[i] == FREE) {
        reach[static_cast<std::size_t>(b)] = std::max(reach[static_cast<std::size_t>(b)], r);
        ++votes[static_cast<std::size_t>(b)];
      }
    }
  }

  // A single stray return must not carve a whole spoke across the map.
  const int64_t quorum = std::max(min_cells, 1);
  for (std::size_t b = 0; b < reach.size(); ++b) {
    if (votes[b] < quorum) {
      reach[b] = 0.0;
    }
  }

  for (std::size_t i = 0; i < cells; ++i) {
    if (grid.state[i] == UNKNOWN &&
      radius[i] <= reach[static_cast<std::size_t>(bin_index[i])])
    {
      grid.state[i] = FREE;
    }
  }
}

CostGrid build_costmap(
  const float * ground_xyz, std::size_t n_ground, int ground_stride,
  const float * obstacle_xyz, std::size_t n_obstacle, int obstacle_stride,
  const CostmapOptions & opts)
{
  const std::vector<std::array<double, 2>> ground =
    filter_xy(ground_xyz, n_ground, ground_stride);
  const std::vector<std::array<double, 2>> obstacles =
    filter_xy(obstacle_xyz, n_obstacle, obstacle_stride);

  const std::optional<std::array<double, 4>> box =
    opts.bounds.has_value() ? opts.bounds : extent(ground, obstacles);

  if (!box.has_value()) {
    // A blank frame yields a 1x1 UNKNOWN grid at the REQUESTED resolution
    // rather than an exception; the callback still has something to publish a
    // clear marker against.
    return CostGrid(0.0, 0.0, opts.resolution, 1, 1, opts.plane);
  }

  double resolution = opts.resolution;
  Dims d = dims(*box, resolution);
  if (static_cast<int64_t>(d.nx) * d.ny > kMaxCells) {
    // A single stray far-field return must not size the whole map; fall back
    // to the extent of the surface we actually intend to drive on.
    const std::optional<std::array<double, 4>> ground_box = extent(ground, {});
    const std::array<double, 4> fallback = ground_box.has_value() ? *ground_box : *box;
    d = dims(fallback, resolution);
    // Coarsen, never crop. Rescaling nx/ny at a fixed resolution used to shrink
    // the map onto a corner of the observed ground and drop every return
    // outside it, so the planner reported "no route" beyond that corner with
    // nothing logged. A bigger cell keeps the whole surface on the map, and the
    // caller reads the honoured spacing back off grid.resolution. _dims rounds
    // up, so one rescale can land marginally over the cap: iterate, and keep
    // the 1e-3 nudge, without which the loop can fail to make progress.
    while (static_cast<int64_t>(d.nx) * d.ny > kMaxCells) {
      resolution *= std::sqrt(static_cast<double>(d.nx) * d.ny /
          static_cast<double>(kMaxCells)) + 1e-3;
      d = dims(fallback, resolution);
    }
  }

  CostGrid grid(d.origin_x, d.origin_y, resolution, d.nx, d.ny, opts.plane);
  const std::size_t cells = static_cast<std::size_t>(d.nx) * static_cast<std::size_t>(d.ny);

  std::vector<int32_t> ground_counts(cells, 0);
  std::vector<int32_t> obstacle_counts(cells, 0);
  accumulate(ground, d, resolution, ground_counts);
  accumulate(obstacles, d, resolution, obstacle_counts);

  const int32_t ground_threshold = static_cast<int32_t>(std::max(1, opts.min_ground_points));
  const int32_t obstacle_threshold = static_cast<int32_t>(std::max(1, opts.min_obstacle_points));

  for (std::size_t i = 0; i < cells; ++i) {
    if (ground_counts[i] >= ground_threshold) {
      grid.state[i] = FREE;
    }
  }

  if (opts.ray_fill) {
    // Close the gaps between beam rings BEFORE obstacles are stamped, so a
    // carved cell can still be overridden by a return that lands in it.
    carve_ray_free(grid, opts.sensor_x, opts.sensor_y, opts.ray_fill_bins,
      opts.ray_fill_min_cells);
  }

  // Obstacles overwrite ground: a pole planted on inlier road still blocks.
  std::vector<uint8_t> occupied(cells, 0);
  bool any_occupied = false;
  for (std::size_t i = 0; i < cells; ++i) {
    if (obstacle_counts[i] >= obstacle_threshold) {
      occupied[i] = 1;
      any_occupied = true;
      grid.state[i] = OCCUPIED;
    }
  }

  std::vector<double> clearance(cells, kInf);
  if (any_occupied) {
    distance_transform_edt(occupied.data(), d.nx, d.ny, resolution, clearance.data());
    // The transform measures centre to centre, but the return that made a cell
    // OCCUPIED sits anywhere inside it, up to half a cell diagonal from that
    // centre. Taking the raw distance as clearance overstates it by that much,
    // which at the default 0.5 m resolution let a cell 0.36 m from a real
    // obstacle pass as clear of a 0.6 m robot radius. Subtract the worst case
    // so the reported clearance is one the geometry actually guarantees; the
    // price is up to half a cell of over-inflation.
    const double shave = 0.5 * kSqrt2 * resolution;
    for (std::size_t i = 0; i < cells; ++i) {
      clearance[i] = std::max(clearance[i] - shave, 0.0);
    }
  }
  // With nothing occupied every cell is infinitely clear; the transform of an
  // all-true mask is meaningless, so the +inf fill above stands.

  // Only FREE cells are promoted: UNKNOWN next to an obstacle stays UNKNOWN,
  // and OCCUPIED is never downgraded. The bound is inclusive.
  for (std::size_t i = 0; i < cells; ++i) {
    if (grid.state[i] == FREE && clearance[i] <= opts.robot_radius) {
      grid.state[i] = INFLATED;
    }
  }

  // Costs decay from just under 2.0 at the clearance boundary towards 1.0, so
  // the planner prefers the middle of the corridor without ever dipping below
  // the admissible floor of 1.0. FREE is recomputed AFTER inflation, so no
  // blocked cell ever carries a finite cost.
  for (std::size_t i = 0; i < cells; ++i) {
    if (grid.state[i] != FREE) {
      continue;
    }
    grid.cost[i] = 1.0f;
    if (clearance[i] < opts.inflation_radius) {
      grid.cost[i] = static_cast<float>(
        1.0 + std::exp(-opts.cost_scaling * (clearance[i] - opts.robot_radius)));
    }
  }

  return grid;
}

}  // namespace ransac_seg_cpp
