#include "ransac_seg_cpp/cost_grid.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "ransac_seg_cpp/numeric.hpp"

namespace ransac_seg_cpp
{

const std::array<Step, 4> kSteps4{{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};

const std::array<Step, 8> kSteps8{
  {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}}};

CostGrid::CostGrid(
  double origin_x_in, double origin_y_in, double resolution_in, int nx_in, int ny_in,
  OptPlane plane_in)
: origin_x(origin_x_in),
  origin_y(origin_y_in),
  resolution(resolution_in),
  nx(nx_in),
  ny(ny_in),
  state(static_cast<std::size_t>(nx_in < 0 ? 0 : nx_in) *
    static_cast<std::size_t>(ny_in < 0 ? 0 : ny_in), static_cast<int8_t>(UNKNOWN)),
  cost(state.size(), std::numeric_limits<float>::infinity()),
  plane(std::move(plane_in))
{
}

bool CostGrid::in_bounds(int ix, int iy) const
{
  return ix >= 0 && ix < nx && iy >= 0 && iy < ny;
}

bool CostGrid::is_traversable(int ix, int iy) const
{
  if (!in_bounds(ix, iy)) {
    return false;
  }
  const std::size_t k = idx(ix, iy);
  return state[k] == FREE && std::isfinite(cost[k]);
}

std::vector<uint8_t> CostGrid::traversable_mask() const
{
  std::vector<uint8_t> mask(state.size(), 0);
  for (std::size_t k = 0; k < state.size(); ++k) {
    mask[k] = (state[k] == FREE && std::isfinite(cost[k])) ? 1u : 0u;
  }
  return mask;
}

std::vector<std::array<int32_t, 2>> CostGrid::cells_with_state(int8_t value) const
{
  std::vector<std::array<int32_t, 2>> cells;
  // ix outer, iy inner: lexicographically ascending, which is np.argwhere's
  // order and what the snap tie-break assumes.
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      if (state[idx(ix, iy)] == value) {
        cells.push_back({ix, iy});
      }
    }
  }
  return cells;
}

namespace
{

/// floor((v - origin) / resolution) as a 32-bit cell index, saturating.
///
/// floor on the double *then* cast. Origins are routinely negative
/// (origin_x = xmin - resolution), and truncation toward zero would fold the
/// whole strip [-resolution, 0) into cell 0.
///
/// The saturation is the difference from the Python, whose scalar world_to_cell
/// returns an arbitrary-precision int: here a 1e12 m goal used to be an
/// out-of-range double-to-int conversion, which is undefined and produced
/// INT_MIN, which then signed-overflowed snap_to_traversable's window. INT_MIN
/// and INT_MAX are just as far outside any real grid as 2e12 is, so every
/// bounds test answers the same and the arithmetic is defined.
int32_t cell_index(double v, double origin, double resolution)
{
  const int64_t k = floor_index((v - origin) / resolution);
  if (k > std::numeric_limits<int32_t>::max()) {return std::numeric_limits<int32_t>::max();}
  if (k < std::numeric_limits<int32_t>::min()) {return std::numeric_limits<int32_t>::min();}
  return static_cast<int32_t>(k);
}

}  // namespace

std::pair<int, int> CostGrid::world_to_cell(double x, double y) const
{
  return {cell_index(x, origin_x, resolution), cell_index(y, origin_y, resolution)};
}

void CostGrid::world_to_cell(
  const double * x, const double * y, std::size_t n, int32_t * out_ix, int32_t * out_iy) const
{
  for (std::size_t i = 0; i < n; ++i) {
    out_ix[i] = cell_index(x[i], origin_x, resolution);
    out_iy[i] = cell_index(y[i], origin_y, resolution);
  }
}

std::pair<double, double> CostGrid::cell_to_world(int ix, int iy) const
{
  // Cell centre: origin is the lower-left corner of cell (0, 0). Cast to
  // double first so out-of-range indices extrapolate linearly.
  return {
    origin_x + (static_cast<double>(ix) + 0.5) * resolution,
    origin_y + (static_cast<double>(iy) + 0.5) * resolution};
}

std::vector<std::array<double, 3>> CostGrid::cell_to_world_xyz(
  const std::vector<std::array<int64_t, 2>> & cells) const
{
  std::vector<std::array<double, 3>> out;
  out.reserve(cells.size());
  for (const auto & cell : cells) {
    const double x = origin_x + (static_cast<double>(cell[0]) + 0.5) * resolution;
    const double y = origin_y + (static_cast<double>(cell[1]) + 0.5) * resolution;
    out.push_back({x, y, plane_z(plane, x, y)});
  }
  return out;
}

double plane_z(const OptPlane & plane, double x, double y)
{
  if (!plane) {
    return 0.0;
  }
  const double nz = plane->normal.z();
  if (std::fabs(nz) < 1e-6) {
    return 0.0;
  }
  return -(plane->normal.x() * x + plane->normal.y() * y + plane->offset) / nz;
}

void plane_z(
  const OptPlane & plane, const double * x, const double * y, std::size_t n, double * out)
{
  if (!plane || std::fabs(plane->normal.z()) < 1e-6) {
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = 0.0;
    }
    return;
  }
  const double a = plane->normal.x();
  const double b = plane->normal.y();
  const double c = plane->offset;
  const double nz = plane->normal.z();
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = -(a * x[i] + b * y[i] + c) / nz;
  }
}

void neighbors(const CostGrid & grid, int ix, int iy, int connectivity, std::vector<Neighbor> & out)
{
  const Step * steps = nullptr;
  std::size_t count = 0;
  if (connectivity == 4) {
    steps = kSteps4.data();
    count = kSteps4.size();
  } else if (connectivity == 8) {
    steps = kSteps8.data();
    count = kSteps8.size();
  } else {
    throw std::invalid_argument(
            "connectivity must be 4 or 8, got " + std::to_string(connectivity));
  }

  out.clear();
  for (std::size_t s = 0; s < count; ++s) {
    const int dx = steps[s].dx;
    const int dy = steps[s].dy;
    const int jx = ix + dx;
    const int jy = iy + dy;
    if (!grid.is_traversable(jx, jy)) {
      continue;
    }
    if (dx != 0 && dy != 0) {
      // The vehicle has width; a corner it cannot fit through is not an edge.
      if (!grid.is_traversable(jx, iy) || !grid.is_traversable(ix, jy)) {
        continue;
      }
      out.push_back({jx, jy, grid.resolution * kSqrt2});
    } else {
      out.push_back({jx, jy, grid.resolution});
    }
  }
}

std::optional<std::pair<int, int>> snap_to_traversable(
  const CostGrid & grid, double x, double y, double radius)
{
  if (grid.nx == 0 || grid.ny == 0 || radius < 0.0) {
    return std::nullopt;
  }
  // DECLARED DEVIATION from grid.py, whose scalar world_to_cell raises
  // ValueError on a NaN coordinate. The goal comes straight off the
  // astar.goal_x / astar.goal_y parameters, so raising here would take the node
  // down from a subscription callback; "no traversable cell" is the answer the
  // caller already knows how to publish.
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(radius)) {
    return std::nullopt;
  }

  const auto centre = grid.world_to_cell(x, y);  // may sit outside the grid
  // int64 throughout: `centre` saturates to INT_MIN/INT_MAX for a far-field
  // goal and the +/- span around it would wrap in int, flipping the window's
  // sign and turning "far outside the grid" into "inside it".
  const int64_t span = floor_index(std::ceil(radius / grid.resolution)) + 1;
  const int64_t cx = centre.first;
  const int64_t cy = centre.second;
  const int lo_x = static_cast<int>(std::max<int64_t>(0, cx - span));
  const int hi_x = static_cast<int>(std::min<int64_t>(grid.nx, cx + span + 1));
  const int lo_y = static_cast<int>(std::max<int64_t>(0, cy - span));
  const int hi_y = static_cast<int>(std::min<int64_t>(grid.ny, cy + span + 1));
  if (lo_x >= hi_x || lo_y >= hi_y) {
    return std::nullopt;
  }

  bool found = false;
  double best_dist = 0.0;
  int best_ix = 0;
  int best_iy = 0;
  // ix outer, iy inner and a strict improvement test: ties keep the first
  // cell seen, which is the lexicographic minimum (dist, ix, iy).
  for (int ix = lo_x; ix < hi_x; ++ix) {
    for (int iy = lo_y; iy < hi_y; ++iy) {
      if (!grid.is_traversable(ix, iy)) {
        continue;
      }
      const auto world = grid.cell_to_world(ix, iy);
      const double dist = std::hypot(world.first - x, world.second - y);
      if (dist > radius) {  // inclusive bound: dist == radius is accepted
        continue;
      }
      if (!found || dist < best_dist) {
        found = true;
        best_dist = dist;
        best_ix = ix;
        best_iy = iy;
      }
    }
  }
  if (!found) {
    return std::nullopt;
  }
  return std::make_pair(best_ix, best_iy);
}

}  // namespace ransac_seg_cpp
