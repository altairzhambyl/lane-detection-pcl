// Tri-state cost grid — C++ port of the type half of ransac_seg/grid.py.
//
// Keeping UNKNOWN distinct from FREE is the whole point: a plain "occupied set"
// cannot tell "no data" from "road", and a planner reading one happily routed
// through the LiDAR shadow behind a wall. Every finite cost is >= 1.0 so a
// metric step-length heuristic stays admissible for A*.
//
// Storage is x-major (`state[size_t(ix) * ny + iy]`) to match numpy's default C
// order on an (nx, ny) array, so a cell's linear index is the same number in
// both implementations and the two can be diffed cell by cell.
//
// This header deliberately depends on nothing but Eigen and the STL — the
// planners, the clusterer and the lane fitter all compile against it.

#ifndef RANSAC_SEG_CPP__COST_GRID_HPP_
#define RANSAC_SEG_CPP__COST_GRID_HPP_

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace ransac_seg_cpp
{

/// Cell occupancy states; the values match ransac_seg/grid.py exactly.
enum CellState : int8_t
{
  UNKNOWN = -1,
  FREE = 0,
  OCCUPIED = 1,
  INFLATED = 2
};

/// A sweep that produces a few million cells is a bug (bad units, a stray
/// return at 10 km), not a map; rasterising it would stall the callback.
inline constexpr int64_t kMaxCells = 4000000;

/// Diagonal step length factor, hoisted so every planner uses one constant.
inline const double kSqrt2 = std::sqrt(2.0);

/// One integer cell offset.
struct Step
{
  int dx;
  int dy;
};

extern const std::array<Step, 4> kSteps4;
extern const std::array<Step, 8> kSteps8;

/// A fitted ground plane `normal . x + offset = 0`, as grid.py consumes it.
///
/// Deliberately not PlaneModel: the grid only needs the geometry, and taking
/// the smaller type keeps this header free of ground_plane.hpp.
struct Plane
{
  Eigen::Vector3d normal{0.0, 0.0, 1.0};
  double offset = 0.0;
};

using OptPlane = std::optional<Plane>;

/// A rasterised, inflated traversability map in the sensor's xy plane.
struct CostGrid
{
  double origin_x = 0.0;   ///< world x of the LOWER-LEFT corner of cell (0, 0)
  double origin_y = 0.0;   ///< world y of the LOWER-LEFT corner of cell (0, 0)
  double resolution = 0.5;
  int nx = 0;
  int ny = 0;
  std::vector<int8_t> state;  ///< nx*ny, x-major
  std::vector<float> cost;    ///< nx*ny, x-major; float32 like the Python array
  OptPlane plane;

  CostGrid() = default;

  /// Allocate an nx*ny grid, every cell UNKNOWN with +inf cost.
  CostGrid(
    double origin_x, double origin_y, double resolution, int nx, int ny, OptPlane plane);

  std::size_t idx(int ix, int iy) const {return static_cast<std::size_t>(ix) * ny + iy;}

  bool in_bounds(int ix, int iy) const;

  /// FREE *and* finitely costed. INFLATED/UNKNOWN/OCCUPIED all carry +inf, so
  /// this single predicate is what every planner asks.
  bool is_traversable(int ix, int iy) const;

  /// nx*ny x-major mask, 1 where is_traversable.
  std::vector<uint8_t> traversable_mask() const;

  /// Cells holding `value`, lexicographically ascending by (ix, iy).
  ///
  /// The order is np.argwhere's and the deterministic snap tie-break relies on
  /// it, so it is part of the contract, not an implementation detail.
  std::vector<std::array<int32_t, 2>> cells_with_state(int8_t value) const;

  /// Cell containing a world point. Unclamped: negative and >= nx results are
  /// legal and callers depend on the raw value.
  std::pair<int, int> world_to_cell(double x, double y) const;

  /// Vectorised world_to_cell over `n` points.
  void world_to_cell(
    const double * x, const double * y, std::size_t n, int32_t * out_ix, int32_t * out_iy) const;

  /// World xy of a cell CENTRE; extrapolates linearly for out-of-range indices.
  std::pair<double, double> cell_to_world(int ix, int iy) const;

  /// Cell centres lifted onto the stored plane. Empty input -> empty output.
  std::vector<std::array<double, 3>> cell_to_world_xyz(
    const std::vector<std::array<int64_t, 2>> & cells) const;
};

/// Solve the plane for z at an xy; 0.0 when absent or near-horizontal-normal.
///
/// The guard is on |normal.z| < 1e-6, not on a tilt angle: it is the divisor
/// that blows up, and degrading to zero beats projecting markers to infinity.
double plane_z(const OptPlane & plane, double x, double y);

/// Vectorised plane_z over `n` points.
void plane_z(const OptPlane & plane, const double * x, const double * y, std::size_t n, double * out);

/// A traversable neighbour and the metric length of the step reaching it.
struct Neighbor
{
  int ix;
  int iy;
  double step_m;
};

/// Traversable neighbours of (ix, iy), in kSteps order; `out` is cleared first.
///
/// A diagonal step is only offered when both cells it squeezes past are
/// traversable too. Without that check an 8-connected search walks through the
/// shared corner of two diagonally touching blocked cells; measured on the
/// Python side, that produced a 13.58 m path straight through a wall where the
/// legal route is ~30.9 m. The squeezed cells are tested for TRAVERSABILITY,
/// not merely bounds, so an UNKNOWN or off-grid squeeze also blocks.
///
/// Throws std::invalid_argument for any connectivity other than 4 or 8.
void neighbors(const CostGrid & grid, int ix, int iy, int connectivity, std::vector<Neighbor> & out);

/// Nearest traversable cell whose CENTRE is within `radius` metres, or nullopt.
///
/// Ties resolve to the smaller (ix, iy) so repeated calls on the same frame
/// give the planner an identical start cell.
std::optional<std::pair<int, int>> snap_to_traversable(
  const CostGrid & grid, double x, double y, double radius);

}  // namespace ransac_seg_cpp

#endif  // RANSAC_SEG_CPP__COST_GRID_HPP_
