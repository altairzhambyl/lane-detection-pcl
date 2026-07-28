// Exact Euclidean distance transform — the C++ stand-in for
// scipy.ndimage.distance_transform_edt(sampling=resolution) as grid.py calls it.
//
// It has to be the EXACT transform, not a chamfer or Manhattan approximation.
// The costmap subtracts half a cell diagonal from the result and then compares
// it against the robot radius, so an approximation of a few percent moves cells
// across the FREE/INFLATED boundary and changes every decayed cost value; the
// Python pinning tests bound the inflated disc area to a 0.75x-1.35x band and
// assert clearance >= robot_radius as a hard inequality.
//
// Felzenszwalb & Huttenlocher's lower-envelope algorithm, O(nx*ny), run in
// SQUARED distance space so the per-row work is exact integer arithmetic; the
// sqrt and the metric scaling happen once at the end.

#ifndef RANSAC_SEG_CPP__EDT_HPP_
#define RANSAC_SEG_CPP__EDT_HPP_

#include <cstdint>

namespace ransac_seg_cpp
{

/// Distance in METRES from each cell centre to the nearest occupied cell centre.
///
/// `occupied` is an nx*ny x-major mask (`occupied[ix * ny + iy]`, non-zero means
/// occupied), matching CostGrid's layout. Occupied cells get 0.0. When nothing
/// is occupied every cell gets +infinity — the transform of an empty mask is
/// meaningless, and the costmap short-circuits that case anyway.
///
/// `out` must have room for nx*ny doubles.
void distance_transform_edt(
  const uint8_t * occupied, int nx, int ny, double resolution, double * out);

}  // namespace ransac_seg_cpp

#endif  // RANSAC_SEG_CPP__EDT_HPP_
