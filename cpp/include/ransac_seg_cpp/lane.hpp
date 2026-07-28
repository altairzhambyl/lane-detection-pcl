// Lane boundary fitting in the frame of a fitted road centreline — C++ port of
// ransac_seg/lane.py.
//
// The method the Python replaced took min/max y per integer-x column and
// splined the result, and every part of that failed measurably: min/max let a
// single parked car drag the boundary 2.994 m out, a fixed x axis inflated a
// curved road to 8.06 m because the extents were measured along the sensor's x
// rather than across the road, and a hardcoded spline budget produced different
// geometry on the same road depending only on how many range bins the sweep
// happened to fill. So: fit a centreline first, measure everything
// perpendicular to it, use percentiles instead of extrema, and scale the
// smoothing budget with the bin count.
//
// The percentile edges are inset BY CONSTRUCTION and that is not a bug to fix:
// on a road of uniform return density the 2nd percentile sits 2% of the width
// inside the true edge, so a 6.00 m road reads 5.76 m and the tests compare
// against 5.76 deliberately.

#ifndef RANSAC_SEG_CPP__LANE_HPP_
#define RANSAC_SEG_CPP__LANE_HPP_

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace ransac_seg_cpp
{

/// A lane boundary pair expressed in the frame of the fitted centreline.
///
/// `centerline`, `left` and `right` are 2 * n_samples row-major xy; `width` and
/// `arclength` are n_samples. `valid` is exactly `reason.empty()`.
struct LaneFit
{
  std::vector<double> centerline;
  std::vector<double> left;
  std::vector<double> right;
  std::vector<double> width;
  std::vector<double> arclength;
  bool valid = false;
  /// "", "too_few_points", "gap", "bad_bins", "crossed" or
  /// "width_out_of_range".
  std::string reason;
};

/// Defaults mirror ransac_seg.lane.fit_lane.
struct LaneOptions
{
  double bin_length = 2.0;
  int min_points_per_bin = 8;
  double edge_percentile = 2.0;
  int centerline_degree = 2;
  int robust_iterations = 3;
  double max_gap = 6.0;
  double smoothing_per_point = 0.02;
  int n_samples = 120;
  double min_width = 2.0;
  double max_width = 12.0;
  double min_bin_fill = 0.5;
};

/// Fit left/right lane boundaries in the frame of a fitted road centreline.
///
/// `xyz` is the already range-gated ground return set with `stride` columns
/// (2 or more; only x and y are read, so an (N, 3) cloud is accepted). NEVER
/// THROWS: every failure comes back as a LaneFit carrying a machine-readable
/// `reason`, so a caller can publish a clear marker instead of guessing.
/// "too_few_points", "gap" and "bad_bins" carry empty arrays; "crossed" and
/// "width_out_of_range" carry full-length ones.
///
/// "bad_bins" is the one reason with no counterpart in ransac_seg.lane, and it
/// is a DECLARED DEVIATION. It answers a `bin_length` that is not finite and
/// positive, and an arc-length span that would need more than a million bins —
/// a bin_length in the wrong units, or a far-field return the caller did not
/// range-gate. numpy meets both with a MemoryError or a wrapped index; this
/// used to meet them with an uncaught std::bad_alloc, std::length_error or
/// std::invalid_argument thrown straight out of a subscription callback, which
/// is the "never throws" contract above being false rather than a behaviour to
/// preserve.
LaneFit fit_lane(const float * xyz, std::size_t count, int stride, const LaneOptions & opts);

/// Internals exposed only because the tests pin their exact rules.
namespace lane_detail
{

/// Resample a 2 * K row-major polyline to uniform 0.25 m vertex spacing.
///
/// The vertex count uses Python's half-to-even round(), not std::round: they
/// differ on exact halves and one extra vertex shifts every downstream
/// nearest-vertex projection.
std::vector<double> resample(const std::vector<double> & curve);

/// The [start, stop) slice of ascending bin centres forming the longest
/// gap-free run, scored by METRES SPANNED rather than bin count — three bins
/// spanning 30 m is more road than five spanning 10 m. A gap of exactly
/// `max_gap` does not break the run; ties take the first maximum.
std::pair<std::size_t, std::size_t> longest_run(
  const std::vector<double> & centres, double max_gap);

}  // namespace lane_detail

}  // namespace ransac_seg_cpp

#endif  // RANSAC_SEG_CPP__LANE_HPP_
