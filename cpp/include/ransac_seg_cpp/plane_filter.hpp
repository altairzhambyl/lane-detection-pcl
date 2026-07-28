// Temporal low-pass filter over RANSAC plane estimates — port of
// ransac_seg/plane_filter.py.
//
// RANSAC redraws its minimal samples every frame, so consecutive fits of the
// same physical road differ slightly. That jitter moves the plane vertically at
// range by an amount comparable to the inlier band itself, which makes ground
// membership flicker for far-field points and makes any downstream rasterisation
// strobe.
//
// An innovation gate sits in front of the blend: a fit that disagrees violently
// with the accumulated plane is a bad fit (a wall, a truck flank, a frame where
// the road left the field of view), not a real road change, so it is dropped
// rather than averaged in.

#ifndef RANSAC_SEG_CPP__PLANE_FILTER_HPP_
#define RANSAC_SEG_CPP__PLANE_FILTER_HPP_

#include <optional>

#include "ransac_seg_cpp/ground_plane.hpp"

namespace ransac_seg_cpp
{

/// Low-pass filter over successive PlaneModel estimates with an innovation gate.
class PlaneFilter
{
public:
  /// Configure blend weight, gate thresholds and the miss budget.
  explicit PlaneFilter(
    double alpha = 0.3, double max_tilt_jump_deg = 5.0, double max_offset_jump = 0.5,
    int reset_after_misses = 5);

  // Public because the node rewrites them live from its parameter callback,
  // exactly as the Python node assigns filter.alpha and friends.
  double alpha;
  double max_tilt_jump_deg;
  double max_offset_jump;
  int reset_after_misses;

  /// The most recent filtered plane, or nullopt.
  const std::optional<PlaneModel> & last() const {return last_;}

  /// Forget the accumulated plane and the miss counter.
  void reset();

  /// Fold one estimate into the filter and return the plane to use.
  ///
  /// nullopt (or a gated-out estimate) holds the last good plane and counts as
  /// a miss; after `reset_after_misses` consecutive misses the filter gives up
  /// and returns nullopt so callers fall back rather than steering by a stale
  /// road.
  std::optional<PlaneModel> update(const std::optional<PlaneModel> & model);

private:
  std::optional<PlaneModel> accept(const PlaneModel & model);
  std::optional<PlaneModel> miss();

  std::optional<PlaneModel> last_;
  int misses_ = 0;
};

}  // namespace ransac_seg_cpp

#endif  // RANSAC_SEG_CPP__PLANE_FILTER_HPP_
