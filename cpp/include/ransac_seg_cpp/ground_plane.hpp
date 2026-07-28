// RANSAC ground-plane segmentation — C++ port of ransac_seg/ground_plane.py.
//
// The Python estimator is vectorised numpy and costs 15-27 ms on a real 60-70k
// sweep, which is most of the per-frame budget once six nodes each repeat it.
// This is the same algorithm, decision for decision, so the two can be compared
// on identical frames: same minimal 3-point hypotheses, same scale-invariant
// degeneracy test, same adaptive iteration budget from the inlier ratio, same
// tilt gate, same SVD total-least-squares refit kept only if it does not lose
// inliers.
//
// Points arrive as a flat float array because that is how both a KITTI .bin and
// a PointCloud2 payload already sit in memory; copying them into a container of
// vectors costs more than the fit.

#ifndef RANSAC_SEG_CPP__GROUND_PLANE_HPP_
#define RANSAC_SEG_CPP__GROUND_PLANE_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <Eigen/Dense>

namespace ransac_seg_cpp
{

/// Defaults mirror ransac_seg/ground_plane.py; distances are metres.
struct RansacParams
{
  double distance_threshold = 0.20;
  int max_iterations = 200;
  int min_iterations = 12;
  double confidence = 0.99;
  double max_tilt_deg = 10.0;
  double min_inlier_ratio = 0.15;
  int max_scoring_points = 20000;
  bool refine = true;
  uint64_t seed = 0;
};

/// A fitted plane `normal . x + offset = 0` with a unit, +z-oriented normal.
struct PlaneModel
{
  Eigen::Vector3d normal{0.0, 0.0, 1.0};
  double offset = 0.0;
  double inlier_ratio = 0.0;
  int iterations = 0;
  bool refined = false;

  /// Angle in degrees between the plane normal and +z.
  double tilt_deg() const;
};

/// Inlier mask plus the model, or no model when nothing cleared the gates.
struct Segmentation
{
  std::vector<uint8_t> ground;   ///< one byte per input point, 1 = ground
  std::optional<PlaneModel> model;
};

/// Fit a ground plane over `count` points held as x,y,z triples in `xyz`.
///
/// Non-finite points are ignored by the fit but still occupy their slot in the
/// returned mask, so it always indexes the caller's array.
Segmentation fit_plane_ransac(
  const float * xyz, std::size_t count, const RansacParams & params = {});

/// Fit the ground, falling back to a fixed height cut when no plane is found.
///
/// Mirrors `segment_ground`: an absent model tells the caller the fallback ran.
Segmentation segment_ground(
  const float * xyz, std::size_t count, const RansacParams & params = {},
  double fallback_height = 0.2);

}  // namespace ransac_seg_cpp

#endif  // RANSAC_SEG_CPP__GROUND_PLANE_HPP_
