#include "ransac_seg_cpp/ground_plane.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace ransac_seg_cpp
{
namespace
{

constexpr std::size_t kMinSample = 3;

/// Flip a plane so its normal points along +z, keeping the same surface.
inline void orient_up(Eigen::Vector3d & normal, double & offset)
{
  if (normal.z() < 0.0) {
    normal = -normal;
    offset = -offset;
  }
}

/// Plane through three points, or false when they are degenerate.
///
/// The cross-product magnitude is compared against the sample's own extent so
/// the test is unit-free: three near-collinear or near-coincident draws give a
/// tiny cross product relative to the edges that produced it.
bool plane_from_sample(
  const Eigen::Vector3d & p0, const Eigen::Vector3d & p1, const Eigen::Vector3d & p2,
  Eigen::Vector3d & normal, double & offset)
{
  const Eigen::Vector3d e1 = p1 - p0;
  const Eigen::Vector3d e2 = p2 - p0;
  normal = e1.cross(e2);
  const double norm = normal.norm();
  const double scale = e1.norm() * e2.norm();
  if (scale <= 0.0 || norm < 1e-6 * scale) {
    return false;
  }
  normal /= norm;
  offset = -normal.dot(p0);
  orient_up(normal, offset);
  return true;
}

/// Samples still needed to reach `confidence`, given an inlier ratio.
int required_iterations(double inlier_ratio, double confidence, int cap)
{
  if (inlier_ratio <= 0.0) {
    return cap;
  }
  const double all_inliers = std::pow(inlier_ratio, static_cast<double>(kMinSample));
  if (all_inliers >= 1.0) {
    return 1;
  }
  const double denominator = std::log1p(-all_inliers);
  if (denominator >= 0.0) {
    return cap;
  }
  const double needed = std::ceil(std::log1p(-confidence) / denominator);
  return static_cast<int>(std::min(static_cast<double>(cap), needed));
}

/// Total-least-squares refit over the consensus set, via SVD of the centred cloud.
bool refit_total_least_squares(
  const std::vector<Eigen::Vector3d> & points, const std::vector<uint32_t> & inliers,
  Eigen::Vector3d & normal, double & offset)
{
  if (inliers.size() < kMinSample) {
    return false;
  }
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  for (const uint32_t i : inliers) {
    centroid += points[i];
  }
  centroid /= static_cast<double>(inliers.size());

  // Accumulate the scatter matrix rather than materialising an N x 3 matrix:
  // its eigenvectors are the right-singular vectors of the centred cloud, and
  // the smallest belongs to the direction of least variance — the normal.
  Eigen::Matrix3d scatter = Eigen::Matrix3d::Zero();
  for (const uint32_t i : inliers) {
    const Eigen::Vector3d d = points[i] - centroid;
    scatter += d * d.transpose();
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(scatter);
  if (solver.info() != Eigen::Success) {
    return false;
  }
  const Eigen::Vector3d eigenvalues = solver.eigenvalues();
  // Points filling a sphere rather than a sheet describe no meaningful plane.
  if (eigenvalues(0) > eigenvalues(2) * 0.98) {
    return false;
  }
  normal = solver.eigenvectors().col(0).normalized();
  offset = -normal.dot(centroid);
  orient_up(normal, offset);
  return true;
}

}  // namespace

double PlaneModel::tilt_deg() const
{
  return std::acos(std::clamp(normal.z(), -1.0, 1.0)) * 180.0 / M_PI;
}

Segmentation fit_plane_ransac(
  const float * xyz, std::size_t count, const RansacParams & params)
{
  Segmentation out;
  out.ground.assign(count, 0);

  // Finite points only, keeping the map back to the caller's indices.
  std::vector<Eigen::Vector3d> points;
  std::vector<uint32_t> origin;
  points.reserve(count);
  origin.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const float x = xyz[3 * i], y = xyz[3 * i + 1], z = xyz[3 * i + 2];
    if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
      points.emplace_back(x, y, z);
      origin.push_back(static_cast<uint32_t>(i));
    }
  }
  const std::size_t total = points.size();
  if (total < kMinSample) {
    return out;
  }

  std::mt19937_64 rng(params.seed);
  std::uniform_int_distribution<std::size_t> pick(0, total - 1);
  const double cos_limit = std::cos(params.max_tilt_deg * M_PI / 180.0);

  // Hypotheses are scored against a fixed random subset on large clouds:
  // consensus size is a ratio, so a few thousand points estimate it closely
  // while keeping the loop O(subset). The winner is re-scored against all.
  std::vector<uint32_t> scoring;
  if (params.max_scoring_points > 0 &&
    static_cast<std::size_t>(params.max_scoring_points) < total)
  {
    scoring.resize(total);
    for (std::size_t i = 0; i < total; ++i) {
      scoring[i] = static_cast<uint32_t>(i);
    }
    std::shuffle(scoring.begin(), scoring.end(), rng);
    scoring.resize(static_cast<std::size_t>(params.max_scoring_points));
  } else {
    scoring.resize(total);
    for (std::size_t i = 0; i < total; ++i) {
      scoring[i] = static_cast<uint32_t>(i);
    }
  }

  Eigen::Vector3d best_normal = Eigen::Vector3d::UnitZ();
  double best_offset = 0.0;
  std::size_t best_count = 0;
  bool have_best = false;
  int budget = std::max(params.min_iterations, 1);
  int iterations = 0;

  while (iterations < std::min(budget, params.max_iterations)) {
    ++iterations;
    const Eigen::Vector3d & p0 = points[pick(rng)];
    const Eigen::Vector3d & p1 = points[pick(rng)];
    const Eigen::Vector3d & p2 = points[pick(rng)];

    Eigen::Vector3d normal;
    double offset;
    if (!plane_from_sample(p0, p1, p2, normal, offset)) {
      continue;
    }
    if (normal.z() < cos_limit) {
      // Wall, vehicle flank or kerb face rather than road surface.
      continue;
    }

    std::size_t hits = 0;
    for (const uint32_t i : scoring) {
      if (std::abs(points[i].dot(normal) + offset) <= params.distance_threshold) {
        ++hits;
      }
    }

    if (hits > best_count) {
      best_count = hits;
      best_normal = normal;
      best_offset = offset;
      have_best = true;
      const double ratio = static_cast<double>(hits) / static_cast<double>(scoring.size());
      budget = std::max(
        params.min_iterations,
        std::min(params.max_iterations,
        required_iterations(ratio, params.confidence, params.max_iterations)));
    }
  }

  if (!have_best) {
    return out;
  }

  std::vector<uint32_t> inliers;
  inliers.reserve(total);
  for (std::size_t i = 0; i < total; ++i) {
    if (std::abs(points[i].dot(best_normal) + best_offset) <= params.distance_threshold) {
      inliers.push_back(static_cast<uint32_t>(i));
    }
  }

  bool refined = false;
  if (params.refine) {
    Eigen::Vector3d normal;
    double offset;
    if (refit_total_least_squares(points, inliers, normal, offset) && normal.z() >= cos_limit) {
      std::vector<uint32_t> candidate;
      candidate.reserve(inliers.size());
      for (std::size_t i = 0; i < total; ++i) {
        if (std::abs(points[i].dot(normal) + offset) <= params.distance_threshold) {
          candidate.push_back(static_cast<uint32_t>(i));
        }
      }
      // Keep the refit only if it explains at least as much of the cloud; one
      // skewed by a sloped shoulder can lose inliers.
      if (candidate.size() >= inliers.size()) {
        best_normal = normal;
        best_offset = offset;
        inliers.swap(candidate);
        refined = true;
      }
    }
  }

  const double ratio = static_cast<double>(inliers.size()) / static_cast<double>(total);
  if (ratio < params.min_inlier_ratio) {
    return out;
  }

  for (const uint32_t i : inliers) {
    out.ground[origin[i]] = 1;
  }
  PlaneModel model;
  model.normal = best_normal;
  model.offset = best_offset;
  model.inlier_ratio = ratio;
  model.iterations = iterations;
  model.refined = refined;
  out.model = model;
  return out;
}

Segmentation segment_ground(
  const float * xyz, std::size_t count, const RansacParams & params, double fallback_height)
{
  Segmentation out = fit_plane_ransac(xyz, count, params);
  if (out.model.has_value()) {
    return out;
  }
  // No plane cleared the gates. Fall back to the fixed height cut and leave the
  // model empty, so the caller can tell a fit from a fallback and say so.
  out.ground.assign(count, 0);
  for (std::size_t i = 0; i < count; ++i) {
    const float z = xyz[3 * i + 2];
    if (std::isfinite(z) && z < fallback_height) {
      out.ground[i] = 1;
    }
  }
  return out;
}

}  // namespace ransac_seg_cpp
