#include "ransac_seg_cpp/plane_filter.hpp"

#include <algorithm>
#include <cmath>

namespace ransac_seg_cpp
{

namespace
{

/// Angle in degrees between two unit normals, clamped so acos cannot NaN on a
/// dot product that rounds a hair past 1.
double tilt_between_deg(const Eigen::Vector3d & a, const Eigen::Vector3d & b)
{
  const double dot = std::clamp(a.dot(b), -1.0, 1.0);
  constexpr double kPi = 3.14159265358979323846;
  return std::acos(dot) * 180.0 / kPi;
}

}  // namespace

PlaneFilter::PlaneFilter(
  double alpha_in, double max_tilt_jump_deg_in, double max_offset_jump_in,
  int reset_after_misses_in)
: alpha(alpha_in),
  max_tilt_jump_deg(max_tilt_jump_deg_in),
  max_offset_jump(max_offset_jump_in),
  reset_after_misses(reset_after_misses_in)
{
}

void PlaneFilter::reset()
{
  last_.reset();
  misses_ = 0;
}

std::optional<PlaneModel> PlaneFilter::accept(const PlaneModel & model)
{
  last_ = model;
  misses_ = 0;
  return last_;
}

std::optional<PlaneModel> PlaneFilter::miss()
{
  ++misses_;
  if (misses_ >= reset_after_misses) {
    reset();
  }
  return last_;
}

std::optional<PlaneModel> PlaneFilter::update(const std::optional<PlaneModel> & model)
{
  if (!model) {
    return miss();
  }
  if (!last_) {
    return accept(*model);
  }

  const double tilt_jump = tilt_between_deg(last_->normal, model->normal);
  const double offset_jump = std::fabs(model->offset - last_->offset);
  if (tilt_jump > max_tilt_jump_deg || offset_jump > max_offset_jump) {
    return miss();
  }

  const Eigen::Vector3d blended = (1.0 - alpha) * last_->normal + alpha * model->normal;
  const double norm = blended.norm();
  if (norm < 1e-9) {
    // Opposed normals cannot be averaged; trust the fresh measurement.
    return accept(*model);
  }

  PlaneModel filtered = *model;  // carries inlier_ratio/iterations/refined
  filtered.normal = blended / norm;
  filtered.offset = (1.0 - alpha) * last_->offset + alpha * model->offset;
  last_ = filtered;
  misses_ = 0;
  return last_;
}

}  // namespace ransac_seg_cpp
