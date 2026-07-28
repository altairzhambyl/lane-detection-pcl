#include "ransac_seg_cpp/numeric.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace ransac_seg_cpp
{

double percentile_linear(std::vector<double> values, double q)
{
  if (values.empty()) {
    throw std::invalid_argument("percentile_linear needs at least one value");
  }
  if (!(q >= 0.0 && q <= 100.0)) {
    throw std::invalid_argument("percentile_linear needs q in [0, 100]");
  }
  std::sort(values.begin(), values.end());

  // numpy's 'linear' method: a virtual rank on [0, n-1], interpolated between
  // the two order statistics that bracket it. Nearest-rank would quantise the
  // answer to a sample value and bias small bins.
  const double rank = static_cast<double>(values.size() - 1) * q / 100.0;
  const double floor_rank = std::floor(rank);
  const std::size_t lo = static_cast<std::size_t>(floor_rank);
  if (lo + 1 >= values.size()) {
    return values.back();
  }
  const double frac = rank - floor_rank;
  return values[lo] + frac * (values[lo + 1] - values[lo]);
}

double median(std::vector<double> values)
{
  if (values.empty()) {
    throw std::invalid_argument("median needs at least one value");
  }
  const std::size_t n = values.size();
  const std::size_t mid = n / 2;
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
  const double upper = values[mid];
  if (n % 2 == 1) {
    return upper;
  }
  // Even sample: numpy averages the two middle values. nth_element already
  // partitioned, so the lower middle is the max of the left partition.
  const double lower =
    *std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid));
  return 0.5 * (lower + upper);
}

double banker_round(double v)
{
  if (!std::isfinite(v)) {
    return v;
  }
  const double floored = std::floor(v);
  const double frac = v - floored;
  if (frac > 0.5) {
    return floored + 1.0;
  }
  if (frac < 0.5) {
    return floored;
  }
  // Exactly halfway: pick the even neighbour. std::round would go away from
  // zero here, which is the one case Python's round() disagrees with.
  return (std::fmod(floored, 2.0) == 0.0) ? floored : floored + 1.0;
}

void interp(
  const double * xp, const double * fp, std::size_t n,
  const double * x, std::size_t m, double * out)
{
  if (n == 0) {
    throw std::invalid_argument("interp needs at least one sample point");
  }
  for (std::size_t i = 0; i < m; ++i) {
    const double xi = x[i];
    if (!(xi > xp[0])) {          // also catches NaN, which numpy maps to fp[0]
      out[i] = fp[0];
      continue;
    }
    if (xi >= xp[n - 1]) {
      out[i] = fp[n - 1];
      continue;
    }
    const std::size_t hi =
      static_cast<std::size_t>(std::upper_bound(xp, xp + n, xi) - xp);
    const std::size_t lo = hi - 1;
    const double span = xp[hi] - xp[lo];
    out[i] = (span == 0.0) ? fp[lo] : fp[lo] + (xi - xp[lo]) * (fp[hi] - fp[lo]) / span;
  }
}

void gradient(
  const double * v, std::size_t n, std::size_t stride,
  double * out, std::size_t out_stride)
{
  if (n < 2) {
    throw std::invalid_argument("gradient needs at least two samples");
  }
  out[0] = v[stride] - v[0];
  for (std::size_t i = 1; i + 1 < n; ++i) {
    out[i * out_stride] = 0.5 * (v[(i + 1) * stride] - v[(i - 1) * stride]);
  }
  out[(n - 1) * out_stride] = v[(n - 1) * stride] - v[(n - 2) * stride];
}

void linspace(double a, double b, std::size_t n, double * out)
{
  if (n == 0) {
    return;
  }
  if (n == 1) {
    out[0] = a;
    return;
  }
  const double step = (b - a) / static_cast<double>(n - 1);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = a + step * static_cast<double>(i);
  }
  // a + step*(n-1) drifts by an ulp or two; the endpoint is load-bearing for
  // arc-length resampling, so pin it exactly the way numpy does.
  out[n - 1] = b;
}

std::vector<std::size_t> stable_argsort(const std::vector<int64_t> & keys)
{
  std::vector<std::size_t> order(keys.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(
    order.begin(), order.end(),
    [&keys](std::size_t a, std::size_t b) {return keys[a] < keys[b];});
  return order;
}

int64_t floor_index(double v)
{
  // kMaxLatticeIndex is a power of two, so the bound converts to double exactly
  // and neither comparison is itself a rounding decision.
  const double limit = static_cast<double>(kMaxLatticeIndex);
  double clamped = std::floor(v);
  // Written as a NEGATED comparison so NaN — for which every ordering test is
  // false — takes this branch too, without a separate isnan. That matters: this
  // runs once per coordinate per point in the voxel and bin lattices, and the
  // two-compare form compiles to conditional moves where an isnan-first form
  // put a third branch in the loop.
  if (!(clamped > -limit)) {clamped = -limit;}
  if (clamped > limit) {clamped = limit;}
  return static_cast<int64_t>(clamped);
}

}  // namespace ransac_seg_cpp
