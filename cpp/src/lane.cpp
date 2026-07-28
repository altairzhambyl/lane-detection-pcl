#include "ransac_seg_cpp/lane.hpp"

#include "ransac_seg_cpp/numeric.hpp"
#include "ransac_seg_cpp/spline.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>

#include <Eigen/Dense>

namespace ransac_seg_cpp
{

namespace
{

// Vertex spacing of the internal centreline polyline. Fine enough that the
// nearest-vertex projection quantises arc length by half of it, coarse enough
// that a 60 m road stays a few hundred vertices.
constexpr double kCurveSpacing = 0.25;
constexpr int kMaxCurveVertices = 4000;

// Most bins fit_lane will lay down before it calls the request unrepresentable.
//
// A gated sweep is 1-100 m of road at a 2 m bin, so ~50 bins; a million is four
// orders of magnitude of headroom and 8 MB of centres. Past it the request is a
// bug — a bin_length in the wrong units, or an ungated far-field return — and
// the Python answers it with a MemoryError while this used to answer with an
// uncaught std::bad_alloc out of a subscription callback. build_costmap draws
// the same line for the same reason (kMaxCells in cost_grid.hpp).
constexpr int64_t kMaxBins = 1000000;

LaneFit empty_fit(const char * reason)
{
  LaneFit fit;
  fit.valid = false;
  fit.reason = reason;
  return fit;
}

/// Coerce a strided float cloud to finite (N, 2) xy, dropping non-finite rows.
std::vector<double> as_xy(const float * xyz, std::size_t count, int stride)
{
  std::vector<double> xy;
  if (xyz == nullptr || count == 0 || stride < 2) {return xy;}
  xy.reserve(2 * count);
  for (std::size_t i = 0; i < count; ++i) {
    const float * row = xyz + i * static_cast<std::size_t>(stride);
    if (!std::isfinite(row[0]) || !std::isfinite(row[1])) {continue;}
    xy.push_back(static_cast<double>(row[0]));
    xy.push_back(static_cast<double>(row[1]));
  }
  return xy;
}

/// Dominant direction of the mean-centred cloud, oriented towards +x.
Eigen::Vector2d principal_axis(const std::vector<double> & xy, Eigen::Vector2d & mean_out)
{
  const std::size_t n = xy.size() / 2;
  Eigen::Vector2d mean = Eigen::Vector2d::Zero();
  for (std::size_t i = 0; i < n; ++i) {
    mean.x() += xy[2 * i];
    mean.y() += xy[2 * i + 1];
  }
  mean /= static_cast<double>(n);
  mean_out = mean;

  double sxx = 0.0;
  double sxy = 0.0;
  double syy = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double dx = xy[2 * i] - mean.x();
    const double dy = xy[2 * i + 1] - mean.y();
    sxx += dx * dx;
    sxy += dx * dy;
    syy += dy * dy;
  }

  // The first right-singular vector of the centred cloud is the top eigenvector
  // of its scatter matrix, and at 2x2 that is closed form — no iterative solver,
  // and no dependence on an SVD's internal sign convention beyond the +x flip
  // below, which is the only sign rule the Python states.
  const double half_trace = 0.5 * (sxx + syy);
  const double gap = 0.5 * (sxx - syy);
  const double top = half_trace + std::hypot(gap, sxy);
  Eigen::Vector2d axis;
  if (sxy != 0.0) {
    // Both (sxy, top - sxx) and (top - syy, sxy) span the eigenspace; take the
    // longer one so a near-axis-aligned cloud does not normalise noise.
    const Eigen::Vector2d a{sxy, top - sxx};
    const Eigen::Vector2d b{top - syy, sxy};
    axis = a.squaredNorm() >= b.squaredNorm() ? a : b;
  } else {
    axis = sxx >= syy ? Eigen::Vector2d{1.0, 0.0} : Eigen::Vector2d{0.0, 1.0};
  }
  // A heading pointing backwards would run arc length far to near and mirror
  // the left/right labels, so pin it to the sensor's forward +x. Strict: an
  // axis with x exactly 0 is left alone, as in the Python.
  if (axis.x() < 0.0) {axis = -axis;}
  return axis / axis.norm();
}

/// Unit tangents and cumulative arc length of a 2 * K row-major polyline.
void tangents(
  const std::vector<double> & curve, std::vector<double> & tangent,
  std::vector<double> & arclength)
{
  const std::size_t k = curve.size() / 2;
  tangent.assign(2 * k, 0.0);
  // edge_order=1: the one-sided end differences set the end tangents, which set
  // the end normals, which place the outermost boundary points.
  gradient(curve.data(), k, 2, tangent.data(), 2);
  gradient(curve.data() + 1, k, 2, tangent.data() + 1, 2);
  for (std::size_t i = 0; i < k; ++i) {
    double norm = std::hypot(tangent[2 * i], tangent[2 * i + 1]);
    // Force the norm to 1 rather than dividing by ~0: that leaves a zero
    // tangent, which is finite, instead of an infinity.
    if (norm < 1e-12) {norm = 1.0;}
    tangent[2 * i] /= norm;
    tangent[2 * i + 1] /= norm;
  }
  arclength.assign(k, 0.0);
  for (std::size_t i = 1; i < k; ++i) {
    arclength[i] = arclength[i - 1] +
      std::hypot(curve[2 * i] - curve[2 * i - 2], curve[2 * i + 1] - curve[2 * i - 1]);
  }
}

/// One-dimensional index over the curve vertices for nearest-vertex queries.
///
/// DECLARED DEVIATION from scipy's cKDTree. The curve is not a point cloud: it
/// is a polyline of a few hundred vertices strung along one dominant direction,
/// so sorting the vertices by their projection u onto that direction and
/// walking outwards from the query's own u is both exact and cheaper than a
/// tree. Exact because a vertex whose |u - u_query| already exceeds the best
/// distance found so far cannot be nearer than it, so the walk stops with a
/// proof rather than a heuristic. On an exact distance tie it returns the
/// SMALLEST vertex index — scipy's kd-tree tie order is not reproducible, so
/// the rule is stated rather than guessed.
///
/// (A uniform 2-D bucket grid was tried first and measured SLOWER than scipy:
/// road returns sit up to a lane half-width off the curve, so every query
/// swept five rings of mostly empty cells. 105 ms against 88 ms for the Python
/// on the 120k-point obstacle scene.)
class CurveIndex
{
public:
  explicit CurveIndex(const std::vector<double> & curve)
  {
    const std::size_t k = curve.size() / 2;
    for (std::size_t i = 0; i < k; ++i) {
      mean_x_ += curve[2 * i];
      mean_y_ += curve[2 * i + 1];
    }
    mean_x_ /= static_cast<double>(k);
    mean_y_ /= static_cast<double>(k);

    double sxx = 0.0;
    double sxy = 0.0;
    double syy = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
      const double dx = curve[2 * i] - mean_x_;
      const double dy = curve[2 * i + 1] - mean_y_;
      sxx += dx * dx;
      sxy += dx * dy;
      syy += dy * dy;
    }
    const double top = 0.5 * (sxx + syy) + std::hypot(0.5 * (sxx - syy), sxy);
    if (sxy != 0.0) {
      const double ax = sxy;
      const double ay = top - sxx;
      const double bx = top - syy;
      const double by = sxy;
      if (ax * ax + ay * ay >= bx * bx + by * by) {
        dir_x_ = ax;
        dir_y_ = ay;
      } else {
        dir_x_ = bx;
        dir_y_ = by;
      }
      const double norm = std::hypot(dir_x_, dir_y_);
      dir_x_ /= norm;
      dir_y_ /= norm;
    } else if (sxx >= syy) {
      dir_x_ = 1.0;
      dir_y_ = 0.0;
    } else {
      dir_x_ = 0.0;
      dir_y_ = 1.0;
    }

    order_.resize(k);
    std::iota(order_.begin(), order_.end(), 0);
    std::vector<double> u(k);
    for (std::size_t i = 0; i < k; ++i) {u[i] = project_u(curve[2 * i], curve[2 * i + 1]);}
    std::sort(
      order_.begin(), order_.end(),
      [&u](int a, int b) {return u[static_cast<std::size_t>(a)] < u[static_cast<std::size_t>(b)];});
    // Vertex coordinates are copied into u-order so the walk below is one
    // sequential pass over three contiguous arrays instead of a gather.
    sorted_u_.resize(k);
    sorted_x_.resize(k);
    sorted_y_.resize(k);
    for (std::size_t rank = 0; rank < k; ++rank) {
      const std::size_t v = static_cast<std::size_t>(order_[rank]);
      sorted_u_[rank] = u[v];
      sorted_x_[rank] = curve[2 * v];
      sorted_y_[rank] = curve[2 * v + 1];
    }
  }

  std::size_t nearest(double px, double py) const
  {
    const std::ptrdiff_t k = static_cast<std::ptrdiff_t>(sorted_u_.size());
    const double uq = project_u(px, py);
    std::ptrdiff_t right = static_cast<std::ptrdiff_t>(
      std::lower_bound(sorted_u_.begin(), sorted_u_.end(), uq) - sorted_u_.begin());
    std::ptrdiff_t left = right - 1;

    double best = std::numeric_limits<double>::infinity();
    std::size_t best_index = 0;
    const double infinite = std::numeric_limits<double>::infinity();
    while (left >= 0 || right < k) {
      const double gap_left = left >= 0 ? uq - sorted_u_[static_cast<std::size_t>(left)] : infinite;
      const double gap_right =
        right < k ? sorted_u_[static_cast<std::size_t>(right)] - uq : infinite;
      const double gap = std::min(gap_left, gap_right);
      // Nothing further out along u can beat the incumbent, because the
      // distance to a vertex is at least its separation in u. Expanding in
      // order of |u - u_query| means this fires as early as it possibly can.
      if (gap * gap >= best) {break;}
      consider(gap_left <= gap_right ? left-- : right++, px, py, best, best_index);
    }
    return best_index;
  }

private:
  void consider(
    std::ptrdiff_t rank, double px, double py, double & best, std::size_t & best_index) const
  {
    const std::size_t r = static_cast<std::size_t>(rank);
    const double dx = sorted_x_[r] - px;
    const double dy = sorted_y_[r] - py;
    const double d2 = dx * dx + dy * dy;
    const std::size_t v = static_cast<std::size_t>(order_[r]);
    // Smallest vertex index wins an exact tie; see the class comment.
    if (d2 < best || (d2 == best && v < best_index)) {
      best = d2;
      best_index = v;
    }
  }

  double project_u(double x, double y) const
  {
    return (x - mean_x_) * dir_x_ + (y - mean_y_) * dir_y_;
  }

  double mean_x_ = 0.0;
  double mean_y_ = 0.0;
  double dir_x_ = 1.0;
  double dir_y_ = 0.0;
  std::vector<int> order_;
  std::vector<double> sorted_u_;
  std::vector<double> sorted_x_;
  std::vector<double> sorted_y_;
};

/// Project points onto a polyline, returning (s, d) in its local frame.
///
/// `s` is arc length along the curve and `d` the signed lateral offset,
/// positive to the LEFT of the direction of travel. The nearest CURVE VERTEX is
/// used, not the nearest point on the nearest segment: the 0.25 m spacing is
/// chosen so this quantises arc length by half of it, and switching to exact
/// segment projection would be a behaviour change, not a refinement.
///
/// This is the hot loop of the whole fit — every point is re-projected on every
/// refinement pass — which is why the Python hands scipy workers=-1 and why
/// this loop is an OpenMP parallel for.
void project(
  const std::vector<double> & curve, const std::vector<double> & xy,
  std::vector<double> & s_out, std::vector<double> & d_out)
{
  std::vector<double> tangent;
  std::vector<double> arclength;
  tangents(curve, tangent, arclength);
  const CurveIndex index(curve);
  const std::size_t n = xy.size() / 2;
  s_out.assign(n, 0.0);
  d_out.assign(n, 0.0);
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
    const std::size_t u = static_cast<std::size_t>(i);
    const std::size_t v = index.nearest(xy[2 * u], xy[2 * u + 1]);
    const double dx = xy[2 * u] - curve[2 * v];
    const double dy = xy[2 * u + 1] - curve[2 * v + 1];
    const double tx = tangent[2 * v];
    const double ty = tangent[2 * v + 1];
    s_out[u] = arclength[v] + dx * tx + dy * ty;
    d_out[u] = -dx * ty + dy * tx;
  }
}

/// Bin index per sample plus the absolute bin centres present.
///
/// False means the requested binning is not representable — see kMaxBins — and
/// the outputs are left empty.
bool bin_edges(
  const std::vector<double> & s, double bin_length,
  std::vector<int64_t> & index_out, std::vector<double> & centres_out)
{
  index_out.clear();
  centres_out.clear();
  if (!(bin_length > 0.0) || !std::isfinite(bin_length)) {
    return false;
  }
  const double origin = *std::min_element(s.begin(), s.end());
  if (!std::isfinite(origin)) {
    return false;
  }
  index_out.assign(s.size(), 0);
  int64_t lo = std::numeric_limits<int64_t>::max();
  int64_t hi = std::numeric_limits<int64_t>::min();
  for (std::size_t i = 0; i < s.size(); ++i) {
    // Saturating: an unbounded s (a far-field return the caller did not gate)
    // divided by bin_length leaves the int64 range, and the plain cast used to
    // land on INT64_MIN and signed-overflow the rebasing just below.
    const int64_t k = floor_index((s[i] - origin) / bin_length);
    index_out[i] = k;
    lo = std::min(lo, k);
    hi = std::max(hi, k);
  }
  // A no-op given the origin, kept because the Python keeps it.
  if (lo != 0) {
    for (int64_t & k : index_out) {k -= lo;}
    hi -= lo;
  }
  if (hi >= kMaxBins) {
    index_out.clear();
    return false;
  }
  // Centres are ABSOLUTE s coordinates (origin may be negative), not 0-based:
  // refine_centerline evaluates its polynomial at arc lengths clamped to
  // centres.front()/back(), which live in this frame.
  centres_out.assign(static_cast<std::size_t>(hi) + 1, 0.0);
  for (std::size_t b = 0; b < centres_out.size(); ++b) {
    centres_out[b] = origin + (static_cast<double>(b) + 0.5) * bin_length;
  }
  return true;
}

/// Per-bin views of `values`, EXACTLY `n_bins` of them including empty ones so
/// the bin indices stay aligned with the centres.
///
/// The Python reaches this through np.argsort(kind="stable") and a slice, which
/// numpy answers with a radix sort. A comparison sort is not equivalent in
/// cost: routing this through a std::stable_sort of the 120k bin indices made
/// the whole fit 84.85 ms against 56.27 ms for the single pass below (p50 of
/// 25 single-threaded runs on the obstacle scene). So the bins are filled by
/// one pass in original order instead. That yields the same groups: a stable sort by bin
/// index leaves each bin's members in their original relative order, which is
/// exactly the order this pass appends them in.
std::vector<std::vector<double>> grouped(
  const std::vector<double> & values, const std::vector<int64_t> & index, std::size_t n_bins)
{
  std::vector<std::vector<double>> groups(n_bins);
  std::vector<std::size_t> sizes(n_bins, 0);
  for (int64_t bin : index) {++sizes[static_cast<std::size_t>(bin)];}
  for (std::size_t b = 0; b < n_bins; ++b) {groups[b].reserve(sizes[b]);}
  for (std::size_t i = 0; i < index.size(); ++i) {
    groups[static_cast<std::size_t>(index[i])].push_back(values[i]);
  }
  return groups;
}

struct Bins
{
  std::vector<double> centres;
  std::vector<std::vector<double>> groups;
  std::vector<int64_t> counts;
  /// False when bin_edges refused the request; distinct from "no bin passed
  /// the fill floor", which is an ordinary empty result.
  bool ok = true;
};

/// Bin the lateral residuals by arc length and keep the fully observed bins.
///
/// The bins at the two ends of a sweep are only partially covered by returns,
/// so their percentiles describe a sliver of road rather than a cross-section
/// and taper the reported width. A floor RELATIVE to the typical bin population
/// drops them instead of letting them bend the fit.
Bins occupied_bins(
  const std::vector<double> & curve, const std::vector<double> & xy,
  double bin_length, int min_points_per_bin, double min_bin_fill)
{
  Bins bins;
  std::vector<double> s;
  std::vector<double> d;
  project(curve, xy, s, d);
  std::vector<int64_t> index;
  std::vector<double> centres;
  if (!bin_edges(s, bin_length, index, centres)) {
    bins.ok = false;
    return bins;
  }
  std::vector<std::vector<double>> groups = grouped(d, index, centres.size());

  std::vector<int64_t> counts(groups.size());
  std::vector<double> populated;
  for (std::size_t b = 0; b < groups.size(); ++b) {
    counts[b] = static_cast<int64_t>(groups[b].size());
    if (counts[b] >= min_points_per_bin) {populated.push_back(static_cast<double>(counts[b]));}
  }
  if (populated.empty()) {return bins;}

  // np.median over ONLY the populated counts; even-length medians are the mean
  // of the two middle values, which is why numeric::median is used here.
  const double floor_thresh =
    std::max(static_cast<double>(min_points_per_bin), min_bin_fill * median(populated));
  for (std::size_t b = 0; b < groups.size(); ++b) {
    // `counts[b] > 0` is the one guard the Python does not have, and it can
    // only fire when min_points_per_bin <= 0 drives the floor to zero — where
    // the Python hands np.percentile an empty bin and raises. fit_lane's
    // contract is that it never throws, and there is no reference behaviour to
    // diverge from in a regime the reference crashes in.
    if (counts[b] > 0 && static_cast<double>(counts[b]) >= floor_thresh) {
      bins.centres.push_back(centres[b]);
      bins.groups.push_back(std::move(groups[b]));
      bins.counts.push_back(counts[b]);
    }
  }
  return bins;
}

/// Push a centreline polyline towards the median of the returns around it.
std::vector<double> refine_centerline(
  std::vector<double> curve, const std::vector<double> & xy, double bin_length,
  int degree, int iterations, int min_points_per_bin, double min_bin_fill)
{
  for (int pass = 0; pass < iterations; ++pass) {
    // Re-project onto the CURRENT curve every pass, so s is true arc length
    // rather than a fixed-axis coordinate.
    const Bins bins =
      occupied_bins(curve, xy, bin_length, min_points_per_bin, min_bin_fill);
    // A refused binning breaks out here too; fit_lane's own occupied_bins call
    // refuses identically and turns it into the "bad_bins" LaneFit.
    if (!bins.ok || static_cast<int>(bins.centres.size()) < degree + 1) {break;}

    std::vector<double> medians(bins.groups.size());
    std::vector<double> weights(bins.groups.size());
    for (std::size_t b = 0; b < bins.groups.size(); ++b) {
      // MEDIAN, not mean: a roadside car is a one-sided outlier and a mean
      // would let it drag the centreline towards it.
      medians[b] = median(bins.groups[b]);
      // sqrt(count): a bin's median is noisier the fewer points it holds, and
      // an unweighted fit lets the thinnest far-range bin steer the curve.
      weights[b] = std::sqrt(static_cast<double>(bins.counts[b]));
    }
    const std::vector<double> coeffs =
      polyfit_weighted(bins.centres, medians, weights, degree);

    std::vector<double> tangent;
    std::vector<double> arclength;
    tangents(curve, tangent, arclength);
    const std::size_t k = curve.size() / 2;
    std::vector<double> moved(2 * k);
    for (std::size_t i = 0; i < k; ++i) {
      // The clamp is load-bearing, not defensive: extrapolating the degree-2
      // fit past the last occupied bin whips the curve ends out and the next
      // iteration fits that error, diverging.
      const double at =
        std::min(std::max(arclength[i], bins.centres.front()), bins.centres.back());
      const double offset = polyval(coeffs, at);
      moved[2 * i] = curve[2 * i] + offset * -tangent[2 * i + 1];
      moved[2 * i + 1] = curve[2 * i + 1] + offset * tangent[2 * i];
    }
    curve = lane_detail::resample(moved);
  }
  return curve;
}

}  // namespace

namespace lane_detail
{

std::vector<double> resample(const std::vector<double> & curve)
{
  const std::size_t k = curve.size() / 2;
  if (k < 2) {return curve;}
  std::vector<double> cumulative(k, 0.0);
  for (std::size_t i = 1; i < k; ++i) {
    cumulative[i] = cumulative[i - 1] +
      std::hypot(curve[2 * i] - curve[2 * i - 2], curve[2 * i + 1] - curve[2 * i - 1]);
  }
  const double total = cumulative.back();
  if (total <= 0.0) {return curve;}

  // banker_round, not std::round: Python's round() is half-to-even and the two
  // differ on exact halves, which is one extra vertex and a shift in every
  // downstream nearest-vertex projection.
  //
  // Clamped BEFORE the cast, which is also the Python's order
  // (`int(np.clip(round(...) + 1, 2, _MAX_CURVE_VERTICES))`). Casting first put
  // a 1e10 m curve's 4e10 vertices through an out-of-range double-to-int
  // conversion — undefined, and the clamp that exists to catch exactly this
  // then judged whatever came out.
  const double requested = banker_round(total / kCurveSpacing) + 1.0;
  const int count = static_cast<int>(
    std::min(static_cast<double>(kMaxCurveVertices), std::max(2.0, requested)));

  std::vector<double> targets(static_cast<std::size_t>(count));
  linspace(0.0, total, targets.size(), targets.data());

  std::vector<double> column(k);
  std::vector<double> out(2 * targets.size());
  for (int axis = 0; axis < 2; ++axis) {
    for (std::size_t i = 0; i < k; ++i) {
      column[i] = curve[2 * i + static_cast<std::size_t>(axis)];
    }
    std::vector<double> values(targets.size());
    interp(cumulative.data(), column.data(), k, targets.data(), targets.size(), values.data());
    for (std::size_t i = 0; i < targets.size(); ++i) {
      out[2 * i + static_cast<std::size_t>(axis)] = values[i];
    }
  }
  return out;
}

std::pair<std::size_t, std::size_t> longest_run(
  const std::vector<double> & centres, double max_gap)
{
  if (centres.empty()) {return {0, 0};}
  std::vector<std::size_t> starts{0};
  std::vector<std::size_t> stops;
  for (std::size_t i = 0; i + 1 < centres.size(); ++i) {
    // Strictly greater: a gap of exactly max_gap does not break the run. This
    // is np.diff(centres) > max_gap verbatim, INCLUDING its knife edge — when
    // max_gap is an exact multiple of bin_length, adjacent centres differ by
    // max_gap to within an ULP and either side can land above or below. The
    // reference is exactly as sensitive, so a tolerance here would not restore
    // agreement, it would only make this half insensitive; see the port report.
    if (centres[i + 1] - centres[i] > max_gap) {
      stops.push_back(i + 1);
      starts.push_back(i + 1);
    }
  }
  stops.push_back(centres.size());

  std::size_t best = 0;
  double best_span = -std::numeric_limits<double>::infinity();
  for (std::size_t r = 0; r < starts.size(); ++r) {
    const double span = centres[stops[r] - 1] - centres[starts[r]];
    // Ties take the FIRST maximum, like np.argmax.
    if (span > best_span) {
      best_span = span;
      best = r;
    }
  }
  return {starts[best], stops[best]};
}

}  // namespace lane_detail

LaneFit fit_lane(const float * xyz, std::size_t count, int stride, const LaneOptions & opts)
{
  const std::vector<double> xy = as_xy(xyz, count, stride);
  const std::size_t n = xy.size() / 2;
  if (n < static_cast<std::size_t>(3 * opts.min_points_per_bin)) {
    return empty_fit("too_few_points");
  }

  Eigen::Vector2d mean;
  const Eigen::Vector2d axis = principal_axis(xy, mean);
  double lo = std::numeric_limits<double>::infinity();
  double hi = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < n; ++i) {
    const double along =
      (xy[2 * i] - mean.x()) * axis.x() + (xy[2 * i + 1] - mean.y()) * axis.y();
    lo = std::min(lo, along);
    hi = std::max(hi, along);
  }
  std::vector<double> curve{
    mean.x() + lo * axis.x(), mean.y() + lo * axis.y(),
    mean.x() + hi * axis.x(), mean.y() + hi * axis.y()};
  curve = lane_detail::resample(curve);
  curve = refine_centerline(
    std::move(curve), xy, opts.bin_length, opts.centerline_degree,
    opts.robust_iterations, opts.min_points_per_bin, opts.min_bin_fill);

  Bins bins = occupied_bins(
    curve, xy, opts.bin_length, opts.min_points_per_bin, opts.min_bin_fill);
  if (!bins.ok) {return empty_fit("bad_bins");}
  if (bins.centres.size() < 3) {return empty_fit("gap");}

  std::vector<double> lo_edge(bins.groups.size());
  std::vector<double> hi_edge(bins.groups.size());
  for (std::size_t b = 0; b < bins.groups.size(); ++b) {
    // numpy's DEFAULT 'linear' percentile method; nearest-rank drifts the
    // boundaries past the 0.15 m tolerance the tests hold this to.
    lo_edge[b] = percentile_linear(bins.groups[b], opts.edge_percentile);
    hi_edge[b] = percentile_linear(bins.groups[b], 100.0 - opts.edge_percentile);
  }

  // Never let a spline invent road across an empty stretch: cut at the gap and
  // keep the longer contiguous run instead.
  const auto run = lane_detail::longest_run(bins.centres, opts.max_gap);
  const std::vector<double> bin_s(
    bins.centres.begin() + static_cast<std::ptrdiff_t>(run.first),
    bins.centres.begin() + static_cast<std::ptrdiff_t>(run.second));
  const std::vector<double> lo_run(
    lo_edge.begin() + static_cast<std::ptrdiff_t>(run.first),
    lo_edge.begin() + static_cast<std::ptrdiff_t>(run.second));
  const std::vector<double> hi_run(
    hi_edge.begin() + static_cast<std::ptrdiff_t>(run.first),
    hi_edge.begin() + static_cast<std::ptrdiff_t>(run.second));
  if (bin_s.size() < 3) {return empty_fit("gap");}

  std::vector<double> samples(static_cast<std::size_t>(std::max(0, opts.n_samples)));
  linspace(bin_s.front(), bin_s.back(), samples.size(), samples.data());
  // LEFT takes the HIGH percentile because d is positive to the left.
  const std::vector<double> left_d =
    smooth(bin_s, hi_run, samples, opts.smoothing_per_point);
  const std::vector<double> right_d =
    smooth(bin_s, lo_run, samples, opts.smoothing_per_point);

  std::vector<double> tangent;
  std::vector<double> arclength;
  tangents(curve, tangent, arclength);
  const std::size_t k = curve.size() / 2;
  std::vector<double> normal(2 * k);
  for (std::size_t i = 0; i < k; ++i) {
    normal[2 * i] = -tangent[2 * i + 1];
    normal[2 * i + 1] = tangent[2 * i];
  }

  LaneFit fit;
  fit.centerline.assign(2 * samples.size(), 0.0);
  fit.left.assign(2 * samples.size(), 0.0);
  fit.right.assign(2 * samples.size(), 0.0);
  std::vector<double> column(k);
  std::vector<double> centre_axis(samples.size());
  std::vector<double> normal_xy(2 * samples.size());
  for (int axis_i = 0; axis_i < 2; ++axis_i) {
    const std::size_t a = static_cast<std::size_t>(axis_i);
    for (std::size_t i = 0; i < k; ++i) {column[i] = curve[2 * i + a];}
    // interp CLAMPS outside [arclength.front(), arclength.back()], and that is
    // the defined behaviour here: `samples` lives in the projection s-frame and
    // can fall outside the curve's own arc length range.
    interp(
      arclength.data(), column.data(), k, samples.data(), samples.size(), centre_axis.data());
    for (std::size_t i = 0; i < samples.size(); ++i) {fit.centerline[2 * i + a] = centre_axis[i];}
    for (std::size_t i = 0; i < k; ++i) {column[i] = normal[2 * i + a];}
    interp(
      arclength.data(), column.data(), k, samples.data(), samples.size(), centre_axis.data());
    for (std::size_t i = 0; i < samples.size(); ++i) {normal_xy[2 * i + a] = centre_axis[i];}
  }

  fit.width.assign(samples.size(), 0.0);
  for (std::size_t i = 0; i < samples.size(); ++i) {
    // Interpolating two unit normals shortens the result, so renormalise with
    // the same <1e-12 -> 1.0 guard the tangents use.
    double norm = std::hypot(normal_xy[2 * i], normal_xy[2 * i + 1]);
    if (norm < 1e-12) {norm = 1.0;}
    const double nx = normal_xy[2 * i] / norm;
    const double ny = normal_xy[2 * i + 1] / norm;
    fit.left[2 * i] = fit.centerline[2 * i] + left_d[i] * nx;
    fit.left[2 * i + 1] = fit.centerline[2 * i + 1] + left_d[i] * ny;
    fit.right[2 * i] = fit.centerline[2 * i] + right_d[i] * nx;
    fit.right[2 * i + 1] = fit.centerline[2 * i + 1] + right_d[i] * ny;
    // A DIFFERENCE OF LATERAL OFFSETS, not the Euclidean distance between the
    // two polylines: the offsets are what was measured, the polylines are a
    // rendering of them.
    fit.width[i] = left_d[i] - right_d[i];
  }
  fit.arclength = samples;

  bool crossed = false;
  bool out_of_range = false;
  for (double w : fit.width) {
    if (w <= 0.0) {crossed = true;}
    if (w < opts.min_width || w > opts.max_width) {out_of_range = true;}
  }
  if (crossed) {
    fit.reason = "crossed";
  } else if (out_of_range) {
    fit.reason = "width_out_of_range";
  }
  fit.valid = fit.reason.empty();
  return fit;
}

}  // namespace ransac_seg_cpp
