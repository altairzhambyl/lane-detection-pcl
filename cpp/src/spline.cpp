#include "ransac_seg_cpp/spline.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>

namespace ransac_seg_cpp
{

namespace
{

/// Largest interior-knot count worth spending on 20-odd bins; see the header.
constexpr int kMaxInteriorKnots = 8;
/// Penalty weights are searched relative to the data term's own scale, so the
/// bracket is dimensionless and does not care whether s is metres or metres^2.
constexpr double kLambdaLoFactor = 1e-12;
constexpr double kLambdaHiFactor = 1e12;
constexpr int kBisectionSteps = 100;

/// NURBS-book A2.1: index of the knot span containing `t`, clamped to [k, m-1].
int find_span(const std::vector<double> & knots, int m, int k, double t)
{
  if (t >= knots[static_cast<std::size_t>(m)]) {return m - 1;}
  if (t <= knots[static_cast<std::size_t>(k)]) {return k;}
  int lo = k;
  int hi = m;
  int mid = (lo + hi) / 2;
  while (t < knots[static_cast<std::size_t>(mid)] ||
    t >= knots[static_cast<std::size_t>(mid) + 1])
  {
    if (t < knots[static_cast<std::size_t>(mid)]) {hi = mid;} else {lo = mid;}
    mid = (lo + hi) / 2;
  }
  return mid;
}

/// NURBS-book A2.2: the k+1 non-zero basis functions on `span` at `t`.
///
/// `out` and the two scratch arrays hold kMaxDegree + 1 entries, which the
/// constructor's degree check guarantees is enough. They are fixed-size on
/// purpose: eval_spline calls this once per query point.
void basis_funs(const std::vector<double> & knots, int span, int k, double t, double * out)
{
  double left[kMaxDegree + 1] = {};
  double right[kMaxDegree + 1] = {};
  out[0] = 1.0;
  for (int j = 1; j <= k; ++j) {
    left[j] = t - knots[static_cast<std::size_t>(span + 1 - j)];
    right[j] = knots[static_cast<std::size_t>(span + j)] - t;
    double saved = 0.0;
    for (int r = 0; r < j; ++r) {
      const double temp = out[r] / (right[r + 1] + left[j - r]);
      out[r] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    out[j] = saved;
  }
}

}  // namespace

std::vector<double> polyfit_weighted(
  const std::vector<double> & x, const std::vector<double> & y,
  const std::vector<double> & w, int degree)
{
  if (degree < 0) {
    throw std::invalid_argument("polyfit_weighted: degree must be non-negative");
  }
  if (x.size() != y.size() || x.size() != w.size()) {
    throw std::invalid_argument("polyfit_weighted: x, y and w must be the same length");
  }
  const int n_coeff = degree + 1;
  const std::size_t n = x.size();
  if (n < static_cast<std::size_t>(n_coeff)) {
    throw std::invalid_argument("polyfit_weighted: fewer samples than coefficients");
  }

  // Centre and scale before building the Vandermonde. On raw arc lengths of a
  // few tens of metres the degree-2 columns span six orders of magnitude and
  // the fit loses digits the 0.15 m width tolerance cannot spare.
  double mean = 0.0;
  for (double v : x) {mean += v;}
  mean /= static_cast<double>(n);
  double scale = 0.0;
  for (double v : x) {scale = std::max(scale, std::abs(v - mean));}
  if (!(scale > 0.0)) {scale = 1.0;}

  Eigen::MatrixXd lhs(static_cast<Eigen::Index>(n), n_coeff);
  Eigen::VectorXd rhs(static_cast<Eigen::Index>(n));
  for (std::size_t i = 0; i < n; ++i) {
    const double u = (x[i] - mean) / scale;
    // numpy's weight multiplies the row on both sides, so it is a residual
    // weight, not a variance.
    double power = 1.0;
    for (int j = n_coeff - 1; j >= 0; --j) {
      lhs(static_cast<Eigen::Index>(i), j) = w[i] * power;
      power *= u;
    }
    rhs(static_cast<Eigen::Index>(i)) = w[i] * y[i];
  }

  const Eigen::VectorXd fit = lhs.colPivHouseholderQr().solve(rhs);

  // Unwind (x - mean)/scale back into the raw monomial basis numpy returns:
  // u^j = scale^-j * sum_i C(j, i) x^i (-mean)^(j-i).
  std::vector<double> raw(static_cast<std::size_t>(n_coeff), 0.0);  // raw[i] = coeff of x^i
  for (int j = 0; j < n_coeff; ++j) {
    const double a_j = fit(n_coeff - 1 - j) / std::pow(scale, j);
    double binom = 1.0;
    for (int i = j; i >= 0; --i) {
      raw[static_cast<std::size_t>(i)] += a_j * binom * std::pow(-mean, j - i);
      binom = binom * i / static_cast<double>(j - i + 1);
    }
  }

  std::vector<double> coeffs(static_cast<std::size_t>(n_coeff));
  for (int i = 0; i < n_coeff; ++i) {
    coeffs[static_cast<std::size_t>(i)] = raw[static_cast<std::size_t>(degree - i)];
  }
  return coeffs;
}

double polyval(const std::vector<double> & coeffs, double x)
{
  double acc = 0.0;
  for (double c : coeffs) {acc = acc * x + c;}
  return acc;
}

SmoothingSpline::SmoothingSpline(
  const std::vector<double> & x, const std::vector<double> & y, double s, int k)
{
  const std::size_t n = x.size();
  if (n != y.size()) {
    throw std::invalid_argument("SmoothingSpline: x and y must be the same length");
  }
  if (k < 1 || k > kMaxDegree) {
    throw std::invalid_argument(
            "SmoothingSpline: degree must be in [1, " + std::to_string(kMaxDegree) + "]");
  }
  if (n <= static_cast<std::size_t>(k)) {
    throw std::invalid_argument("SmoothingSpline: needs more samples than the degree");
  }
  for (std::size_t i = 1; i < n; ++i) {
    if (!(x[i] > x[i - 1])) {
      throw std::invalid_argument("SmoothingSpline: x must be strictly increasing");
    }
  }
  degree_ = k;
  a_ = x.front();
  b_ = x.back();

  // FITPACK's first step: the least-squares polynomial of degree k with no
  // interior knots. If it already meets the budget it is the answer, and curfit
  // returns exactly that.
  const std::vector<double> ones(n, 1.0);
  std::vector<double> poly = polyfit_weighted(x, y, ones, k);
  double poly_rss = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double r = y[i] - polyval(poly, x[i]);
    poly_rss += r * r;
  }

  const int n_interior =
    std::min(kMaxInteriorKnots, std::max(0, static_cast<int>(n) - k - 1));
  if (poly_rss <= s || n_interior == 0) {
    // With no interior knot the spline space IS the degree-k polynomials, so
    // the lambda -> 0 fit the spec asks for in the over-budget case is this one.
    poly_mode_ = true;
    poly_ = std::move(poly);
    rss_ = poly_rss;
    return;
  }

  const int m = n_interior + k + 1;  // number of control coefficients
  knots_.assign(static_cast<std::size_t>(m + k + 1), 0.0);
  for (int i = 0; i <= k; ++i) {
    knots_[static_cast<std::size_t>(i)] = a_;
    knots_[static_cast<std::size_t>(m + k - i)] = b_;
  }
  for (int i = 1; i <= n_interior; ++i) {
    knots_[static_cast<std::size_t>(k + i)] =
      a_ + (b_ - a_) * static_cast<double>(i) / static_cast<double>(n_interior + 1);
  }

  Eigen::MatrixXd basis = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), m);
  Eigen::VectorXd rhs(static_cast<Eigen::Index>(n));
  double nz[kMaxDegree + 1];
  for (std::size_t i = 0; i < n; ++i) {
    const int span = find_span(knots_, m, k, x[i]);
    basis_funs(knots_, span, k, x[i], nz);
    for (int j = 0; j <= k; ++j) {
      basis(static_cast<Eigen::Index>(i), span - k + j) = nz[j];
    }
    rhs(static_cast<Eigen::Index>(i)) = y[i];
  }

  // Second-order difference penalty: its null space is the coefficients linear
  // in index, i.e. straight lines, so RSS rises monotonically with lambda from
  // the near-interpolating fit towards the least-squares line.
  Eigen::MatrixXd diff = Eigen::MatrixXd::Zero(m - 2, m);
  for (int r = 0; r + 2 < m; ++r) {
    diff(r, r) = 1.0;
    diff(r, r + 1) = -2.0;
    diff(r, r + 2) = 1.0;
  }

  const Eigen::MatrixXd normal = basis.transpose() * basis;
  const Eigen::VectorXd projected = basis.transpose() * rhs;
  const Eigen::MatrixXd penalty = diff.transpose() * diff;
  const double scale = normal.trace() / std::max(penalty.trace(), 1e-300);

  const auto fit_at = [&](double lambda, Eigen::VectorXd & out_c) {
      const Eigen::MatrixXd system = normal + lambda * penalty;
      out_c = system.ldlt().solve(projected);
      const Eigen::VectorXd residual = rhs - basis * out_c;
      return residual.squaredNorm();
    };

  Eigen::VectorXd c_lo;
  const double lam_lo = kLambdaLoFactor * scale;
  const double rss_lo = fit_at(lam_lo, c_lo);
  if (rss_lo >= s) {
    // Even at the interpolating limit the budget is unreachable; the spec's
    // answer is the lambda -> 0 fit rather than an unbounded search.
    coeffs_.assign(c_lo.data(), c_lo.data() + m);
    lambda_ = lam_lo;
    rss_ = rss_lo;
    return;
  }

  Eigen::VectorXd c_hi;
  double lam_hi = kLambdaHiFactor * scale;
  const double rss_hi = fit_at(lam_hi, c_hi);
  if (rss_hi <= s) {
    coeffs_.assign(c_hi.data(), c_hi.data() + m);
    lambda_ = lam_hi;
    rss_ = rss_hi;
    return;
  }

  // Bisect on log(lambda): RSS spans orders of magnitude in lambda and a linear
  // bisection would spend every step in the stiff end of the bracket.
  double log_lo = std::log(lam_lo);
  double log_hi = std::log(lam_hi);
  Eigen::VectorXd c_mid = c_lo;
  double rss_mid = rss_lo;
  for (int step = 0; step < kBisectionSteps; ++step) {
    const double log_mid = 0.5 * (log_lo + log_hi);
    rss_mid = fit_at(std::exp(log_mid), c_mid);
    if (rss_mid > s) {log_hi = log_mid;} else {log_lo = log_mid;}
    if (std::abs(rss_mid - s) <= 1e-12 * std::max(1.0, std::abs(s))) {break;}
  }
  lambda_ = std::exp(0.5 * (log_lo + log_hi));
  rss_ = fit_at(lambda_, c_mid);
  coeffs_.assign(c_mid.data(), c_mid.data() + m);
}

double SmoothingSpline::eval_spline(double t) const
{
  const int m = static_cast<int>(coeffs_.size());
  const int span = find_span(knots_, m, degree_, t);
  double nz[kMaxDegree + 1];
  basis_funs(knots_, span, degree_, t, nz);
  double acc = 0.0;
  for (int j = 0; j <= degree_; ++j) {
    acc += nz[j] * coeffs_[static_cast<std::size_t>(span - degree_ + j)];
  }
  return acc;
}

double SmoothingSpline::operator()(double q) const
{
  if (poly_mode_) {return polyval(poly_, q);}
  return eval_spline(std::min(std::max(q, a_), b_));
}

std::vector<double> smooth(
  const std::vector<double> & s_bins, const std::vector<double> & values,
  const std::vector<double> & s_query, double smoothing_per_point)
{
  const std::size_t n_bins = s_bins.size();
  const double budget = smoothing_per_point * static_cast<double>(n_bins);
  const int degree =
    std::min(kMaxDegree, std::max(1, static_cast<int>(n_bins) - 1));
  const SmoothingSpline spline(s_bins, values, budget, degree);
  std::vector<double> out(s_query.size());
  for (std::size_t i = 0; i < s_query.size(); ++i) {out[i] = spline(s_query[i]);}
  return out;
}

}  // namespace ransac_seg_cpp
