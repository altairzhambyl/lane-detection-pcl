// Curve fitting primitives for the lane fit: numpy's weighted polyfit and a
// stand-in for scipy's UnivariateSpline.
//
// DELIBERATE, DOCUMENTED DEVIATION — SmoothingSpline is not a port of
// scipy.interpolate.UnivariateSpline. UnivariateSpline is FITPACK's curfit: a
// least-squares B-spline of degree k whose INTERIOR KNOTS ARE CHOSEN
// AUTOMATICALLY, inserted iteratively where the residuals are largest until
// sum((y_i - spl(x_i))^2) <= s. There is no C++ equivalent in the toolchain and
// a faithful fpcurf translation is roughly 700 lines of Fortran-derived
// numerics, so this class keeps the property ransac_seg/lane.py's docstring
// says is load-bearing and replaces the mechanism that is not.
//
// KEPT: `s` is a TOTAL squared-residual budget scaled by the bin count
// (smoothing_per_point * n_bins). That is the whole point of _smooth — a
// hardcoded budget means a different amount of smoothing per sample as range
// sparsity changes how many bins the sweep fills, so the same road would come
// back with different geometry from a sparser scan.
//
// REPLACED: the knot-placement mechanism. Degree k = clamp(n - 1, 1, 3), same
// as the Python. The basis sits on a UNIFORM clamped knot vector over
// [x.front(), x.back()] with clamp(n - k - 1, 0, 8) interior knots, and the fit
// is penalised least squares with a second-order difference penalty on the
// coefficients whose weight lambda is found by bisection on log(lambda) so that
// RSS(lambda) == s. The two limits bracket FITPACK's own range: lambda -> 0
// approaches interpolation, and the degree-k least-squares polynomial over the
// interval — FITPACK's own first step, the answer curfit returns when the
// polynomial already satisfies the budget — is tested for explicitly before any
// bisection. If RSS(polynomial) <= s that polynomial IS the answer; if RSS at
// lambda -> 0 still exceeds s, the lambda -> 0 fit is returned.
//
// (The second-order penalty's own lambda -> infinity limit is the least-squares
// STRAIGHT LINE, not the degree-k polynomial, because the penalty's null space
// is the coefficients linear in index. That is why the polynomial limit is
// handled by an explicit test rather than by letting lambda run away: it makes
// the stated FITPACK behaviour exact instead of approximate, and it also
// guarantees the bisection is bracketed — RSS is monotone in lambda, and the
// bisection only runs when RSS(poly) > s >= ... i.e. when a root exists.)
//
// The deviation is measured, not asserted: cpp/tools/dump_golden.py dumps
// scipy's answers on every synthetic road in test/test_lane.py and
// test_spline.cpp reports the max absolute deviation against them.

#ifndef RANSAC_SEG_CPP__SPLINE_HPP_
#define RANSAC_SEG_CPP__SPLINE_HPP_

#include <cstddef>
#include <vector>

namespace ransac_seg_cpp
{

/// np.polyfit(x, y, degree, w=weights): coefficients highest power first.
///
/// The weight multiplies BOTH sides of each row, so this minimises
/// sum((w_i * (y_i - p(x_i)))^2) — numpy's convention, not a covariance weight.
/// x is centred and scaled before the Vandermonde is built and the coefficients
/// are unwound afterwards: numpy rescales the Vandermonde columns for the same
/// reason, and a naive fit on raw arc lengths of tens of metres at degree 2
/// loses precision the lane fit cannot afford.
///
/// Throws std::invalid_argument on mismatched lengths, a negative degree, or
/// fewer samples than coefficients.
std::vector<double> polyfit_weighted(
  const std::vector<double> & x, const std::vector<double> & y,
  const std::vector<double> & w, int degree);

/// np.polyval: evaluate coefficients given highest power first, by Horner.
double polyval(const std::vector<double> & coeffs, double x);

/// Highest B-spline degree SmoothingSpline accepts.
///
/// The reference caps its own degree the same way (`clip(n_bins - 1, 1, 3)` in
/// ransac_seg.lane._smooth), so nothing in the port needs a quartic. The cap is
/// enforced in the constructor rather than left to the caller because the
/// evaluator's scratch buffers are sized from it: `smooth()` clamping the
/// degree used to be the only thing standing between a k >= 4 construction and
/// a silent stack overwrite in basis_funs.
inline constexpr int kMaxDegree = 3;

/// Least-squares smoothing spline with a total squared-residual budget.
///
/// See the file header for what this does and does not share with
/// scipy.interpolate.UnivariateSpline. Requires strictly increasing `x`,
/// 1 <= k <= kMaxDegree and n > k — all guaranteed by the distinct ascending
/// bin centres of the caller; violations throw std::invalid_argument rather
/// than returning nonsense or corrupting the stack.
class SmoothingSpline
{
public:
  SmoothingSpline(
    const std::vector<double> & x, const std::vector<double> & y,
    double s, int k);

  /// Evaluate at `q`. Outside [x.front(), x.back()] the query is clamped to the
  /// interval; fit_lane never queries outside it, so this is a guard rather
  /// than a modelled extrapolation.
  double operator()(double q) const;

  /// sum((y_i - f(x_i))^2) of the returned fit, for the tests and the report.
  double residual_sum_squares() const {return rss_;}

  /// True when the budget was met by the degree-k polynomial alone, so no
  /// interior knot was ever used (FITPACK's first step).
  bool is_polynomial() const {return poly_mode_;}

  /// The penalty weight the bisection settled on; 0 in polynomial mode.
  double lambda() const {return lambda_;}

private:
  double eval_spline(double t) const;

  bool poly_mode_ = false;
  double lambda_ = 0.0;
  double rss_ = 0.0;
  int degree_ = 3;
  double a_ = 0.0;
  double b_ = 0.0;
  std::vector<double> poly_;    ///< highest power first, polynomial mode only
  std::vector<double> knots_;   ///< clamped, non-decreasing
  std::vector<double> coeffs_;  ///< B-spline control coefficients
};

/// ransac_seg.lane._smooth: budget = smoothing_per_point * n_bins,
/// k = clamp(n_bins - 1, 1, 3), evaluated at every `s_query`.
std::vector<double> smooth(
  const std::vector<double> & s_bins, const std::vector<double> & values,
  const std::vector<double> & s_query, double smoothing_per_point);

}  // namespace ransac_seg_cpp

#endif  // RANSAC_SEG_CPP__SPLINE_HPP_
