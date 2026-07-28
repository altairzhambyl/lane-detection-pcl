// numpy semantics the rest of the port depends on, in one place.
//
// The Python reference leans on numpy's *specific* definitions, not on "some
// reasonable percentile" or "some reasonable rounding". Every track that needs
// one of these would otherwise reimplement it, and the plausible-looking
// variants each move the result: nearest-rank percentile drifts the lane
// boundaries past their 0.15 m tolerance, std::round instead of half-to-even
// shifts _resample's vertex count by one (moving every downstream
// nearest-vertex projection), and second-order gradient ends move the end
// normals and therefore the outermost boundary points.
//
// Pointer-and-stride signatures rather than containers because the callers hold
// interleaved (N, 2) curves, which is what np.gradient(curve, axis=0) operates
// on column by column.

#ifndef RANSAC_SEG_CPP__NUMERIC_HPP_
#define RANSAC_SEG_CPP__NUMERIC_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ransac_seg_cpp
{

/// np.percentile with the default 'linear' method; `q` is in [0, 100].
///
/// Takes `values` by value because it has to sort; pass an rvalue to move.
/// Throws std::invalid_argument on an empty sample or a `q` outside [0, 100],
/// matching numpy raising rather than inventing a value.
double percentile_linear(std::vector<double> values, double q);

/// np.median: the mean of the two middle order statistics on an even sample.
///
/// Throws std::invalid_argument on an empty sample.
double median(std::vector<double> values);

/// Python's round() on a float: ties to even, so 2.5 -> 2 and 3.5 -> 4.
///
/// Implemented without std::nearbyint so it does not depend on the process's
/// current FP rounding mode. Non-finite input is returned unchanged.
double banker_round(double v);

/// np.interp: piecewise linear over ascending `xp`, clamped to the end values.
///
/// The clamping outside [xp[0], xp[n-1]] is defined behaviour fit_lane relies
/// on when its sample range overshoots the fitted arc length, not an accident.
/// Throws std::invalid_argument when `n` is 0.
void interp(
  const double * xp, const double * fp, std::size_t n,
  const double * x, std::size_t m, double * out);

/// np.gradient with edge_order=1 over a strided column.
///
/// Central difference in the interior, first-order one-sided at both ends.
/// Unit spacing, like the Python call site. Throws std::invalid_argument when
/// `n` < 2, which is where numpy also refuses.
void gradient(
  const double * v, std::size_t n, std::size_t stride,
  double * out, std::size_t out_stride);

/// np.linspace(a, b, n): `out[n - 1]` is exactly `b` for n >= 2.
void linspace(double a, double b, std::size_t n, double * out);

/// np.argsort(keys, kind="stable"): indices that sort `keys`, ties in order.
std::vector<std::size_t> stable_argsort(const std::vector<int64_t> & keys);

/// Widest lattice index floor_index will return.
///
/// 2^61, not int64's own limit: every lattice user then subtracts two indices
/// (`hi - lo`, `k - lo`) or steps a neighbour (`key.x + 1`), and those stay
/// representable only if the endpoints leave headroom. At 2^61 the widest
/// difference is 2^62 and a +/-1 step cannot wrap.
inline constexpr int64_t kMaxLatticeIndex = INT64_C(1) << 61;

/// std::floor(v) as an integer lattice index, saturating instead of trapping.
///
/// The Python reference indexes voxel and bin lattices with numpy int64 built
/// from arbitrary-precision Python ints, so a stray return at 1e25 m gives it a
/// merely absurd index. The same `static_cast<int64_t>(std::floor(...))` in C++
/// is undefined when the value does not fit, and at -O2 it produced INT64_MIN,
/// which then signed-overflowed every span and ravel computation downstream —
/// including the guards written to catch exactly that. Saturating keeps the
/// index absurd (so the guards fire) and keeps the arithmetic defined.
///
/// +/-infinity saturate to the corresponding limit; NaN goes to -limit, which
/// places it outside every real lattice rather than at the origin, where it
/// would silently join the data.
int64_t floor_index(double v);

}  // namespace ransac_seg_cpp

#endif  // RANSAC_SEG_CPP__NUMERIC_HPP_
