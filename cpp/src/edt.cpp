#include "ransac_seg_cpp/edt.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace ransac_seg_cpp
{

namespace
{

// A finite stand-in for "no source seen yet". The lower-envelope intersection
// divides one parabola height by another, and a genuine infinity turns that
// into inf-inf = NaN, which silently corrupts the envelope. Anything larger
// than the biggest reachable squared distance works, and every cell that keeps
// this value is overwritten with +inf by the caller-visible empty-mask path.
constexpr double kUnreached = 1e30;

/// One-dimensional squared-distance transform of `f` (Felzenszwalb & Huttenlocher).
///
/// `d[q] = min_p (q - p)^2 + f[p]`. `v` and `z` are scratch buffers of length n
/// and n + 1, hoisted by the caller so the 2D pass allocates once per image
/// rather than once per row.
void lower_envelope(
  const double * f, int n, double * d, int * v, double * z)
{
  int k = 0;
  v[0] = 0;
  z[0] = -std::numeric_limits<double>::infinity();
  z[1] = std::numeric_limits<double>::infinity();

  for (int q = 1; q < n; ++q) {
    double s = ((f[q] + double(q) * q) - (f[v[k]] + double(v[k]) * v[k])) /
      (2.0 * q - 2.0 * v[k]);
    while (s <= z[k]) {
      --k;
      s = ((f[q] + double(q) * q) - (f[v[k]] + double(v[k]) * v[k])) /
        (2.0 * q - 2.0 * v[k]);
    }
    ++k;
    v[k] = q;
    z[k] = s;
    z[k + 1] = std::numeric_limits<double>::infinity();
  }

  k = 0;
  for (int q = 0; q < n; ++q) {
    while (z[k + 1] < q) {
      ++k;
    }
    const double delta = double(q) - v[k];
    d[q] = delta * delta + f[v[k]];
  }
}

}  // namespace

void distance_transform_edt(
  const uint8_t * occupied, int nx, int ny, double resolution, double * out)
{
  if (nx <= 0 || ny <= 0) {
    return;
  }
  const std::size_t cells = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);

  bool any = false;
  for (std::size_t i = 0; i < cells; ++i) {
    if (occupied[i] != 0) {
      any = true;
      break;
    }
  }
  if (!any) {
    for (std::size_t i = 0; i < cells; ++i) {
      out[i] = std::numeric_limits<double>::infinity();
    }
    return;
  }

  const int longest = nx > ny ? nx : ny;
  std::vector<double> f(static_cast<std::size_t>(longest));
  std::vector<double> d(static_cast<std::size_t>(longest));
  std::vector<int> v(static_cast<std::size_t>(longest));
  std::vector<double> z(static_cast<std::size_t>(longest) + 1);

  // Pass 1: down each column (iy varies, contiguous in this x-major layout).
  for (int ix = 0; ix < nx; ++ix) {
    double * column = out + static_cast<std::size_t>(ix) * ny;
    const uint8_t * mask = occupied + static_cast<std::size_t>(ix) * ny;
    for (int iy = 0; iy < ny; ++iy) {
      f[static_cast<std::size_t>(iy)] = mask[iy] != 0 ? 0.0 : kUnreached;
    }
    lower_envelope(f.data(), ny, column, v.data(), z.data());
  }

  // Pass 2: across each row, seeded with pass 1's per-column squared distances.
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      f[static_cast<std::size_t>(ix)] = out[static_cast<std::size_t>(ix) * ny + iy];
    }
    lower_envelope(f.data(), nx, d.data(), v.data(), z.data());
    for (int ix = 0; ix < nx; ++ix) {
      out[static_cast<std::size_t>(ix) * ny + iy] = d[static_cast<std::size_t>(ix)];
    }
  }

  // Metres last: the sweeps above stay in exact integer cell^2 arithmetic.
  for (std::size_t i = 0; i < cells; ++i) {
    out[i] = std::sqrt(out[i]) * resolution;
  }
}

}  // namespace ransac_seg_cpp
