// The exact Euclidean transform is pinned against a brute-force O(n^2) search
// on random masks. Exact equality, not a tolerance: both sides end at
// sqrt(integer) * resolution, so any disagreement means the lower envelope is
// wrong rather than merely rounded differently.

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "ransac_seg_cpp/edt.hpp"

namespace
{

using ransac_seg_cpp::distance_transform_edt;

std::vector<double> brute_force(
  const std::vector<uint8_t> & occupied, int nx, int ny, double resolution)
{
  std::vector<double> out(static_cast<std::size_t>(nx) * ny,
    std::numeric_limits<double>::infinity());
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      int64_t best = -1;
      for (int jx = 0; jx < nx; ++jx) {
        for (int jy = 0; jy < ny; ++jy) {
          if (!occupied[static_cast<std::size_t>(jx) * ny + jy]) {
            continue;
          }
          const int64_t dx = ix - jx;
          const int64_t dy = iy - jy;
          const int64_t d2 = dx * dx + dy * dy;
          if (best < 0 || d2 < best) {
            best = d2;
          }
        }
      }
      if (best >= 0) {
        out[static_cast<std::size_t>(ix) * ny + iy] =
          std::sqrt(static_cast<double>(best)) * resolution;
      }
    }
  }
  return out;
}

TEST(Edt, MatchesBruteForceOnRandomMasks)
{
  std::mt19937 rng(1234);
  const double resolution = 0.37;
  for (int trial = 0; trial < 40; ++trial) {
    std::uniform_int_distribution<int> dim(1, 23);
    const int nx = dim(rng);
    const int ny = dim(rng);
    std::vector<uint8_t> occupied(static_cast<std::size_t>(nx) * ny, 0);
    std::bernoulli_distribution hit(0.02 + 0.2 * (trial % 5));
    bool any = false;
    for (auto & cell : occupied) {
      cell = hit(rng) ? 1 : 0;
      any = any || cell != 0;
    }
    if (!any) {
      occupied[0] = 1;
    }

    std::vector<double> got(occupied.size());
    distance_transform_edt(occupied.data(), nx, ny, resolution, got.data());
    const std::vector<double> want = brute_force(occupied, nx, ny, resolution);

    for (std::size_t i = 0; i < got.size(); ++i) {
      ASSERT_DOUBLE_EQ(got[i], want[i]) << "trial " << trial << " cell " << i
                                        << " (" << nx << "x" << ny << ")";
    }
  }
}

TEST(Edt, MatchesBruteForceOnThinAndSingleCellGrids)
{
  const double resolution = 1.0;
  const std::vector<std::pair<int, int>> shapes = {{1, 1}, {1, 17}, {17, 1}, {2, 9}};
  for (const auto & shape : shapes) {
    const int nx = shape.first;
    const int ny = shape.second;
    std::vector<uint8_t> occupied(static_cast<std::size_t>(nx) * ny, 0);
    occupied[occupied.size() / 2] = 1;

    std::vector<double> got(occupied.size());
    distance_transform_edt(occupied.data(), nx, ny, resolution, got.data());
    const std::vector<double> want = brute_force(occupied, nx, ny, resolution);
    for (std::size_t i = 0; i < got.size(); ++i) {
      ASSERT_DOUBLE_EQ(got[i], want[i]) << nx << "x" << ny << " cell " << i;
    }
  }
}

TEST(Edt, OccupiedCellsAreZero)
{
  const int nx = 6, ny = 5;
  std::vector<uint8_t> occupied(static_cast<std::size_t>(nx) * ny, 0);
  occupied[0] = 1;
  occupied[static_cast<std::size_t>(3) * ny + 2] = 1;

  std::vector<double> got(occupied.size());
  distance_transform_edt(occupied.data(), nx, ny, 0.5, got.data());

  EXPECT_DOUBLE_EQ(got[0], 0.0);
  EXPECT_DOUBLE_EQ(got[static_cast<std::size_t>(3) * ny + 2], 0.0);
}

TEST(Edt, ScalesWithResolution)
{
  // Diagonal neighbour of the single source: sqrt(2) cells.
  const int nx = 3, ny = 3;
  std::vector<uint8_t> occupied(9, 0);
  occupied[static_cast<std::size_t>(1) * ny + 1] = 1;

  std::vector<double> got(9);
  distance_transform_edt(occupied.data(), nx, ny, 0.25, got.data());

  EXPECT_DOUBLE_EQ(got[0], std::sqrt(2.0) * 0.25);
  EXPECT_DOUBLE_EQ(got[static_cast<std::size_t>(0) * ny + 1], 0.25);
}

TEST(Edt, EmptyMaskIsInfiniteEverywhere)
{
  const int nx = 4, ny = 7;
  const std::vector<uint8_t> occupied(static_cast<std::size_t>(nx) * ny, 0);
  std::vector<double> got(occupied.size());
  distance_transform_edt(occupied.data(), nx, ny, 0.5, got.data());
  for (double value : got) {
    EXPECT_TRUE(std::isinf(value));
    EXPECT_GT(value, 0.0);
  }
}

}  // namespace
