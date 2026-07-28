// These pin numpy's exact definitions, not "a reasonable percentile". Every
// expectation below was computed by hand from the numpy documentation and
// cross-checked against the Python reference the port has to agree with.

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "ransac_seg_cpp/numeric.hpp"

namespace
{

using ransac_seg_cpp::banker_round;
using ransac_seg_cpp::gradient;
using ransac_seg_cpp::interp;
using ransac_seg_cpp::linspace;
using ransac_seg_cpp::median;
using ransac_seg_cpp::percentile_linear;
using ransac_seg_cpp::stable_argsort;

TEST(Numeric, BankerRoundingTiesToEven)
{
  // Python's round(2.5) is 2 and round(3.5) is 4; std::round gives 3 and 4,
  // and that one-vertex difference moves every downstream projection.
  EXPECT_DOUBLE_EQ(banker_round(2.5), 2.0);
  EXPECT_DOUBLE_EQ(banker_round(3.5), 4.0);
  EXPECT_DOUBLE_EQ(banker_round(0.5), 0.0);
  EXPECT_DOUBLE_EQ(banker_round(1.5), 2.0);
  EXPECT_DOUBLE_EQ(banker_round(-2.5), -2.0);
  EXPECT_DOUBLE_EQ(banker_round(-3.5), -4.0);
  EXPECT_DOUBLE_EQ(banker_round(2.4), 2.0);
  EXPECT_DOUBLE_EQ(banker_round(2.6), 3.0);
  EXPECT_DOUBLE_EQ(banker_round(-0.4), 0.0);
}

TEST(Numeric, MedianOfEvenSampleAveragesTheMiddlePair)
{
  EXPECT_DOUBLE_EQ(median({1.0, 2.0, 3.0, 4.0}), 2.5);
  EXPECT_DOUBLE_EQ(median({4.0, 1.0, 3.0, 2.0}), 2.5);  // unsorted input
  EXPECT_DOUBLE_EQ(median({1.0, 2.0, 3.0}), 2.0);
  EXPECT_DOUBLE_EQ(median({7.0}), 7.0);
  EXPECT_THROW(median(std::vector<double>{}), std::invalid_argument);
}

TEST(Numeric, PercentileUsesNumpysLinearMethod)
{
  // [1, 2, 3, 4]: rank = 3*q/100.
  //   q=50  -> rank 1.5 -> 2 + 0.5*(3-2) = 2.5
  //   q=25  -> rank 0.75 -> 1 + 0.75*(2-1) = 1.75
  //   q=2   -> rank 0.06 -> 1 + 0.06*(2-1) = 1.06
  const std::vector<double> v{1.0, 2.0, 3.0, 4.0};
  EXPECT_DOUBLE_EQ(percentile_linear(v, 50.0), 2.5);
  EXPECT_DOUBLE_EQ(percentile_linear(v, 25.0), 1.75);
  EXPECT_NEAR(percentile_linear(v, 2.0), 1.06, 1e-12);

  // Nearest-rank would return an actual sample here; linear must not.
  EXPECT_NE(percentile_linear(v, 2.0), 1.0);

  EXPECT_DOUBLE_EQ(percentile_linear(v, 0.0), 1.0);
  EXPECT_DOUBLE_EQ(percentile_linear(v, 100.0), 4.0);
  EXPECT_DOUBLE_EQ(percentile_linear({5.0}, 42.0), 5.0);
  EXPECT_THROW(percentile_linear(std::vector<double>{}, 50.0), std::invalid_argument);
  EXPECT_THROW(percentile_linear({1.0}, 101.0), std::invalid_argument);
}

TEST(Numeric, InterpIsLinearInsideAndClampedOutside)
{
  const double xp[] = {0.0, 1.0, 2.0};
  const double fp[] = {10.0, 20.0, 40.0};
  const double x[] = {-5.0, 0.0, 0.5, 1.0, 1.25, 2.0, 7.0};
  double out[7] = {};
  interp(xp, fp, 3, x, 7, out);

  EXPECT_DOUBLE_EQ(out[0], 10.0);  // clamped to fp.front()
  EXPECT_DOUBLE_EQ(out[1], 10.0);
  EXPECT_DOUBLE_EQ(out[2], 15.0);
  EXPECT_DOUBLE_EQ(out[3], 20.0);
  EXPECT_DOUBLE_EQ(out[4], 25.0);
  EXPECT_DOUBLE_EQ(out[5], 40.0);
  EXPECT_DOUBLE_EQ(out[6], 40.0);  // clamped to fp.back()

  // A single sample point is a constant everywhere, as np.interp gives.
  const double one_xp[] = {3.0};
  const double one_fp[] = {9.0};
  double single[2] = {};
  const double probes[] = {-1.0, 100.0};
  interp(one_xp, one_fp, 1, probes, 2, single);
  EXPECT_DOUBLE_EQ(single[0], 9.0);
  EXPECT_DOUBLE_EQ(single[1], 9.0);

  EXPECT_THROW(interp(xp, fp, 0, x, 1, out), std::invalid_argument);
}

TEST(Numeric, GradientIsCentralInsideAndFirstOrderAtTheEnds)
{
  // Second-order ends would give 0.5*(-3*v0 + 4*v1 - v2) at the front, which
  // for this sample is 0.5*(-3 + 16 - 9) = 2.0 rather than 3.0.
  const double v[] = {1.0, 4.0, 9.0, 16.0};
  double out[4] = {};
  gradient(v, 4, 1, out, 1);
  EXPECT_DOUBLE_EQ(out[0], 3.0);
  EXPECT_DOUBLE_EQ(out[1], 4.0);
  EXPECT_DOUBLE_EQ(out[2], 6.0);
  EXPECT_DOUBLE_EQ(out[3], 7.0);

  // Strided: the y column of an interleaved (N, 2) curve.
  const double curve[] = {0.0, 1.0, 0.0, 3.0, 0.0, 6.0};
  double dy[3] = {};
  gradient(curve + 1, 3, 2, dy, 1);
  EXPECT_DOUBLE_EQ(dy[0], 2.0);
  EXPECT_DOUBLE_EQ(dy[1], 2.5);
  EXPECT_DOUBLE_EQ(dy[2], 3.0);

  EXPECT_THROW(gradient(v, 1, 1, out, 1), std::invalid_argument);
}

TEST(Numeric, LinspaceHitsBothEndpointsExactly)
{
  double out[5] = {};
  linspace(0.0, 1.0, 5, out);
  EXPECT_DOUBLE_EQ(out[0], 0.0);
  EXPECT_DOUBLE_EQ(out[4], 1.0);
  EXPECT_DOUBLE_EQ(out[2], 0.5);

  // 0.1 steps do not sum to 1.0 in binary; the endpoint must still be exact.
  double ten[11] = {};
  linspace(0.0, 1.0, 11, ten);
  EXPECT_EQ(ten[10], 1.0);

  double one[1] = {};
  linspace(2.0, 9.0, 1, one);
  EXPECT_DOUBLE_EQ(one[0], 2.0);
}

TEST(Numeric, ArgsortIsStable)
{
  const std::vector<int64_t> keys{2, 1, 2, 1, 0};
  const std::vector<std::size_t> expected{4, 1, 3, 0, 2};
  EXPECT_EQ(stable_argsort(keys), expected);
  EXPECT_TRUE(stable_argsort(std::vector<int64_t>{}).empty());
}

}  // namespace
