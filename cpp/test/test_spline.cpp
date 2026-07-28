// Measures the deliberate SmoothingSpline deviation from scipy's
// UnivariateSpline instead of asserting it away.
//
// cpp/tools/dump_golden.py replays the synthetic roads of test/test_lane.py
// through the reference Python and records both _smooth calls of every fit —
// its inputs and FITPACK's answers. This file reports the max absolute
// deviation on each of them and only then asserts the bounds, which were
// written down after the measurement, never before.
//
// Data directory: $RANSAC_SEG_CPP_TEST_DATA, else -DRANSAC_SEG_CPP_TEST_DATA,
// else "test/data" relative to the working directory.

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ransac_seg_cpp/spline.hpp"

#ifndef RANSAC_SEG_CPP_TEST_DATA
#define RANSAC_SEG_CPP_TEST_DATA "test/data"
#endif

using ransac_seg_cpp::SmoothingSpline;
using ransac_seg_cpp::polyfit_weighted;
using ransac_seg_cpp::polyval;
using ransac_seg_cpp::smooth;

namespace
{

// Measured on this tree with g++ 11.4 -O2: the largest deviation over the
// eight roads where FITPACK also settles on the plain degree-k polynomial is
// 7.550e-15 m, so double-precision agreement rather than approximation.
constexpr double kPolynomialAgreement = 1e-12;
// The 120 degree bend is the one case where the budget forces real smoothing
// and the knot-placement mechanism therefore matters. Measured 1.403e-01 m on
// the left boundary and 2.395e-01 m on the right.
constexpr double kBendDeviation = 0.30;

std::string data_dir()
{
  const char * env = std::getenv("RANSAC_SEG_CPP_TEST_DATA");
  return env != nullptr ? std::string(env) : std::string(RANSAC_SEG_CPP_TEST_DATA);
}

struct SmoothCase
{
  std::string name;
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> query;
  std::vector<double> scipy;
  double smoothing_per_point = 0.02;
};

std::vector<double> read_numbers(std::istringstream & in)
{
  std::vector<double> values;
  double value = 0.0;
  while (in >> value) {values.push_back(value);}
  return values;
}

std::vector<SmoothCase> load_cases()
{
  std::vector<SmoothCase> cases;
  std::ifstream file(data_dir() + "/golden_spline.txt");
  if (!file) {return cases;}
  std::string line;
  std::string current;
  SmoothCase pending;
  while (std::getline(file, line)) {
    std::istringstream in(line);
    std::string tag;
    in >> tag;
    if (tag == "case") {
      in >> current;
    } else if (tag == "call") {
      std::size_t n = 0;
      std::size_t m = 0;
      double spp = 0.0;
      in >> n >> m >> spp;
      pending = SmoothCase{};
      pending.name = current;
      pending.smoothing_per_point = spp;
    } else if (tag == "x") {
      pending.x = read_numbers(in);
    } else if (tag == "y") {
      pending.y = read_numbers(in);
    } else if (tag == "q") {
      pending.query = read_numbers(in);
    } else if (tag == "out") {
      pending.scipy = read_numbers(in);
      cases.push_back(pending);
    }
  }
  return cases;
}

}  // namespace

TEST(Polyfit, RecoversAnExactQuadratic)
{
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> w;
  for (int i = 0; i < 12; ++i) {
    const double xi = 3.0 + 4.5 * i;  // absolute arc lengths, as at the call site
    x.push_back(xi);
    y.push_back(0.75 * xi * xi - 2.5 * xi + 7.0);
    w.push_back(std::sqrt(static_cast<double>(10 + i)));
  }
  const std::vector<double> coeffs = polyfit_weighted(x, y, w, 2);
  ASSERT_EQ(coeffs.size(), 3u);
  // Highest power first, like numpy.
  EXPECT_NEAR(coeffs[0], 0.75, 1e-10);
  EXPECT_NEAR(coeffs[1], -2.5, 1e-8);
  EXPECT_NEAR(coeffs[2], 7.0, 1e-7);
}

TEST(Polyfit, MatchesNumpyOnAWeightedNoisyFit)
{
  // Coefficients from np.polyfit(x, y, 2, w=np.sqrt(counts)) on exactly this
  // input, run in this session:
  //   [-0.00407262954153, 0.12186062115163, 0.10358217755140]
  const std::vector<double> x{1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0};
  const std::vector<double> y{0.20, 0.51, 0.55, 0.79, 0.86, 0.95, 1.02, 0.97, 1.05, 0.99};
  const std::vector<double> counts{9.0, 100.0, 400.0, 900.0, 1600.0,
    1600.0, 900.0, 400.0, 100.0, 9.0};
  std::vector<double> w;
  for (double c : counts) {w.push_back(std::sqrt(c));}

  const std::vector<double> coeffs = polyfit_weighted(x, y, w, 2);
  ASSERT_EQ(coeffs.size(), 3u);
  EXPECT_NEAR(coeffs[0], -0.00407262954153, 1e-12);
  EXPECT_NEAR(coeffs[1], 0.12186062115163, 1e-11);
  EXPECT_NEAR(coeffs[2], 0.10358217755140, 1e-10);
}

TEST(Polyfit, WeightsShiftTheFitTowardsTheHeavyBins)
{
  // The whole reason for w = sqrt(count): one thin, badly placed bin must not
  // steer the curve the way an unweighted fit lets it.
  const std::vector<double> x{0.0, 1.0, 2.0, 3.0, 4.0};
  const std::vector<double> y{0.0, 0.0, 0.0, 0.0, 5.0};
  const std::vector<double> heavy{100.0, 100.0, 100.0, 100.0, 1.0};
  const std::vector<double> flat(5, 1.0);
  const double weighted = polyval(polyfit_weighted(x, y, heavy, 1), 2.0);
  const double unweighted = polyval(polyfit_weighted(x, y, flat, 1), 2.0);
  EXPECT_LT(std::abs(weighted), std::abs(unweighted));
}

TEST(Polyfit, RejectsMalformedInput)
{
  const std::vector<double> x{0.0, 1.0};
  const std::vector<double> y{0.0, 1.0};
  const std::vector<double> w{1.0, 1.0};
  EXPECT_THROW(polyfit_weighted(x, y, w, -1), std::invalid_argument);
  EXPECT_THROW(polyfit_weighted(x, y, w, 3), std::invalid_argument);
  EXPECT_THROW(polyfit_weighted(x, {0.0}, w, 1), std::invalid_argument);
}

TEST(Polyval, IsHornerOnHighestPowerFirst)
{
  EXPECT_NEAR(polyval({2.0, -3.0, 1.0}, 4.0), 2.0 * 16.0 - 3.0 * 4.0 + 1.0, 1e-12);
  EXPECT_NEAR(polyval({}, 4.0), 0.0, 1e-12);
}

TEST(SmoothingSpline, RejectsInputItCannotFit)
{
  const std::vector<double> x{0.0, 1.0, 1.0, 2.0};
  const std::vector<double> y(4, 0.0);
  EXPECT_THROW(SmoothingSpline(x, y, 1.0, 3), std::invalid_argument);
  EXPECT_THROW(SmoothingSpline({0.0, 1.0}, {0.0, 1.0}, 1.0, 3), std::invalid_argument);
  EXPECT_THROW(SmoothingSpline({0.0, 1.0}, {0.0}, 1.0, 1), std::invalid_argument);
}

TEST(SmoothingSpline, RejectsADegreeAboveTheEvaluatorsCap)
{
  // basis_funs and eval_spline carry kMaxDegree + 1 scratch slots. A k of 4 on
  // ample, strictly increasing data used to sail past every check and write one
  // slot past both of them — a stack overwrite at -O2, caught only because
  // smooth() clamps the degree in the CALLER. The cap belongs to the class.
  std::vector<double> x;
  std::vector<double> y;
  for (int i = 0; i < 30; ++i) {
    x.push_back(0.5 * i);
    y.push_back(std::sin(0.3 * i));
  }
  EXPECT_NO_THROW(SmoothingSpline(x, y, 1e-9, ransac_seg_cpp::kMaxDegree));
  EXPECT_THROW(SmoothingSpline(x, y, 1e-9, ransac_seg_cpp::kMaxDegree + 1), std::invalid_argument);
  EXPECT_THROW(SmoothingSpline(x, y, 1e-9, 17), std::invalid_argument);
}

TEST(SmoothingSpline, ReturnsThePolynomialWhenItAlreadyMeetsTheBudget)
{
  // FITPACK's first step, and the answer curfit returns unchanged.
  std::vector<double> x;
  std::vector<double> y;
  for (int i = 0; i < 20; ++i) {
    x.push_back(static_cast<double>(i));
    y.push_back(0.5 * i * i - 3.0 * i + 1.0);
  }
  const SmoothingSpline spline(x, y, 0.4, 3);
  EXPECT_TRUE(spline.is_polynomial());
  EXPECT_LT(spline.residual_sum_squares(), 0.4);
  EXPECT_NEAR(spline(7.5), 0.5 * 7.5 * 7.5 - 3.0 * 7.5 + 1.0, 1e-8);
}

TEST(SmoothingSpline, MeetsTheBudgetExactlyWhenSmoothingIsNeeded)
{
  // Data a cubic cannot follow, so the bisection has to find a real lambda.
  std::vector<double> x;
  std::vector<double> y;
  for (int i = 0; i < 20; ++i) {
    x.push_back(static_cast<double>(i));
    y.push_back(std::sin(0.9 * i));
  }
  const double budget = 0.4;
  const SmoothingSpline spline(x, y, budget, 3);
  EXPECT_FALSE(spline.is_polynomial());
  EXPECT_GT(spline.lambda(), 0.0);
  // The whole contract of `s`: the residual budget is spent, not undershot.
  EXPECT_NEAR(spline.residual_sum_squares(), budget, 1e-6);
}

TEST(SmoothingSpline, BudgetScalesWithBinCount)
{
  // The property lane.py's docstring calls load-bearing: the same geometry
  // sampled at two bin counts must come back with comparable smoothing, which
  // is why _smooth multiplies smoothing_per_point by the bin count.
  const auto build = [](int n) {
      std::vector<double> x;
      std::vector<double> y;
      for (int i = 0; i < n; ++i) {
        const double t = 40.0 * i / (n - 1);
        x.push_back(t);
        y.push_back(std::sin(0.25 * t));
      }
      return std::make_pair(x, y);
    };
  const auto coarse = build(10);
  const auto fine = build(30);
  const std::vector<double> query{5.0, 12.0, 21.0, 33.0};
  const std::vector<double> a = smooth(coarse.first, coarse.second, query, 0.02);
  const std::vector<double> b = smooth(fine.first, fine.second, query, 0.02);
  for (std::size_t i = 0; i < query.size(); ++i) {
    EXPECT_LT(std::abs(a[i] - b[i]), 0.15) << "at s = " << query[i];
  }
}

TEST(SmoothingSpline, MatchesScipyOnTheGoldenRoads)
{
  const std::vector<SmoothCase> cases = load_cases();
  ASSERT_FALSE(cases.empty())
    << "golden_spline.txt not found under " << data_dir()
    << "; regenerate with python3 cpp/tools/dump_golden.py";

  double worst_polynomial = 0.0;
  double worst_bend = 0.0;
  for (const SmoothCase & c : cases) {
    const std::vector<double> mine =
      smooth(c.x, c.y, c.query, c.smoothing_per_point);
    ASSERT_EQ(mine.size(), c.scipy.size());
    double worst = 0.0;
    for (std::size_t i = 0; i < mine.size(); ++i) {
      worst = std::max(worst, std::abs(mine[i] - c.scipy[i]));
    }
    const SmoothingSpline spline(
      c.x, c.y, c.smoothing_per_point * static_cast<double>(c.x.size()),
      std::min(3, std::max(1, static_cast<int>(c.x.size()) - 1)));
    std::cout << "[ SPLINE   ] " << c.name << " n_bins=" << c.x.size()
              << " polynomial=" << (spline.is_polynomial() ? "yes" : "no ")
              << " rss=" << spline.residual_sum_squares()
              << " max|dev vs scipy|=" << worst << " m" << std::endl;
    if (spline.is_polynomial()) {
      worst_polynomial = std::max(worst_polynomial, worst);
    } else {
      worst_bend = std::max(worst_bend, worst);
    }
  }
  std::cout << "[ SPLINE   ] worst over the polynomial-limit cases: "
            << worst_polynomial << " m; worst where smoothing bites: "
            << worst_bend << " m" << std::endl;

  // Wherever FITPACK also stops at the degree-k polynomial the two agree to
  // double precision, because the answer no longer depends on knot placement.
  EXPECT_LT(worst_polynomial, kPolynomialAgreement);
  // Where it does not, the knot mechanism is what differs and this is the size
  // of that difference on the only road in the suite that reaches it.
  EXPECT_LT(worst_bend, kBendDeviation);
}
