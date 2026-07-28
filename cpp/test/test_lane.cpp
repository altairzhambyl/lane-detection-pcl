// Port of test/test_lane.py, plus a differential check against the reference
// Python on the very same clouds.
//
// cpp/tools/dump_golden.py writes each synthetic road out as float32 (which is
// what fit_lane's signature takes anyway) together with the LaneFit the Python
// produced from it, so "the ports agree" is a measurement here rather than a
// claim. The scenario assertions below are the Python's own, tolerances
// included: every one of them is a case the min/max-per-x-column fitter this
// method replaced was measured to fail.
//
// Data directory: $RANSAC_SEG_CPP_TEST_DATA, else -DRANSAC_SEG_CPP_TEST_DATA,
// else "test/data" relative to the working directory.

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ransac_seg_cpp/lane.hpp"

#ifndef RANSAC_SEG_CPP_TEST_DATA
#define RANSAC_SEG_CPP_TEST_DATA "test/data"
#endif

using ransac_seg_cpp::LaneFit;
using ransac_seg_cpp::LaneOptions;
using ransac_seg_cpp::fit_lane;
using ransac_seg_cpp::lane_detail::longest_run;
using ransac_seg_cpp::lane_detail::resample;

namespace
{

constexpr double kTrueWidth = 6.00;
// Inset of the default 2nd/98th percentile edges on a uniform-density road.
// NOT a bug to fix: the tests compare against this estimand deliberately.
constexpr double kExpectedWidth = kTrueWidth * (1.0 - 2.0 * 0.02);

// Measured this session against the Python on the golden roads: the largest
// disagreement anywhere outside the sharp bend is 5.3e-14 m. The bend is the
// one case where the SmoothingSpline deviation bites (see test_spline.cpp);
// there the centreline still agrees to 2.8e-14 m and the boundaries part by
// 1.8e-01 m.
constexpr double kParityTolerance = 1e-10;
constexpr double kBendBoundaryTolerance = 0.35;

std::string data_dir()
{
  const char * env = std::getenv("RANSAC_SEG_CPP_TEST_DATA");
  return env != nullptr ? std::string(env) : std::string(RANSAC_SEG_CPP_TEST_DATA);
}

struct GoldenCase
{
  std::string name;
  std::vector<float> cloud;
  std::size_t count = 0;
  int stride = 2;
  LaneOptions opts;
  bool valid = false;
  std::string reason;
  std::vector<double> centerline;
  std::vector<double> left;
  std::vector<double> right;
  std::vector<double> width;
  std::vector<double> arclength;

  LaneFit run() const {return fit_lane(cloud.data(), count, stride, opts);}
};

std::vector<double> read_numbers(std::istringstream & in)
{
  std::vector<double> values;
  double value = 0.0;
  while (in >> value) {values.push_back(value);}
  return values;
}

std::map<std::string, GoldenCase> load_golden()
{
  std::map<std::string, GoldenCase> cases;
  const std::string dir = data_dir();
  std::ifstream file(dir + "/golden_lane.txt");
  if (!file) {return cases;}
  std::string line;
  GoldenCase pending;
  while (std::getline(file, line)) {
    std::istringstream in(line);
    std::string tag;
    in >> tag;
    if (tag == "case") {
      pending = GoldenCase{};
      in >> pending.name;
    } else if (tag == "cloud") {
      std::string relative;
      in >> relative >> pending.count >> pending.stride;
      pending.cloud.assign(pending.count * static_cast<std::size_t>(pending.stride), 0.0f);
      std::ifstream binary(dir + "/" + relative, std::ios::binary);
      binary.read(
        reinterpret_cast<char *>(pending.cloud.data()),
        static_cast<std::streamsize>(pending.cloud.size() * sizeof(float)));
    } else if (tag == "opts") {
      const std::vector<double> v = read_numbers(in);
      pending.opts.bin_length = v[0];
      pending.opts.min_points_per_bin = static_cast<int>(v[1]);
      pending.opts.edge_percentile = v[2];
      pending.opts.centerline_degree = static_cast<int>(v[3]);
      pending.opts.robust_iterations = static_cast<int>(v[4]);
      pending.opts.max_gap = v[5];
      pending.opts.smoothing_per_point = v[6];
      pending.opts.n_samples = static_cast<int>(v[7]);
      pending.opts.min_width = v[8];
      pending.opts.max_width = v[9];
      pending.opts.min_bin_fill = v[10];
    } else if (tag == "valid") {
      int value = 0;
      in >> value;
      pending.valid = value != 0;
    } else if (tag == "reason") {
      in >> pending.reason;
      if (pending.reason == "-") {pending.reason.clear();}
    } else if (tag == "centerline") {
      pending.centerline = read_numbers(in);
    } else if (tag == "left") {
      pending.left = read_numbers(in);
    } else if (tag == "right") {
      pending.right = read_numbers(in);
    } else if (tag == "width") {
      pending.width = read_numbers(in);
    } else if (tag == "arclength") {
      pending.arclength = read_numbers(in);
      cases[pending.name] = pending;
    }
  }
  return cases;
}

const std::map<std::string, GoldenCase> & golden()
{
  static const std::map<std::string, GoldenCase> cases = load_golden();
  return cases;
}

const GoldenCase & road(const std::string & name)
{
  const auto found = golden().find(name);
  if (found == golden().end()) {
    ADD_FAILURE() << "golden case '" << name << "' missing under " << data_dir()
                  << "; regenerate with python3 cpp/tools/dump_golden.py";
    static const GoldenCase fallback;
    return fallback;
  }
  return found->second;
}

double max_abs_difference(const std::vector<double> & a, const std::vector<double> & b)
{
  if (a.size() != b.size()) {return std::numeric_limits<double>::infinity();}
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {worst = std::max(worst, std::abs(a[i] - b[i]));}
  return worst;
}

double max_abs_offset(const std::vector<double> & values, double from)
{
  double worst = 0.0;
  for (double v : values) {worst = std::max(worst, std::abs(v - from));}
  return worst;
}

}  // namespace

TEST(Resample, VertexCountUsesHalfToEvenRounding)
{
  // 0.625 m at 0.25 m spacing is exactly 2.5 steps. Python's round() is
  // half-to-even and gives 2, so 3 vertices; std::round would give 3 and a
  // fourth vertex, shifting every downstream nearest-vertex projection.
  const std::vector<double> segment{0.0, 0.0, 0.625, 0.0};
  EXPECT_EQ(resample(segment).size() / 2, 3u);

  // 0.875 m is 3.5 steps: half-to-even gives 4, so 5 vertices.
  const std::vector<double> longer{0.0, 0.0, 0.875, 0.0};
  EXPECT_EQ(resample(longer).size() / 2, 5u);
}

TEST(Resample, DegenerateCurveIsReturnedUnchanged)
{
  const std::vector<double> point{3.0, 4.0, 3.0, 4.0};
  EXPECT_EQ(resample(point), point);
}

TEST(LongestRun, ScoresBySpanNotBinCount)
{
  // Five bins spanning 8 m against three spanning 12 m: metres win, not bins.
  const std::vector<double> centres{0.0, 2.0, 4.0, 6.0, 8.0, 30.0, 36.0, 42.0};
  const auto run = longest_run(centres, 6.0);
  EXPECT_EQ(run.first, 5u);
  EXPECT_EQ(run.second, 8u);
}

TEST(LongestRun, AGapOfExactlyMaxGapDoesNotBreak)
{
  const std::vector<double> centres{0.0, 6.0, 12.0};
  const auto run = longest_run(centres, 6.0);
  EXPECT_EQ(run.first, 0u);
  EXPECT_EQ(run.second, 3u);

  const std::vector<double> wider{0.0, 6.01, 12.0};
  const auto split = longest_run(wider, 6.0);
  EXPECT_EQ(split.first, 1u);
  EXPECT_EQ(split.second, 3u);
}

TEST(LongestRun, TiesTakeTheFirstMaximum)
{
  const std::vector<double> centres{0.0, 4.0, 20.0, 24.0};
  const auto run = longest_run(centres, 6.0);
  EXPECT_EQ(run.first, 0u);
  EXPECT_EQ(run.second, 2u);
}

TEST(LongestRun, EmptyInput)
{
  const auto run = longest_run({}, 6.0);
  EXPECT_EQ(run.first, 0u);
  EXPECT_EQ(run.second, 0u);
}

TEST(FitLane, StraightRoadWidth)
{
  const LaneFit fit = road("straight").run();
  ASSERT_TRUE(fit.valid);
  EXPECT_EQ(fit.reason, "");
  const double against_estimand = max_abs_offset(fit.width, kExpectedWidth);
  const double against_nominal = max_abs_offset(fit.width, kTrueWidth);
  std::cout << "[ LANE     ] straight: max|w - 5.76| = " << against_estimand
            << " m, max|w - 6.00| = " << against_nominal << " m" << std::endl;
  EXPECT_LT(against_estimand, 0.15);
  EXPECT_LT(against_nominal, 0.35);
  EXPECT_EQ(fit.left.size(), fit.right.size());
  EXPECT_EQ(fit.left.size(), fit.centerline.size());
  for (std::size_t i = 1; i < fit.arclength.size(); ++i) {
    EXPECT_GT(fit.arclength[i], fit.arclength[i - 1]);
  }
}

TEST(FitLane, CurvedRoadWidth)
{
  const LaneFit fit = road("curved").run();
  ASSERT_TRUE(fit.valid);
  const double against_estimand = max_abs_offset(fit.width, kExpectedWidth);
  const double against_nominal = max_abs_offset(fit.width, kTrueWidth);
  // The fixed-x-column fitter this replaced was reported at 8.06 m on this
  // geometry by the audit that motivated the rewrite.
  std::cout << "[ LANE     ] curved:   max|w - 5.76| = " << against_estimand
            << " m, max|w - 6.00| = " << against_nominal << " m" << std::endl;
  EXPECT_LT(against_estimand, 0.15);
  EXPECT_LT(against_nominal, 0.35);
}

TEST(FitLane, CurvedCenterlineFollowsTheArc)
{
  const LaneFit fit = road("curved").run();
  ASSERT_TRUE(fit.valid);
  double worst = 0.0;
  for (std::size_t i = 0; i < fit.width.size(); ++i) {
    const double x = fit.centerline[2 * i];
    worst = std::max(worst, std::abs(fit.centerline[2 * i + 1] - 0.01 * x * x));
  }
  std::cout << "[ LANE     ] curved centreline residual vs y = 0.01x^2: " << worst
            << " m" << std::endl;
  EXPECT_LT(worst, 0.5);
}

TEST(FitLane, RoadsideObstacleDoesNotDragTheBoundary)
{
  const GoldenCase & scene = road("obstacle");
  const LaneFit fit = scene.run();
  ASSERT_TRUE(fit.valid);
  const double half = 3.5;

  double worst = 0.0;
  for (std::size_t i = 0; i < fit.width.size(); ++i) {
    const double x = fit.right[2 * i];
    if (x < 7.0 || x > 19.0) {continue;}
    worst = std::max(worst, std::abs(fit.right[2 * i + 1] + half));
  }
  std::cout << "[ LANE     ] obstacle: worst right-edge error over x in [7, 19] = "
            << worst << " m" << std::endl;
  EXPECT_LT(worst, 0.30);

  // The estimator this replaced, on the same cloud: min y per 2 m column.
  std::map<int, double> column_min;
  for (std::size_t i = 0; i < scene.count; ++i) {
    const double x = scene.cloud[2 * i];
    const double y = scene.cloud[2 * i + 1];
    const int column = static_cast<int>(std::floor(x / 2.0));
    const auto found = column_min.find(column);
    if (found == column_min.end() || y < found->second) {column_min[column] = y;}
  }
  double worst_min_based = 0.0;
  for (const auto & entry : column_min) {
    const double centre = (entry.first + 0.5) * 2.0;
    if (centre < 7.0 || centre > 19.0) {continue;}
    worst_min_based = std::max(worst_min_based, std::abs(entry.second + half));
  }
  std::cout << "[ LANE     ] obstacle: the min-per-column estimator is off by "
            << worst_min_based << " m over the same span" << std::endl;
  EXPECT_GT(worst_min_based, 1.0);
}

TEST(FitLane, GapIsNotBridged)
{
  const LaneFit fit = road("gap").run();
  ASSERT_TRUE(fit.valid);
  double max_x = -std::numeric_limits<double>::infinity();
  double min_x = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < fit.width.size(); ++i) {
    max_x = std::max(max_x, fit.centerline[2 * i]);
    min_x = std::min(min_x, fit.centerline[2 * i]);
  }
  // The 14 m near run beats the 10 m far run; nothing is drawn over the 16 m
  // hole, which is exactly what a spline through both runs would have done.
  EXPECT_LT(max_x, 16.0);
  EXPECT_LT(min_x, 4.0);
}

TEST(FitLane, TooFewPoints)
{
  const LaneFit fit = road("tiny").run();
  EXPECT_FALSE(fit.valid);
  EXPECT_EQ(fit.reason, "too_few_points");
  EXPECT_TRUE(fit.centerline.empty());
  EXPECT_TRUE(fit.left.empty());
  EXPECT_TRUE(fit.right.empty());
  EXPECT_TRUE(fit.width.empty());
  EXPECT_TRUE(fit.arclength.empty());
}

TEST(FitLane, EmptyInput)
{
  const LaneFit fit = fit_lane(nullptr, 0, 3, LaneOptions{});
  EXPECT_FALSE(fit.valid);
  EXPECT_EQ(fit.reason, "too_few_points");
  EXPECT_TRUE(fit.width.empty());
}

TEST(FitLane, ShortRunReportsGap)
{
  const LaneFit fit = road("short_run").run();
  EXPECT_FALSE(fit.valid);
  EXPECT_EQ(fit.reason, "gap");
  EXPECT_TRUE(fit.width.empty());
}

TEST(FitLane, CrossedBoundariesAreRejected)
{
  // Percentiles above the median swap the two edges, which is the only way to
  // make a boundary pair cross on synthetic data.
  const LaneFit fit = road("straight_p60").run();
  EXPECT_FALSE(fit.valid);
  EXPECT_EQ(fit.reason, "crossed");
  // crossed carries FULL-LENGTH arrays so a caller can see where it went wrong.
  EXPECT_EQ(fit.width.size(), 120u);
  EXPECT_EQ(fit.centerline.size(), 240u);
}

TEST(FitLane, WidthGate)
{
  const LaneFit narrow = road("straight_minw8").run();
  EXPECT_EQ(narrow.reason, "width_out_of_range");
  EXPECT_FALSE(narrow.valid);
  EXPECT_EQ(narrow.width.size(), 120u);

  const LaneFit wide = road("straight_maxw4").run();
  EXPECT_EQ(wide.reason, "width_out_of_range");
  EXPECT_EQ(wide.width.size(), 120u);
}

TEST(FitLane, SharpBendDoesNotRaiseOrReturnGarbage)
{
  // A degree-2 centreline cannot represent a 120 degree bend, so the fit is
  // expected to fail its width gate — but it must fail by REPORTING, never by
  // raising or by emitting a multivalued curve.
  const LaneFit fit = road("bend").run();
  for (double v : fit.centerline) {EXPECT_TRUE(std::isfinite(v));}
  for (double v : fit.width) {EXPECT_TRUE(std::isfinite(v));}
  for (std::size_t i = 1; i < fit.arclength.size(); ++i) {
    EXPECT_GT(fit.arclength[i], fit.arclength[i - 1]);
  }
  EXPECT_EQ(fit.reason, "width_out_of_range");
}

TEST(FitLane, ThreeColumnInputIsAccepted)
{
  const GoldenCase & scene = road("xyz3");
  EXPECT_EQ(scene.stride, 3);
  EXPECT_TRUE(scene.run().valid);
}

TEST(FitLane, NonFiniteRowsAreDropped)
{
  // Not in the Python suite, but _as_xy filters and this is the only place the
  // filter is exercised on the lane side.
  const GoldenCase & scene = road("straight");
  std::vector<float> polluted;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (std::size_t i = 0; i < scene.count; ++i) {
    if (i % 500 == 0) {polluted.insert(polluted.end(), {nan, 0.0f});}
    polluted.push_back(scene.cloud[2 * i]);
    polluted.push_back(scene.cloud[2 * i + 1]);
  }
  const LaneFit fit = fit_lane(polluted.data(), polluted.size() / 2, 2, LaneOptions{});
  ASSERT_TRUE(fit.valid);
  EXPECT_LT(max_abs_difference(fit.width, scene.run().width), 1e-12);
}

// The header promises fit_lane never throws. These four are the ways it did:
// each one used to abort the process out of a subscription callback, three
// through an allocation the bin lattice asked for and one through the spline
// refusing a descending x. All four are reachable from `lane.lane_bin_length`,
// a plain double ROS parameter, or from an ungated far-field return.

TEST(FitLane, ATinyBinLengthIsRefusedNotAllocated)
{
  const GoldenCase & scene = road("straight");
  LaneOptions opts = scene.opts;
  opts.bin_length = 1e-9;  // ~6e10 bins over a 60 m road
  const LaneFit fit = fit_lane(scene.cloud.data(), scene.count, scene.stride, opts);
  EXPECT_FALSE(fit.valid);
  EXPECT_EQ(fit.reason, "bad_bins");
  EXPECT_TRUE(fit.width.empty());
}

TEST(FitLane, ANegativeBinLengthIsRefused)
{
  // Descending bin centres, which used to reach SmoothingSpline and throw
  // "x must be strictly increasing" straight out of fit_lane.
  const GoldenCase & scene = road("straight");
  LaneOptions opts = scene.opts;
  opts.bin_length = -1.0;
  const LaneFit fit = fit_lane(scene.cloud.data(), scene.count, scene.stride, opts);
  EXPECT_FALSE(fit.valid);
  EXPECT_EQ(fit.reason, "bad_bins");
}

TEST(FitLane, AFarFieldReturnIsRefusedNotAllocated)
{
  // One finite return at 1e10 m past a normal road: the span over the default
  // 2 m bin needs five billion bins.
  const GoldenCase & scene = road("straight");
  std::vector<float> polluted(
    scene.cloud.begin(), scene.cloud.begin() + 2 * static_cast<std::ptrdiff_t>(scene.count));
  polluted.push_back(1e10f);
  polluted.push_back(0.0f);
  const LaneFit fit = fit_lane(polluted.data(), polluted.size() / 2, 2, scene.opts);
  EXPECT_FALSE(fit.valid);
  EXPECT_EQ(fit.reason, "bad_bins");
}

TEST(FitLane, AReturnPastTheInt64LatticeIsRefusedNotWrapped)
{
  // 1e25 / 2 does not fit in an int64, and the plain cast that used to compute
  // the bin index is undefined there: it produced INT64_MIN, which then
  // signed-overflowed the rebasing and handed std::vector a wrapped size.
  const GoldenCase & scene = road("straight");
  std::vector<float> polluted(
    scene.cloud.begin(), scene.cloud.begin() + 2 * static_cast<std::ptrdiff_t>(scene.count));
  polluted.push_back(1e25f);
  polluted.push_back(0.0f);
  const LaneFit fit = fit_lane(polluted.data(), polluted.size() / 2, 2, scene.opts);
  EXPECT_FALSE(fit.valid);
  EXPECT_EQ(fit.reason, "bad_bins");
}

TEST(FitLane, MatchesThePythonOnEveryGoldenRoad)
{
  ASSERT_FALSE(golden().empty())
    << "golden_lane.txt not found under " << data_dir()
    << "; regenerate with python3 cpp/tools/dump_golden.py";

  for (const auto & entry : golden()) {
    const GoldenCase & scene = entry.second;
    const LaneFit fit = scene.run();
    EXPECT_EQ(fit.reason, scene.reason) << scene.name;
    EXPECT_EQ(fit.valid, scene.valid) << scene.name;
    const double centre = max_abs_difference(fit.centerline, scene.centerline);
    const double left = max_abs_difference(fit.left, scene.left);
    const double right = max_abs_difference(fit.right, scene.right);
    const double width = max_abs_difference(fit.width, scene.width);
    const double arc = max_abs_difference(fit.arclength, scene.arclength);
    std::cout << "[ LANE     ] " << entry.first << ": centre=" << centre
              << " left=" << left << " right=" << right << " width=" << width
              << " arclength=" << arc << " (metres)" << std::endl;

    EXPECT_LT(centre, kParityTolerance) << scene.name;
    EXPECT_LT(arc, kParityTolerance) << scene.name;
    if (scene.name == "bend") {
      // The one road whose spline needs real smoothing; the difference is the
      // knot mechanism and nothing else, which is why the centreline above
      // still agrees to double precision.
      EXPECT_LT(left, kBendBoundaryTolerance) << scene.name;
      EXPECT_LT(right, kBendBoundaryTolerance) << scene.name;
      EXPECT_LT(width, kBendBoundaryTolerance) << scene.name;
    } else {
      EXPECT_LT(left, kParityTolerance) << scene.name;
      EXPECT_LT(right, kParityTolerance) << scene.name;
      EXPECT_LT(width, kParityTolerance) << scene.name;
    }
  }
}
