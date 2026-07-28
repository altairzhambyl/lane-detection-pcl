// Per-stage timing of the whole C++ pipeline on real SemanticKITTI sweeps.
//
// tools/bench_rosbag.py and tools/benchmarks.py print the same quantities for
// the Python package, so the two are comparable rather than both merely "fast".
// Every stage is timed separately because the interesting number is not the
// total: it is which stage the shared hub stops paying six times over.
//
// ROS-free on purpose — it reads KITTI .bin files directly, so it builds and
// runs on a bare checkout without a sourced ROS installation. "decode" is
// therefore the strided float copy pointcloud2_to_xyz performs, not a DDS
// deserialisation; the PointCloud2 path adds a memcpy of the same payload.
//
//   bench_pipeline <sequences-dir> <seq> [stride] [max-frames]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <string>
#include <vector>

#include "ransac_seg_cpp/cluster.hpp"
#include "ransac_seg_cpp/costmap.hpp"
#include "ransac_seg_cpp/ground_plane.hpp"
#include "ransac_seg_cpp/lane.hpp"
#include "ransac_seg_cpp/params.hpp"
#include "ransac_seg_cpp/plane_filter.hpp"
#include "ransac_seg_cpp/planners.hpp"

namespace
{

using ransac_seg_cpp::CostGrid;
using Clock = std::chrono::steady_clock;

std::vector<float> read_bin(const std::string & path)
{
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    return {};
  }
  const std::streamsize bytes = in.tellg();
  in.seekg(0);
  std::vector<float> raw(static_cast<std::size_t>(bytes) / sizeof(float));
  in.read(reinterpret_cast<char *>(raw.data()), bytes);
  return raw;
}

std::vector<std::string> list_stems(const std::string & dir)
{
  std::vector<std::string> stems;
  DIR * handle = opendir(dir.c_str());
  if (handle == nullptr) {
    return stems;
  }
  while (dirent * entry = readdir(handle)) {
    const std::string name = entry->d_name;
    if (name.size() > 4 && name.compare(name.size() - 4, 4, ".bin") == 0) {
      stems.push_back(name.substr(0, name.size() - 4));
    }
  }
  closedir(handle);
  std::sort(stems.begin(), stems.end());
  return stems;
}

double percentile(std::vector<double> values, double q)
{
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double pos = q / 100.0 * static_cast<double>(values.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(pos);
  const std::size_t hi = std::min(lo + 1, values.size() - 1);
  return values[lo] + (values[hi] - values[lo]) * (pos - static_cast<double>(lo));
}

double ms_since(const Clock::time_point & start)
{
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

/// One timed stage's samples.
struct Stage
{
  const char * name;
  std::vector<double> samples;
};

void report(const Stage & stage, int frames)
{
  std::printf(
    "  %-14s p50 %7.2f  p95 %7.2f  p99 %7.2f  ms   (%zu/%d frames)\n", stage.name,
    percentile(stage.samples, 50), percentile(stage.samples, 95), percentile(stage.samples, 99),
    stage.samples.size(), frames);
}

/// The perception prefix every Python node repeats: decode, gate, segment,
/// rasterise. Returned so the six-copy comparison runs the identical code.
double perception_prefix(
  const std::vector<float> & raw, const ransac_seg_cpp::CommonParams & params, CostGrid & out_grid)
{
  const auto start = Clock::now();
  const std::size_t n = raw.size() / 4;
  std::vector<float> points;
  points.reserve(n * 3);
  for (std::size_t p = 0; p < n; ++p) {
    const double x = raw[4 * p];
    const double y = raw[4 * p + 1];
    const double r = std::hypot(x, y);
    if (r >= params.min_range && r <= params.max_range && x > 0.0) {
      points.push_back(raw[4 * p]);
      points.push_back(raw[4 * p + 1]);
      points.push_back(raw[4 * p + 2]);
    }
  }
  const std::size_t count = points.size() / 3;
  const auto seg = ransac_seg_cpp::segment_ground(
    points.data(), count, ransac_seg_cpp::ransac_params(params), params.fallback_height);
  std::vector<float> ground;
  std::vector<float> obstacle;
  for (std::size_t p = 0; p < count; ++p) {
    std::vector<float> & sink = seg.ground[p] != 0 ? ground : obstacle;
    sink.insert(sink.end(), points.begin() + 3 * p, points.begin() + 3 * p + 3);
  }
  ransac_seg_cpp::CostmapOptions options;
  options.resolution = params.resolution;
  if (seg.model.has_value()) {
    options.plane = ransac_seg_cpp::Plane{seg.model->normal, seg.model->offset};
  }
  out_grid = ransac_seg_cpp::build_costmap(
    ground.data(), ground.size() / 3, 3, obstacle.data(), obstacle.size() / 3, 3, options);
  return ms_since(start);
}

}  // namespace

int main(int argc, char ** argv)
{
  if (argc < 3) {
    std::fprintf(stderr, "usage: bench_pipeline <sequences-dir> <seq> [stride] [max-frames]\n");
    return 2;
  }
  const std::string root = argv[1];
  const std::string seq = argv[2];
  const int stride = argc > 3 ? std::atoi(argv[3]) : 1;
  const int max_frames = argc > 4 ? std::atoi(argv[4]) : 1000000;

  const std::string velo_dir = root + "/" + seq + "/velodyne";
  const std::vector<std::string> stems = list_stems(velo_dir);
  if (stems.empty()) {
    std::fprintf(stderr, "no .bin files under %s\n", velo_dir.c_str());
    return 1;
  }

  const ransac_seg_cpp::CommonParams params;
  ransac_seg_cpp::PlaneFilter plane_filter(
    params.plane_filter_alpha, params.plane_max_tilt_jump_deg, params.plane_max_offset_jump);

  Stage decode{"decode", {}};
  Stage gate{"range-gate", {}};
  Stage segment{"segment", {}};
  Stage split{"split", {}};
  Stage costmap{"costmap", {}};
  Stage cluster{"cluster", {}};
  Stage lane{"lane", {}};
  Stage bfs_stage{"bfs", {}};
  Stage astar_stage{"astar", {}};
  Stage total{"END TO END", {}};
  Stage hub{"hub prefix x1", {}};
  Stage six{"six copies x6", {}};

  int frames = 0;
  std::size_t points_seen = 0;
  for (std::size_t i = 0; i < stems.size() && frames < max_frames; i += std::max(stride, 1)) {
    const std::vector<float> raw = read_bin(velo_dir + "/" + stems[i] + ".bin");
    if (raw.empty()) {
      continue;
    }
    const std::size_t n = raw.size() / 4;
    const auto frame_start = Clock::now();

    // decode: the strided copy pointcloud2_to_xyz makes out of the payload.
    auto t = Clock::now();
    std::vector<float> decoded;
    decoded.reserve(n * 3);
    for (std::size_t p = 0; p < n; ++p) {
      decoded.push_back(raw[4 * p]);
      decoded.push_back(raw[4 * p + 1]);
      decoded.push_back(raw[4 * p + 2]);
    }
    decode.samples.push_back(ms_since(t));

    t = Clock::now();
    std::vector<float> points;
    points.reserve(decoded.size());
    for (std::size_t p = 0; p < n; ++p) {
      const double x = decoded[3 * p];
      const double y = decoded[3 * p + 1];
      const double r = std::hypot(x, y);
      if (r >= params.min_range && r <= params.max_range && x > 0.0) {
        points.push_back(decoded[3 * p]);
        points.push_back(decoded[3 * p + 1]);
        points.push_back(decoded[3 * p + 2]);
      }
    }
    gate.samples.push_back(ms_since(t));
    const std::size_t count = points.size() / 3;
    if (count < 100) {
      continue;
    }
    points_seen += count;

    t = Clock::now();
    const auto seg = ransac_seg_cpp::segment_ground(
      points.data(), count, ransac_seg_cpp::ransac_params(params), params.fallback_height);
    segment.samples.push_back(ms_since(t));
    const auto filtered = plane_filter.update(seg.model);

    t = Clock::now();
    std::vector<float> ground;
    std::vector<float> obstacle;
    ground.reserve(points.size());
    obstacle.reserve(points.size());
    for (std::size_t p = 0; p < count; ++p) {
      std::vector<float> & sink = seg.ground[p] != 0 ? ground : obstacle;
      sink.insert(sink.end(), points.begin() + 3 * p, points.begin() + 3 * p + 3);
    }
    split.samples.push_back(ms_since(t));

    ransac_seg_cpp::CostmapOptions options;
    options.resolution = params.resolution;
    options.robot_radius = params.robot_radius;
    options.inflation_radius = params.inflation_radius;
    options.cost_scaling = params.cost_scaling;
    options.min_ground_points = params.min_ground_points;
    options.min_obstacle_points = params.min_obstacle_points;
    options.ray_fill = params.ray_fill;
    options.ray_fill_bins = params.ray_fill_bins;
    options.ray_fill_min_cells = params.ray_fill_min_cells;
    if (filtered.has_value()) {
      options.plane = ransac_seg_cpp::Plane{filtered->normal, filtered->offset};
    }
    t = Clock::now();
    const CostGrid grid = ransac_seg_cpp::build_costmap(
      ground.data(), ground.size() / 3, 3, obstacle.data(), obstacle.size() / 3, 3, options);
    costmap.samples.push_back(ms_since(t));

    t = Clock::now();
    const auto clusters = ransac_seg_cpp::cluster_obstacles(
      obstacle.data(), obstacle.size() / 3, 3, ransac_seg_cpp::ClusterOptions{});
    cluster.samples.push_back(ms_since(t));
    (void)clusters;

    t = Clock::now();
    const auto fit = ransac_seg_cpp::fit_lane(
      ground.data(), ground.size() / 3, 3, ransac_seg_cpp::LaneOptions{});
    lane.samples.push_back(ms_since(t));
    (void)fit;

    const auto start_cell = ransac_seg_cpp::snap_to_traversable(grid, 0.0, 0.0, 2.0);
    if (start_cell.has_value()) {
      t = Clock::now();
      const auto hops = ransac_seg_cpp::bfs(grid, *start_cell, 4);
      bfs_stage.samples.push_back(ms_since(t));
      (void)hops;

      t = Clock::now();
      const auto goal = ransac_seg_cpp::select_goal(grid, 25.0, 0.0, 5.0, 1.0, 0.0);
      if (goal.has_value()) {
        const auto path = ransac_seg_cpp::astar(grid, *start_cell, *goal, 8);
        (void)path;
      }
      astar_stage.samples.push_back(ms_since(t));
    }
    total.samples.push_back(ms_since(frame_start));

    // The architectural claim, measured: one shared perception prefix against
    // six independent repetitions of it, which is what six Python processes
    // subscribing to the same cloud actually do.
    CostGrid scratch;
    hub.samples.push_back(perception_prefix(raw, params, scratch));
    double repeated = 0.0;
    for (int copy = 0; copy < 6; ++copy) {
      repeated += perception_prefix(raw, params, scratch);
    }
    six.samples.push_back(repeated);

    ++frames;
  }

  std::printf(
    "seq %s: %d frames (stride %d), %.0f gated points/frame\n", seq.c_str(), frames, stride,
    frames > 0 ? static_cast<double>(points_seen) / frames : 0.0);
  for (const Stage * stage :
    {&decode, &gate, &segment, &split, &costmap, &cluster, &lane, &bfs_stage, &astar_stage,
      &total})
  {
    report(*stage, frames);
  }
  std::printf("shared hub versus the six-process topology:\n");
  report(hub, frames);
  report(six, frames);
  return 0;
}
