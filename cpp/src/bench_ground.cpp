// Score the C++ ground segmentation against SemanticKITTI labels.
//
// Prints the same quantities tools/eval_semantickitti.py prints — IoU,
// precision, recall, latency percentiles — over the same frames, so the two
// implementations are comparable rather than merely both "fast". A port that
// is quicker and wrong is not a port.
//
//   bench_ground <sequences-dir> <seq> [stride] [max-frames]

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <string>
#include <vector>

#include "ransac_seg_cpp/ground_plane.hpp"

namespace
{

// The class set the published ground-segmentation benchmarks use.
constexpr uint16_t kGroundClasses[] = {40, 44, 48, 49, 60, 72};
constexpr uint16_t kUnlabeled = 0;
constexpr double kMinRange = 0.5;
constexpr double kMaxRange = 60.0;

bool is_ground_class(uint16_t label)
{
  for (const uint16_t g : kGroundClasses) {
    if (label == g) {
      return true;
    }
  }
  return false;
}

/// Read a KITTI .bin sweep: float32 x,y,z,intensity per point.
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

/// Read a SemanticKITTI .label file; the semantic class is the low 16 bits.
std::vector<uint16_t> read_labels(const std::string & path)
{
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    return {};
  }
  const std::streamsize bytes = in.tellg();
  in.seekg(0);
  std::vector<uint32_t> packed(static_cast<std::size_t>(bytes) / sizeof(uint32_t));
  in.read(reinterpret_cast<char *>(packed.data()), bytes);
  std::vector<uint16_t> out(packed.size());
  for (std::size_t i = 0; i < packed.size(); ++i) {
    out[i] = static_cast<uint16_t>(packed[i] & 0xFFFFu);
  }
  return out;
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

}  // namespace

int main(int argc, char ** argv)
{
  if (argc < 3) {
    std::fprintf(stderr, "usage: bench_ground <sequences-dir> <seq> [stride] [max-frames]\n");
    return 2;
  }
  const std::string root = argv[1];
  const std::string seq = argv[2];
  const int stride = argc > 3 ? std::atoi(argv[3]) : 1;
  const int max_frames = argc > 4 ? std::atoi(argv[4]) : 1000000;

  const std::string velo_dir = root + "/" + seq + "/velodyne";
  const std::string label_dir = root + "/" + seq + "/labels";
  const std::vector<std::string> stems = list_stems(velo_dir);
  if (stems.empty()) {
    std::fprintf(stderr, "no .bin files under %s\n", velo_dir.c_str());
    return 1;
  }

  ransac_seg_cpp::RansacParams params;
  long tp = 0, fp = 0, fn = 0;
  int frames = 0, no_fit = 0;
  std::vector<double> latencies;
  std::vector<float> gated;

  for (std::size_t i = 0; i < stems.size() && frames < max_frames; i += std::max(stride, 1)) {
    const std::vector<float> raw = read_bin(velo_dir + "/" + stems[i] + ".bin");
    const std::vector<uint16_t> labels = read_labels(label_dir + "/" + stems[i] + ".label");
    if (raw.empty() || labels.empty()) {
      continue;
    }
    const std::size_t n = raw.size() / 4;
    if (labels.size() != n) {
      std::fprintf(stderr, "%s: %zu points but %zu labels\n", stems[i].c_str(), n, labels.size());
      return 1;
    }

    // Range gate first, exactly as the nodes do, carrying labels alongside.
    gated.clear();
    gated.reserve(n * 3);
    std::vector<uint16_t> gated_labels;
    gated_labels.reserve(n);
    for (std::size_t p = 0; p < n; ++p) {
      const float x = raw[4 * p], y = raw[4 * p + 1], z = raw[4 * p + 2];
      const double r = std::sqrt(static_cast<double>(x) * x + static_cast<double>(y) * y);
      if (r > kMinRange && r < kMaxRange) {
        gated.push_back(x);
        gated.push_back(y);
        gated.push_back(z);
        gated_labels.push_back(labels[p]);
      }
    }
    if (gated_labels.size() < 100) {
      continue;
    }

    const auto start = std::chrono::steady_clock::now();
    const auto seg = ransac_seg_cpp::segment_ground(gated.data(), gated_labels.size(), params);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    latencies.push_back(std::chrono::duration<double, std::milli>(elapsed).count());
    if (!seg.model.has_value()) {
      ++no_fit;
    }

    for (std::size_t p = 0; p < gated_labels.size(); ++p) {
      if (gated_labels[p] == kUnlabeled) {
        continue;  // excluded from scoring, not counted as a negative
      }
      const bool truth = is_ground_class(gated_labels[p]);
      const bool pred = seg.ground[p] != 0;
      if (pred && truth) {
        ++tp;
      } else if (pred) {
        ++fp;
      } else if (truth) {
        ++fn;
      }
    }
    ++frames;
  }

  const double iou = tp + fp + fn > 0 ? static_cast<double>(tp) / (tp + fp + fn) : 0.0;
  const double precision = tp + fp > 0 ? static_cast<double>(tp) / (tp + fp) : 0.0;
  const double recall = tp + fn > 0 ? static_cast<double>(tp) / (tp + fn) : 0.0;

  std::printf("seq %s: %d frames (stride %d)\n", seq.c_str(), frames, stride);
  std::printf("  IoU %.3f  precision %.3f  recall %.3f\n", iou, precision, recall);
  std::printf(
    "  latency p50 %.2f ms  p95 %.2f ms  p99 %.2f ms\n",
    percentile(latencies, 50), percentile(latencies, 95), percentile(latencies, 99));
  std::printf("  frames with no plane: %d\n", no_fit);
  return 0;
}
