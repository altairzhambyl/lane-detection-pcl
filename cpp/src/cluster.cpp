#include "ransac_seg_cpp/cluster.hpp"

#include "ransac_seg_cpp/numeric.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace ransac_seg_cpp
{

namespace
{

const double kSqrt3 = std::sqrt(3.0);

/// Union-find over the voxel representatives; the linkage graph is built once
/// and only ever queried for components, so path compression is all it needs.
struct DisjointSet
{
  std::vector<int64_t> parent;

  explicit DisjointSet(std::size_t n)
  : parent(n)
  {
    std::iota(parent.begin(), parent.end(), int64_t{0});
  }

  int64_t find(int64_t a)
  {
    while (parent[static_cast<std::size_t>(a)] != a) {
      parent[static_cast<std::size_t>(a)] =
        parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(a)])];
      a = parent[static_cast<std::size_t>(a)];
    }
    return a;
  }

  void unite(int64_t a, int64_t b)
  {
    a = find(a);
    b = find(b);
    if (a != b) {parent[static_cast<std::size_t>(std::max(a, b))] = std::min(a, b);}
  }
};

/// Key of a uniform bucket grid; three int64 cell coordinates.
struct CellKey
{
  int64_t x;
  int64_t y;
  int64_t z;
  bool operator==(const CellKey & other) const
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct CellHash
{
  std::size_t operator()(const CellKey & k) const
  {
    // Three odd 64-bit constants; the lattice is dense and axis-aligned, so a
    // plain shift-xor mix collides badly along whole rows.
    std::size_t h = static_cast<std::size_t>(k.x) * 0x9e3779b97f4a7c15ULL;
    h ^= static_cast<std::size_t>(k.y) * 0xc2b2ae3d27d4eb4fULL;
    h ^= static_cast<std::size_t>(k.z) * 0x165667b19e3779f9ULL;
    return h ^ (h >> 29);
  }
};

}  // namespace

VoxelDownsample voxel_downsample(const double * xyz, std::size_t count, double voxel_size)
{
  VoxelDownsample result;
  // Empty first, then the parameter check: an empty cloud with voxel_size 0
  // must not throw, and the Python orders the two the same way.
  if (count == 0) {return result;}
  if (!(voxel_size > 0.0)) {
    std::ostringstream message;
    message << "voxel_size must be positive, got " << voxel_size;
    throw std::invalid_argument(message.str());
  }

  std::vector<std::array<int64_t, 3>> keys(count);
  std::array<int64_t, 3> lo{0, 0, 0};
  std::array<int64_t, 3> hi{0, 0, 0};
  for (std::size_t i = 0; i < count; ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      // floor, not truncation: a point at -0.05 belongs to cell -1. Saturating,
      // because a finite far-field return divided by a 0.05 m voxel leaves the
      // int64 range and the plain cast is undefined there — see floor_index.
      const int64_t k =
        floor_index(xyz[3 * i + static_cast<std::size_t>(axis)] / voxel_size);
      keys[i][static_cast<std::size_t>(axis)] = k;
      if (i == 0) {
        lo[static_cast<std::size_t>(axis)] = k;
        hi[static_cast<std::size_t>(axis)] = k;
      } else {
        lo[static_cast<std::size_t>(axis)] = std::min(lo[static_cast<std::size_t>(axis)], k);
        hi[static_cast<std::size_t>(axis)] = std::max(hi[static_cast<std::size_t>(axis)], k);
      }
    }
  }

  double spans[3];
  for (int axis = 0; axis < 3; ++axis) {
    // Each endpoint widened to double BEFORE subtracting: the whole point of
    // the 2^62 test below is to detect a lattice too wide to ravel, so the
    // value it judges must not itself be the wrap it is looking for.
    spans[axis] = static_cast<double>(hi[static_cast<std::size_t>(axis)]) -
      static_cast<double>(lo[static_cast<std::size_t>(axis)]) + 1.0;
  }
  const bool ravel = spans[0] * spans[1] * spans[2] < 4.611686018427388e18;  // 2^62

  std::vector<std::size_t> order(count);
  std::iota(order.begin(), order.end(), std::size_t{0});
  if (ravel) {
    // Ravelling the lattice index into one integer turns the row-wise lexsort
    // into a scalar sort — the Python measured 30.4 ms -> 3.2 ms on an
    // 18k-point obstacle channel. The 2^62 guard exists so a cloud spanning
    // more cells than an int64 can index takes the general path below instead
    // of silently wrapping.
    const int64_t span_y = static_cast<int64_t>(spans[1]);
    const int64_t span_z = static_cast<int64_t>(spans[2]);
    std::vector<int64_t> flat(count);
    for (std::size_t i = 0; i < count; ++i) {
      flat[i] = ((keys[i][0] - lo[0]) * span_y + (keys[i][1] - lo[1])) * span_z +
        (keys[i][2] - lo[2]);
    }
    std::sort(
      order.begin(), order.end(),
      [&flat](std::size_t a, std::size_t b) {return flat[a] < flat[b];});
    result.inverse.assign(count, 0);
    int64_t voxel = -1;
    int64_t previous = 0;
    for (std::size_t rank = 0; rank < count; ++rank) {
      const std::size_t i = order[rank];
      if (rank == 0 || flat[i] != previous) {
        ++voxel;
        previous = flat[i];
      }
      result.inverse[i] = voxel;
    }
  } else {
    // Ascending flat key and lexicographic (kx, ky, kz) are the same ordering,
    // so both paths hand identical labels downstream.
    std::sort(
      order.begin(), order.end(), [&keys](std::size_t a, std::size_t b) {
        return keys[a] < keys[b];
      });
    result.inverse.assign(count, 0);
    int64_t voxel = -1;
    std::array<int64_t, 3> previous{0, 0, 0};
    for (std::size_t rank = 0; rank < count; ++rank) {
      const std::size_t i = order[rank];
      if (rank == 0 || keys[i] != previous) {
        ++voxel;
        previous = keys[i];
      }
      result.inverse[i] = voxel;
    }
  }

  const std::size_t n_voxels =
    count == 0 ? 0 : static_cast<std::size_t>(result.inverse[order.back()]) + 1;
  result.reps.assign(3 * n_voxels, 0.0);
  std::vector<int64_t> members(n_voxels, 0);
  for (std::size_t i = 0; i < count; ++i) {
    const std::size_t v = static_cast<std::size_t>(result.inverse[i]);
    result.reps[3 * v + 0] += xyz[3 * i + 0];
    result.reps[3 * v + 1] += xyz[3 * i + 1];
    result.reps[3 * v + 2] += xyz[3 * i + 2];
    ++members[v];
  }
  for (std::size_t v = 0; v < n_voxels; ++v) {
    const double inv = 1.0 / static_cast<double>(members[v]);
    // The representative is the per-voxel CENTROID, never the voxel centre;
    // the centroid is what makes the 0.2 m lattice a lossy resample rather
    // than a quantisation that shifts every cluster bound by up to half a cell.
    result.reps[3 * v + 0] *= inv;
    result.reps[3 * v + 1] *= inv;
    result.reps[3 * v + 2] *= inv;
  }
  return result;
}

std::vector<int64_t> voxel_labels(const std::vector<double> & reps, double link_radius)
{
  const std::size_t n = reps.size() / 3;
  std::vector<int64_t> labels(n, 0);
  if (n == 0) {return labels;}

  // DECLARED DEVIATION from scipy's cKDTree.query_pairs: a uniform bucket grid
  // of cell size link_radius, searching the 27 neighbouring buckets. Every pair
  // within the radius lands in one of those buckets, so the pair set is exact,
  // the traversal order is deterministic, and at a few thousand voxels it beats
  // building a kd-tree.
  const double cell = link_radius > 0.0 ? link_radius : 1.0;
  std::unordered_map<CellKey, std::vector<int64_t>, CellHash> buckets;
  buckets.reserve(n * 2);
  std::vector<CellKey> cell_of(n);
  for (std::size_t i = 0; i < n; ++i) {
    // Saturating, and to a limit with headroom: the neighbour scan below forms
    // `base.x + dx`, which wraps if a bucket key sits at the int64 edge.
    CellKey key{
      floor_index(reps[3 * i + 0] / cell),
      floor_index(reps[3 * i + 1] / cell),
      floor_index(reps[3 * i + 2] / cell)};
    cell_of[i] = key;
    buckets[key].push_back(static_cast<int64_t>(i));
  }

  DisjointSet sets(n);
  const double radius_sq = link_radius * link_radius;
  bool any_pair = false;
  for (std::size_t i = 0; i < n; ++i) {
    const CellKey base = cell_of[i];
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          const auto found = buckets.find(CellKey{base.x + dx, base.y + dy, base.z + dz});
          if (found == buckets.end()) {continue;}
          for (int64_t j : found->second) {
            if (j <= static_cast<int64_t>(i)) {continue;}
            const std::size_t jj = static_cast<std::size_t>(j);
            const double ex = reps[3 * i + 0] - reps[3 * jj + 0];
            const double ey = reps[3 * i + 1] - reps[3 * jj + 1];
            const double ez = reps[3 * i + 2] - reps[3 * jj + 2];
            // Inclusive, like scipy's query_pairs.
            if (ex * ex + ey * ey + ez * ez <= radius_sq) {
              sets.unite(static_cast<int64_t>(i), j);
              any_pair = true;
            }
          }
        }
      }
    }
  }
  if (!any_pair) {
    // No links at all: every voxel is its own component, which the relabel
    // below would also produce — this just skips the walk.
    std::iota(labels.begin(), labels.end(), int64_t{0});
    return labels;
  }

  // Number by first visit scanning 0..V-1, so labels are ordered by smallest
  // member index and label 0 contains node 0 — scipy's convention.
  std::vector<int64_t> seen(n, -1);
  int64_t next = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t root = static_cast<std::size_t>(sets.find(static_cast<int64_t>(i)));
    if (seen[root] < 0) {seen[root] = next++;}
    labels[i] = seen[root];
  }
  return labels;
}

std::vector<Cluster> cluster_obstacles(
  const float * xyz, std::size_t count, int stride, const ClusterOptions & opts)
{
  // Validate before touching a point, so the contract violation is reported
  // even on an empty cloud.
  if (opts.link_radius < opts.voxel_size * kSqrt3) {
    std::ostringstream message;
    message.setf(std::ios::fixed, std::ios::floatfield);
    message.precision(3);
    message << "link_radius " << opts.link_radius << " is below voxel_size*sqrt(3) ("
            << opts.voxel_size * kSqrt3 << "); the lattice would fragment objects";
    throw std::invalid_argument(message.str());
  }
  if (stride < 3) {
    throw std::invalid_argument("cluster_obstacles: stride must be at least 3");
  }

  std::vector<Cluster> clusters;
  if (count == 0 || xyz == nullptr) {return clusters;}

  // Cluster::indices must address the caller's array, so carry the original row
  // numbers through the non-finite filter rather than reindexing afterwards.
  std::vector<int32_t> rows;
  std::vector<double> pts;
  rows.reserve(count);
  pts.reserve(3 * count);
  for (std::size_t i = 0; i < count; ++i) {
    const float * row = xyz + i * static_cast<std::size_t>(stride);
    if (!std::isfinite(row[0]) || !std::isfinite(row[1]) || !std::isfinite(row[2])) {continue;}
    rows.push_back(static_cast<int32_t>(i));
    pts.push_back(static_cast<double>(row[0]));
    pts.push_back(static_cast<double>(row[1]));
    pts.push_back(static_cast<double>(row[2]));
  }
  const std::size_t n = rows.size();
  if (n == 0) {return clusters;}

  const VoxelDownsample voxels = voxel_downsample(pts.data(), n, opts.voxel_size);
  const std::vector<int64_t> per_voxel = voxel_labels(voxels.reps, opts.link_radius);

  std::vector<int64_t> point_labels(n);
  int64_t max_label = 0;
  for (std::size_t i = 0; i < n; ++i) {
    point_labels[i] = per_voxel[static_cast<std::size_t>(voxels.inverse[i])];
    max_label = std::max(max_label, point_labels[i]);
  }

  std::vector<int64_t> counts(static_cast<std::size_t>(max_label) + 1, 0);
  for (int64_t label : point_labels) {++counts[static_cast<std::size_t>(label)];}

  // min_points gates the RAW POINT count of a component, not its voxel count.
  std::vector<int64_t> candidates;
  for (std::size_t label = 0; label < counts.size(); ++label) {
    if (counts[label] >= static_cast<int64_t>(opts.min_points)) {
      candidates.push_back(static_cast<int64_t>(label));
    }
  }
  if (candidates.empty()) {return clusters;}

  // One stable sort gives contiguous per-label runs, so the per-cluster
  // statistics below are slices rather than repeated full-array masks.
  const std::vector<std::size_t> order = stable_argsort(point_labels);

  std::vector<std::size_t> run_start(counts.size() + 1, 0);
  for (std::size_t label = 0; label < counts.size(); ++label) {
    run_start[label + 1] = run_start[label] + static_cast<std::size_t>(counts[label]);
  }

  for (int64_t label : candidates) {
    const std::size_t start = run_start[static_cast<std::size_t>(label)];
    const std::size_t stop = run_start[static_cast<std::size_t>(label) + 1];
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    Eigen::Vector3d lo{0.0, 0.0, 0.0};
    Eigen::Vector3d hi{0.0, 0.0, 0.0};
    Cluster cluster;
    cluster.indices.reserve(stop - start);
    for (std::size_t rank = start; rank < stop; ++rank) {
      const std::size_t i = order[rank];
      const Eigen::Vector3d p{pts[3 * i + 0], pts[3 * i + 1], pts[3 * i + 2]};
      if (rank == start) {
        lo = p;
        hi = p;
      } else {
        lo = lo.cwiseMin(p);
        hi = hi.cwiseMax(p);
      }
      sum += p;
      cluster.indices.push_back(rows[i]);
    }
    // Bounds and centroid are computed before the gate, exactly as the Python
    // does, so the gate reads the same numbers a survivor would report.
    const double height = hi.z() - lo.z();
    if (height < opts.min_height || height > opts.max_height) {continue;}
    cluster.centroid = sum / static_cast<double>(stop - start);
    cluster.min_bound = lo;
    cluster.max_bound = hi;
    cluster.count = static_cast<int>(stop - start);
    clusters.push_back(std::move(cluster));
  }

  // Stable, so equal counts keep the ascending-label order they arrived in.
  std::stable_sort(
    clusters.begin(), clusters.end(),
    [](const Cluster & a, const Cluster & b) {return a.count > b.count;});
  if (clusters.size() > static_cast<std::size_t>(std::max(0, opts.max_clusters))) {
    clusters.resize(static_cast<std::size_t>(std::max(0, opts.max_clusters)));
  }
  return clusters;
}

}  // namespace ransac_seg_cpp
