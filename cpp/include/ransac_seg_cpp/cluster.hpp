// Obstacle instance extraction from the non-ground returns — C++ port of
// ransac_seg/cluster.py.
//
// The linkage runs on a voxel lattice rather than the raw cloud for the reason
// the Python states: a 120k point frame has ~7 billion pairs while its 0.2 m
// voxels number in the low thousands, and only the second is affordable in a
// per-sweep callback.
//
// Points arrive as a flat float array with a stride because that is how a
// PointCloud2 payload and a KITTI .bin already sit in memory; an (N, 4) cloud
// with an intensity column is accepted and its first three columns used.

#ifndef RANSAC_SEG_CPP__CLUSTER_HPP_
#define RANSAC_SEG_CPP__CLUSTER_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Dense>

namespace ransac_seg_cpp
{

/// One obstacle instance extracted from the non-ground returns.
struct Cluster
{
  Eigen::Vector3d centroid{0.0, 0.0, 0.0};
  Eigen::Vector3d min_bound{0.0, 0.0, 0.0};
  Eigen::Vector3d max_bound{0.0, 0.0, 0.0};
  int count = 0;
  /// Row numbers into the CALLER'S array, before any non-finite row was
  /// dropped, ascending. A consumer indexes its own cloud with these.
  std::vector<int32_t> indices;

  /// The (3,) bounding-box size in metres.
  Eigen::Vector3d extent() const {return max_bound - min_bound;}

  /// The vertical extent in metres.
  double height() const {return extent().z();}
};

/// Per-voxel centroids plus the index of every input point into them.
struct VoxelDownsample
{
  std::vector<double> reps;      ///< 3 * V, row-major xyz
  std::vector<int64_t> inverse;  ///< N, one entry per input point
};

/// Collapse an (N, 3) contiguous double cloud onto a voxel lattice.
///
/// The caller has already coerced and finite-filtered the cloud (that is what
/// cluster_obstacles does), so this takes packed xyz and no stride.
///
/// An empty cloud returns empty arrays BEFORE the voxel_size check, so an empty
/// input with voxel_size 0 does not throw — the Python orders the two the same
/// way. A non-empty cloud with voxel_size <= 0 throws std::invalid_argument.
VoxelDownsample voxel_downsample(const double * xyz, std::size_t count, double voxel_size = 0.2);

/// Defaults mirror ransac_seg.cluster.cluster_obstacles.
struct ClusterOptions
{
  double voxel_size = 0.2;
  double link_radius = 0.6;
  int min_points = 12;
  double min_height = 0.15;
  double max_height = 4.0;
  int max_clusters = 200;
};

/// Connected-component labels over voxel representatives, exposed for tests.
///
/// Two representatives are linked when their Euclidean distance is <=
/// link_radius (inclusive, like scipy's query_pairs: a strict < changes the
/// linkage exactly at the boundary). Labels are numbered by first visit in
/// index order, so label 0 contains node 0 — scipy's connected_components
/// convention, which the ordering of everything downstream inherits.
std::vector<int64_t> voxel_labels(const std::vector<double> & reps, double link_radius);

/// Group non-ground returns into obstacle instances by Euclidean linkage.
///
/// `link_radius` must be at least voxel_size * sqrt(3): two returns that are
/// neighbours in the raw cloud can land in diagonally adjacent voxels whose
/// centroids sit that far apart, so a shorter radius fragments one car into
/// four. A smaller value throws std::invalid_argument (message contains
/// "fragment") before any point is inspected, so it throws on an empty cloud too.
///
/// Survivors are gated on raw point count and on vertical extent, then sorted
/// by count descending (stably, so ties keep ascending label order) and
/// truncated to max_clusters.
std::vector<Cluster> cluster_obstacles(
  const float * xyz, std::size_t count, int stride, const ClusterOptions & opts);

}  // namespace ransac_seg_cpp

#endif  // RANSAC_SEG_CPP__CLUSTER_HPP_
