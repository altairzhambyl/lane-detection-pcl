// The one marker shape three consumers publish: a scalar field over the whole
// costmap, quantised into bands.
//
// dijkstra.py, bfs.py and dfs.py each carried a byte-identical `_publish_field`
// and `_clear`. They differ only in what they put on the x axis — accumulated
// cost, hop count, discovery order — so the drawing lives here once and the
// three consumers stay the size of the thing that is actually different
// between them.

#ifndef RANSAC_SEG_CPP__FIELD_MARKERS_HPP_
#define RANSAC_SEG_CPP__FIELD_MARKERS_HPP_

#include <cstdint>
#include <string>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "ransac_seg_cpp/cost_grid.hpp"

namespace ransac_seg_cpp
{

/// One uniformly coloured CUBE_LIST per non-empty band, led by a DELETEALL.
///
/// The leading DELETEALL is what retires cells that dropped out of the map this
/// frame; without it a band left empty keeps its old cubes. That makes the
/// returned array the sole owner of its topic.
///
/// `values` and `valid` are nx*ny x-major, matching the grid.
visualization_msgs::msg::MarkerArray banded_field(
  const std::string & frame_id, const builtin_interfaces::msg::Time & stamp,
  const std::string & ns, const CostGrid & grid, const std::vector<double> & values,
  const std::vector<uint8_t> & valid, int bands, double lifetime_sec);

/// An array holding just the DELETEALL, for a frame with no field to draw.
visualization_msgs::msg::MarkerArray clear_field(
  const std::string & frame_id, const builtin_interfaces::msg::Time & stamp,
  const std::string & ns);

}  // namespace ransac_seg_cpp

#endif  // RANSAC_SEG_CPP__FIELD_MARKERS_HPP_
