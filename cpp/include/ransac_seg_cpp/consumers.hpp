// Factories for the six frame consumers, one per Python node.
//
// The consumer classes themselves stay in their .cpp files: nothing outside
// needs their types, and keeping them private means a main() or a test can
// assemble any subset of the stack without pulling in every planner's headers.
//
// PARAMETER NAMING, DECLARED DEVIATION FROM THE PYTHON: each consumer's own
// parameters live under its name() prefix, e.g. `astar.connectivity`, because
// four of the six Python nodes declare a parameter called `connectivity` and
// three declare `field_bands` — in one process those names collide. The
// parameter names after the prefix are exactly the Python ones, so a launch
// file translates mechanically.

#ifndef RANSAC_SEG_CPP__CONSUMERS_HPP_
#define RANSAC_SEG_CPP__CONSUMERS_HPP_

#include <array>
#include <memory>
#include <string>

#include "ransac_seg_cpp/perception_core.hpp"

namespace ransac_seg_cpp
{

/// Tri-state costmap tiles plus clustered obstacle boxes; from ransac_node.py.
std::unique_ptr<FrameConsumer> make_ground_segmentation();

/// Fitted road edges and centreline; from curve.py. Wants no costmap.
std::unique_ptr<FrameConsumer> make_lane_boundary();

/// One least-cost route to a parameterised goal; from astar.py.
std::unique_ptr<FrameConsumer> make_astar_planner();

/// The weighted cost-from-ego field; from dijkstra.py.
std::unique_ptr<FrameConsumer> make_dijkstra_planner();

/// The unweighted hop field; from bfs.py.
std::unique_ptr<FrameConsumer> make_bfs_planner();

/// Depth-first discovery order, audited against BFS; from dfs.py.
std::unique_ptr<FrameConsumer> make_dfs_baseline();

/// The six consumer names, in the order algorithms.launch.py starts them.
inline constexpr std::array<const char *, 6> kConsumerNames{
  "ground", "lane", "astar", "bfs", "dfs", "dijkstra"};

/// Build one consumer by name, or nullptr when the name is not one of the six.
std::unique_ptr<FrameConsumer> make_consumer(const std::string & name);

}  // namespace ransac_seg_cpp

#endif  // RANSAC_SEG_CPP__CONSUMERS_HPP_
