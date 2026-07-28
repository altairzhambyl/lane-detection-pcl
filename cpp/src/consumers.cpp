#include "ransac_seg_cpp/consumers.hpp"

#include <memory>
#include <string>

namespace ransac_seg_cpp
{

std::unique_ptr<FrameConsumer> make_consumer(const std::string & name)
{
  if (name == "ground") {
    return make_ground_segmentation();
  }
  if (name == "lane") {
    return make_lane_boundary();
  }
  if (name == "astar") {
    return make_astar_planner();
  }
  if (name == "bfs") {
    return make_bfs_planner();
  }
  if (name == "dfs") {
    return make_dfs_baseline();
  }
  if (name == "dijkstra") {
    return make_dijkstra_planner();
  }
  return nullptr;
}

}  // namespace ransac_seg_cpp
