// The same hub carrying exactly ONE consumer, chosen by the `consumer`
// parameter.
//
// This keeps the Python topology — six processes, six copies of the perception
// prefix — available in C++, which is the only way the shared-hub claim can be
// measured head to head rather than asserted. algorithms_cpp.launch.py starts
// six of these when `separate_nodes:=true`.

#include <memory>
#include <stdexcept>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "ransac_seg_cpp/consumers.hpp"
#include "ransac_seg_cpp/perception_core.hpp"
#include "ransac_seg_cpp/ros_params.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ransac_seg_cpp::PerceptionHub>("perception_node");
  const std::string choice = ransac_seg_cpp::declare(
    *node, "consumer", std::string("ground"),
    "Which single consumer this node runs: ground, lane, astar, bfs, dfs or dijkstra");
  auto consumer = ransac_seg_cpp::make_consumer(choice);
  if (consumer == nullptr) {
    RCLCPP_FATAL(
      node->get_logger(), "unknown consumer '%s'; expected one of ground, lane, astar, bfs, "
      "dfs, dijkstra", choice.c_str());
    rclcpp::shutdown();
    return 2;
  }
  node->add_consumer(std::move(consumer));
  // Same reason as the unknown-consumer branch above: a bad launch value should
  // print why and exit, not abort out of configure() with a core dump.
  try {
    node->start();
  } catch (const std::exception & error) {
    RCLCPP_FATAL(node->get_logger(), "cannot start: %s", error.what());
    rclcpp::shutdown();
    return 2;
  }
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
