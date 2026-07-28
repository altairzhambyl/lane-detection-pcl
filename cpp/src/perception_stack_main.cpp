// The whole stack in one process: one decode/segment/rasterise per sweep,
// six consumers reading the result.
//
// This is the point of the port. algorithms.launch.py starts six processes that
// each subscribe to the same PointCloud2 and each run the identical perception
// prefix over it; bench_pipeline measures what that duplication costs.

#include <memory>
#include <stdexcept>

#include <rclcpp/rclcpp.hpp>

#include "ransac_seg_cpp/consumers.hpp"
#include "ransac_seg_cpp/perception_core.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ransac_seg_cpp::PerceptionHub>("perception_stack");
  // Registration order is the order consumers see each frame, and it mirrors
  // the order algorithms.launch.py starts the Python nodes in.
  for (const char * name : ransac_seg_cpp::kConsumerNames) {
    node->add_consumer(ransac_seg_cpp::make_consumer(name));
  }
  // start() runs every consumer's configure(), which validates its own launch
  // values and throws on a bad one. That refusal is deliberate, but an
  // unhandled throw out of main aborts with a core dump and nothing on /rosout;
  // an operator debugging a launch file needs the reason and an exit code.
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
