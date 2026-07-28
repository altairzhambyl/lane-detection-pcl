// rclcpp binding for the ROS-free parameter model in params.hpp.
//
// params.hpp deliberately knows nothing about ROS so the algorithm tests link
// on a bare checkout. This file is the only place that knows both halves: it
// declares the 27 shared parameters on a node, reads them back into a
// CommonParams, and turns an incoming `ros2 param set` batch into either an
// applied change or a refusal carrying a reason.

#ifndef RANSAC_SEG_CPP__ROS_PARAMS_HPP_
#define RANSAC_SEG_CPP__ROS_PARAMS_HPP_

#include <cstdint>
#include <string>
#include <vector>

#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/rclcpp.hpp>

#include "ransac_seg_cpp/params.hpp"
#include "ransac_seg_cpp/plane_filter.hpp"

namespace ransac_seg_cpp
{

/// Declare one parameter (idempotently) and return its resolved value.
///
/// One overload per declarable type rather than a template, because the
/// declared default is the authority on the intended type and an implicit
/// conversion at the call site is exactly the bug this guards against: a
/// launch file handing an int where a double is wanted.
bool declare(
  rclcpp::Node & node, const std::string & name, bool default_value,
  const std::string & description);
int64_t declare(
  rclcpp::Node & node, const std::string & name, int64_t default_value,
  const std::string & description);
double declare(
  rclcpp::Node & node, const std::string & name, double default_value,
  const std::string & description);
std::string declare(
  rclcpp::Node & node, const std::string & name, const std::string & default_value,
  const std::string & description);
std::vector<double> declare(
  rclcpp::Node & node, const std::string & name, const std::vector<double> & default_value,
  const std::string & description);

/// Declare every shared parameter on `node` with its documented default.
void declare_common_params(rclcpp::Node & node);

/// Read the declared shared parameters off `node` into a CommonParams.
CommonParams read_common_params(const rclcpp::Node & node);

/// Apply a runtime parameter batch, or refuse the whole batch with a reason.
///
/// ALL-OR-NOTHING. The whole request is validated before anything is mutated,
/// so a rejected batch leaves the node byte-identical. Without this,
/// `ros2 param set` reported success for every shared parameter while the node
/// kept running on the snapshot it took in its constructor — an operator
/// retuning inflation or range on a live vehicle got an acknowledgement and no
/// behaviour change.
///
/// Names the node does not have are skipped: rclcpp runs on-set callbacks for
/// declarations too, and a declaration is not a set.
rcl_interfaces::msg::SetParametersResult validate_and_apply(
  const std::vector<rclcpp::Parameter> & request, const rclcpp::Node & node,
  CommonParams & params, int & min_ground_points_frame, PlaneFilter & filter);

}  // namespace ransac_seg_cpp

#endif  // RANSAC_SEG_CPP__ROS_PARAMS_HPP_
