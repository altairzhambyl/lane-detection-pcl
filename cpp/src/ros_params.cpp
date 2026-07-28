#include "ransac_seg_cpp/ros_params.hpp"

#include <cmath>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <rcl_interfaces/msg/parameter_descriptor.hpp>

namespace ransac_seg_cpp
{
namespace
{

rcl_interfaces::msg::ParameterDescriptor describe(const std::string & description)
{
  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.description = description;
  // Dynamic typing so THIS package decides what a type mismatch means. rclcpp's
  // static check refuses `-p resolution:=1` outright — an int where a double is
  // declared — and throws out of the constructor, where the Python node accepts
  // it and coerces. The coercion rules in params.hpp are the shared authority
  // for both, and validate_and_apply still refuses anything it cannot coerce,
  // with a reason naming the parameter.
  descriptor.dynamic_typing = true;
  return descriptor;
}

/// Widen an rclcpp::Parameter into the ROS-free ParamValue params.hpp coerces.
///
/// Byte arrays, string arrays and the other array types have no shared
/// parameter that accepts them, so they arrive here as NOT_SET and are refused
/// by apply_param with a type message rather than silently ignored.
ParamValue to_param_value(const rclcpp::Parameter & parameter)
{
  switch (parameter.get_type()) {
    case rclcpp::ParameterType::PARAMETER_BOOL:
      return ParamValue{parameter.as_bool()};
    case rclcpp::ParameterType::PARAMETER_INTEGER:
      return ParamValue{parameter.as_int()};
    case rclcpp::ParameterType::PARAMETER_DOUBLE:
      return ParamValue{parameter.as_double()};
    case rclcpp::ParameterType::PARAMETER_STRING:
      return ParamValue{parameter.as_string()};
    default:
      // apply_param reports the mismatch against the declared type, which is a
      // better message than anything this function could invent.
      return ParamValue{std::string("<unsupported type>")};
  }
}

/// Coerce a runtime value for min_ground_points_frame, or return the reason.
std::string coerce_frame_minimum(const ParamValue & value, int & out)
{
  if (const bool * b = std::get_if<bool>(&value)) {
    out = *b ? 1 : 0;
    return "";
  }
  if (const int64_t * i = std::get_if<int64_t>(&value)) {
    out = static_cast<int>(*i);
    return "";
  }
  if (const double * d = std::get_if<double>(&value)) {
    if (!std::isfinite(*d)) {
      return "expects a finite integer, got a non-finite double";
    }
    out = static_cast<int>(*d);
    return "";
  }
  return "expects an integer, got a string";
}

// Member pointers for the four declarable types. params.hpp owns the default
// VALUES; this table only says which field each declared name reaches, which is
// the one thing a struct cannot tell rclcpp about itself in C++17. The
// static_assert below fails the build if the two ever drift apart in count.
template<typename T>
struct Field
{
  const char * name;
  T CommonParams::* member;
};

constexpr Field<std::string> kStringFields[] = {
  {"input_topic", &CommonParams::input_topic},
  {"marker_topic", &CommonParams::marker_topic},
  {"frame_id_override", &CommonParams::frame_id_override},
};

constexpr Field<bool> kBoolFields[] = {
  {"ray_fill", &CommonParams::ray_fill},
};

constexpr Field<int> kIntFields[] = {
  {"qos_depth", &CommonParams::qos_depth},
  {"min_ground_points", &CommonParams::min_ground_points},
  {"min_obstacle_points", &CommonParams::min_obstacle_points},
  {"ray_fill_bins", &CommonParams::ray_fill_bins},
  {"ray_fill_min_cells", &CommonParams::ray_fill_min_cells},
  {"max_iterations", &CommonParams::max_iterations},
  {"min_iterations", &CommonParams::min_iterations},
  {"max_scoring_points", &CommonParams::max_scoring_points},
};

constexpr Field<double> kDoubleFields[] = {
  {"resolution", &CommonParams::resolution},
  {"robot_radius", &CommonParams::robot_radius},
  {"inflation_radius", &CommonParams::inflation_radius},
  {"cost_scaling", &CommonParams::cost_scaling},
  {"min_range", &CommonParams::min_range},
  {"max_range", &CommonParams::max_range},
  {"marker_lifetime_sec", &CommonParams::marker_lifetime_sec},
  {"distance_threshold", &CommonParams::distance_threshold},
  {"confidence", &CommonParams::confidence},
  {"max_tilt_deg", &CommonParams::max_tilt_deg},
  {"min_inlier_ratio", &CommonParams::min_inlier_ratio},
  {"fallback_height", &CommonParams::fallback_height},
  {"plane_filter_alpha", &CommonParams::plane_filter_alpha},
  {"plane_max_tilt_jump_deg", &CommonParams::plane_max_tilt_jump_deg},
  {"plane_max_offset_jump", &CommonParams::plane_max_offset_jump},
};

static_assert(
  std::size(kStringFields) + std::size(kBoolFields) + std::size(kIntFields) +
  std::size(kDoubleFields) == std::tuple_size<decltype(kDefaults)>::value,
  "ros_params.cpp lost track of a params.hpp field");

template<typename T, std::size_t N>
T default_of(const Field<T> (& table)[N], std::string_view name)
{
  for (const Field<T> & field : table) {
    if (name == field.name) {
      return default_params().*(field.member);
    }
  }
  throw std::logic_error("no default for parameter '" + std::string(name) + "'");
}

}  // namespace

namespace
{

/// Declare `name` once, with dynamic typing, and leave reading to the caller.
///
/// Deliberately not rclcpp's templated declare_parameter: that one ends in
/// `.get<T>()` on the resolved value, which throws when a launch file supplied
/// an int for a double-defaulted parameter — the exact case the coercion rules
/// in params.hpp exist to absorb.
void ensure_declared(
  rclcpp::Node & node, const std::string & name, const rclcpp::ParameterValue & default_value,
  const std::string & description)
{
  if (!node.has_parameter(name)) {
    node.declare_parameter(name, default_value, describe(description));
  }
}

}  // namespace

bool declare(
  rclcpp::Node & node, const std::string & name, bool default_value,
  const std::string & description)
{
  ensure_declared(node, name, rclcpp::ParameterValue(default_value), description);
  const rclcpp::Parameter parameter = node.get_parameter(name);
  if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER) {
    return parameter.as_int() != 0;
  }
  return parameter.as_bool();
}

int64_t declare(
  rclcpp::Node & node, const std::string & name, int64_t default_value,
  const std::string & description)
{
  ensure_declared(node, name, rclcpp::ParameterValue(default_value), description);
  const rclcpp::Parameter parameter = node.get_parameter(name);
  if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE) {
    // Rounds toward zero, like Python's int().
    return static_cast<int64_t>(parameter.as_double());
  }
  return parameter.as_int();
}

double declare(
  rclcpp::Node & node, const std::string & name, double default_value,
  const std::string & description)
{
  ensure_declared(node, name, rclcpp::ParameterValue(default_value), description);
  const rclcpp::Parameter parameter = node.get_parameter(name);
  // A launch file routinely hands an int where a double is wanted and rclcpp
  // keeps it an int; the declared default is the authority on the type.
  if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER) {
    return static_cast<double>(parameter.as_int());
  }
  return parameter.as_double();
}

std::string declare(
  rclcpp::Node & node, const std::string & name, const std::string & default_value,
  const std::string & description)
{
  ensure_declared(node, name, rclcpp::ParameterValue(default_value), description);
  return node.get_parameter(name).as_string();
}

std::vector<double> declare(
  rclcpp::Node & node, const std::string & name, const std::vector<double> & default_value,
  const std::string & description)
{
  ensure_declared(node, name, rclcpp::ParameterValue(default_value), description);
  const rclcpp::Parameter parameter = node.get_parameter(name);
  if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER_ARRAY) {
    std::vector<double> widened;
    for (const int64_t value : parameter.as_integer_array()) {
      widened.push_back(static_cast<double>(value));
    }
    return widened;
  }
  return parameter.as_double_array();
}

void declare_common_params(rclcpp::Node & node)
{
  for (const ParamSpec & spec : kDefaults) {
    const std::string name = spec.name;
    if (node.has_parameter(name)) {
      continue;
    }
    switch (spec.type) {
      case ParamType::Bool:
        declare(node, name, default_of(kBoolFields, name), spec.description);
        break;
      case ParamType::Int:
        declare(
          node, name, static_cast<int64_t>(default_of(kIntFields, name)), spec.description);
        break;
      case ParamType::Double:
        declare(node, name, default_of(kDoubleFields, name), spec.description);
        break;
      case ParamType::String:
        declare(node, name, default_of(kStringFields, name), spec.description);
        break;
    }
  }
}

CommonParams read_common_params(const rclcpp::Node & node)
{
  CommonParams params = default_params();
  for (const ParamSpec & spec : kDefaults) {
    const std::string name = spec.name;
    if (!node.has_parameter(name)) {
      continue;
    }
    // apply_param carries the coercion rules, so a launch file's int lands in a
    // double field here exactly as it would through a runtime set.
    apply_param(params, name, to_param_value(node.get_parameter(name)));
  }
  return params;
}

rcl_interfaces::msg::SetParametersResult validate_and_apply(
  const std::vector<rclcpp::Parameter> & request, const rclcpp::Node & node,
  CommonParams & params, int & min_ground_points_frame, PlaneFilter & filter)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;

  // Staged copies: nothing the caller can observe is touched until the whole
  // request has passed, so a rejected batch leaves the node as it was.
  CommonParams staged = params;
  int staged_minimum = min_ground_points_frame;
  bool touched_common = false;
  bool touched_minimum = false;

  for (const rclcpp::Parameter & parameter : request) {
    const std::string & name = parameter.get_name();
    if (!node.has_parameter(name)) {
      // A declaration, not a set: the parameter does not exist yet.
      continue;
    }
    if (is_construction_only(name)) {
      result.successful = false;
      result.reason = "'" + name + "' is read once at start-up; restart the node";
      return result;
    }
    const ParamValue value = to_param_value(parameter);
    if (name == "min_ground_points_frame") {
      const std::string error = coerce_frame_minimum(value, staged_minimum);
      if (!error.empty()) {
        result.successful = false;
        result.reason = "'" + name + "': " + error;
        return result;
      }
      touched_minimum = true;
      continue;
    }
    if (find_param(name) == nullptr) {
      result.successful = false;
      result.reason = "'" + name + "' cannot be changed while the node is running";
      return result;
    }
    const std::string error = apply_param(staged, name, value);
    if (!error.empty()) {
      result.successful = false;
      result.reason = "'" + name + "': " + error;
      return result;
    }
    touched_common = true;
  }

  if (touched_minimum) {
    min_ground_points_frame = staged_minimum;
  }
  if (touched_common) {
    params = staged;
    filter.alpha = params.plane_filter_alpha;
    filter.max_tilt_jump_deg = params.plane_max_tilt_jump_deg;
    filter.max_offset_jump = params.plane_max_offset_jump;
  }
  return result;
}

}  // namespace ransac_seg_cpp
