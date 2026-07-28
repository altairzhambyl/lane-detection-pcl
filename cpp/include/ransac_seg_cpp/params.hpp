// Shared node parameters — C++ port of ransac_seg/params.py.
//
// Topics, grid resolution and the whole RANSAC configuration used to be
// literals buried in six callbacks, so retuning for a different sensor meant
// editing (and diverging) six files. Declaring them once here means every node
// exposes the same names and the defaults below are the single place where
// "today's behaviour" is written down.
//
// No rclcpp dependency: the struct and the coercion rules are what the port
// needs, and keeping them ROS-free lets the planner and lane tests link them on
// a bare checkout, exactly as the Python module does.

#ifndef RANSAC_SEG_CPP__PARAMS_HPP_
#define RANSAC_SEG_CPP__PARAMS_HPP_

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>

#include "ransac_seg_cpp/ground_plane.hpp"

namespace ransac_seg_cpp
{

/// Every tunable a perception node reads, resolved once per node start-up.
///
/// These are exactly the 27 keys of ransac_seg/params.py's DEFAULTS dict, in
/// that dict's order, with the same default values. kDefaults below carries the
/// same order, so the two can be zipped when declaring parameters.
struct CommonParams
{
  std::string input_topic = "/sensor/lidar_front/points";
  std::string marker_topic = "/semantic_map/squares";
  int qos_depth = 10;
  std::string frame_id_override = "";
  double resolution = 0.5;
  double robot_radius = 0.6;
  double inflation_radius = 1.5;
  double cost_scaling = 3.0;
  int min_ground_points = 1;
  int min_obstacle_points = 3;
  bool ray_fill = true;
  int ray_fill_bins = 720;
  int ray_fill_min_cells = 2;
  double min_range = 0.5;
  double max_range = 60.0;
  double marker_lifetime_sec = 0.0;
  double distance_threshold = 0.20;
  int max_iterations = 200;
  int min_iterations = 12;
  double confidence = 0.99;
  double max_tilt_deg = 10.0;
  double min_inlier_ratio = 0.15;
  int max_scoring_points = 20000;
  double fallback_height = 0.2;
  double plane_filter_alpha = 0.3;
  double plane_max_tilt_jump_deg = 5.0;
  double plane_max_offset_jump = 0.5;
};

/// The type a parameter's declared default fixes; that default is the authority.
enum class ParamType
{
  Bool,
  Int,
  Double,
  String
};

/// One declarable parameter: name, intended type, and the ParameterDescriptor
/// description string the Python module ships.
struct ParamSpec
{
  const char * name;
  ParamType type;
  const char * description;
};

/// The 27 DEFAULTS entries, in ransac_seg/params.py order.
extern const std::array<ParamSpec, 27> kDefaults;

/// The defaults as a struct, for callers that want a baseline to override.
const CommonParams & default_params();

/// True for the parameters consumed once while the node is being built.
///
/// Rewriting a subscribed topic or a queue depth on a live node cannot change
/// anything, so the runtime callback refuses them instead of acknowledging a
/// change that will not happen.
bool is_construction_only(std::string_view name);

/// A parameter value as a ROS parameter server would hand it over.
using ParamValue = std::variant<bool, int64_t, double, std::string>;

/// Coerce `value` to the declared type of `name` and store it in `params`.
///
/// Returns "" on success, otherwise the reason the node should report; the node
/// prepends the parameter name, matching Python's `f"'{name}': {error}"`.
///
/// The coercion exists because a launch file routinely hands an int where a
/// double is wanted and rclcpp keeps it an int, which then propagates into the
/// maths as one. int -> double widens, double -> int truncates toward zero,
/// bool and int convert both ways, strings pass through untouched.
///
/// DECLARED DEVIATION from params.py's coerce(): Python is dynamically typed,
/// so `bool("false")` is True and a string reaching an int field either parses
/// or raises. Here a string handed to a numeric or bool field is rejected with
/// a message rather than parsed — silently reading the string "false" as true
/// on a live vehicle is worse than refusing the set.
std::string apply_param(CommonParams & params, const std::string & name, const ParamValue & value);

/// The spec for `name`, or nullptr when it is not a DEFAULTS key.
const ParamSpec * find_param(std::string_view name);

/// The subset of `params` that fit_plane_ransac/segment_ground accept.
///
/// `fallback_height` is NOT part of this: ground_plane.hpp already takes it as
/// a separate argument to segment_ground, so it is passed at the call site.
/// `seed` and `refine` stay at their RansacParams defaults, as in Python.
RansacParams ransac_params(const CommonParams & params);

}  // namespace ransac_seg_cpp

#endif  // RANSAC_SEG_CPP__PARAMS_HPP_
