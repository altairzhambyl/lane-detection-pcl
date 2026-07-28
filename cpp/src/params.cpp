#include "ransac_seg_cpp/params.hpp"

#include <cmath>
#include <cstring>

namespace ransac_seg_cpp
{

const std::array<ParamSpec, 27> kDefaults{{
  {"input_topic", ParamType::String, "PointCloud2 topic to subscribe to"},
  {"marker_topic", ParamType::String, "Topic for this node's visualisation markers"},
  {"qos_depth", ParamType::Int, "Queue depth for the cloud subscription and marker publisher"},
  {"frame_id_override", ParamType::String,
    "Frame to stamp outgoing markers with; empty means use the incoming cloud frame"},
  {"resolution", ParamType::Double, "Costmap cell size in metres"},
  {"robot_radius", ParamType::Double, "Cells within this distance of an obstacle are untraversable"},
  {"inflation_radius", ParamType::Double,
    "Distance in metres over which obstacle proximity still raises cost"},
  {"cost_scaling", ParamType::Double, "Exponential decay rate of the inflation cost falloff"},
  {"min_ground_points", ParamType::Int, "Ground returns needed in a cell before it counts as free"},
  {"min_obstacle_points", ParamType::Int,
    "Obstacle returns needed in a cell before it counts as occupied"},
  {"ray_fill", ParamType::Bool,
    "Carve cells the beams crossed as free, closing the gaps between rings"},
  {"ray_fill_bins", ParamType::Int, "Azimuth bins used when carving free space"},
  {"ray_fill_min_cells", ParamType::Int, "Ground cells an azimuth needs before it may carve a ray"},
  {"min_range", ParamType::Double, "Drop returns closer than this; rejects the sensor's own mount"},
  {"max_range", ParamType::Double, "Drop returns beyond this range in metres"},
  {"marker_lifetime_sec", ParamType::Double, "Marker lifetime in seconds; 0 means never expire"},
  {"distance_threshold", ParamType::Double,
    "RANSAC inlier band around the plane in metres; 0.20 was chosen on "
    "SemanticKITTI 00 and 05, not on synthetic data"},
  {"max_iterations", ParamType::Int, "Upper bound on RANSAC hypotheses per frame"},
  {"min_iterations", ParamType::Int, "Hypotheses always drawn before the adaptive budget applies"},
  {"confidence", ParamType::Double, "Target probability of drawing one all-inlier sample"},
  {"max_tilt_deg", ParamType::Double, "Reject planes tilted further than this from +z"},
  {"min_inlier_ratio", ParamType::Double, "Below this inlier fraction the fit is treated as failed"},
  {"max_scoring_points", ParamType::Int, "Cloud subsample size used to score hypotheses"},
  {"fallback_height", ParamType::Double, "Fixed z threshold used when no plane is found"},
  {"plane_filter_alpha", ParamType::Double,
    "Blend weight of a new plane estimate against the held one"},
  {"plane_max_tilt_jump_deg", ParamType::Double,
    "Reject a plane estimate tilting more than this from the held plane"},
  {"plane_max_offset_jump", ParamType::Double,
    "Reject a plane estimate whose offset moves more than this in metres"},
}};

namespace
{

template<typename T>
struct Field
{
  const char * name;
  T CommonParams::* member;
};

// Member tables rather than a 27-way if-chain: the compiler checks each member
// against its declared type, so a field can only be wired to the wrong kind of
// parameter by also changing the struct.
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

template<typename T, std::size_t N>
T CommonParams::* lookup(const Field<T>(&table)[N], const std::string & name)
{
  for (const auto & field : table) {
    if (name == field.name) {
      return field.member;
    }
  }
  return nullptr;
}

const char * value_kind(const ParamValue & value)
{
  switch (value.index()) {
    case 0: return "a bool";
    case 1: return "an integer";
    case 2: return "a double";
    default: return "a string";
  }
}

}  // namespace

const CommonParams & default_params()
{
  static const CommonParams kParams{};
  return kParams;
}

bool is_construction_only(std::string_view name)
{
  return name == "input_topic" || name == "marker_topic" || name == "qos_depth";
}

const ParamSpec * find_param(std::string_view name)
{
  for (const auto & spec : kDefaults) {
    if (name == spec.name) {
      return &spec;
    }
  }
  return nullptr;
}

std::string apply_param(CommonParams & params, const std::string & name, const ParamValue & value)
{
  const ParamSpec * spec = find_param(name);
  if (spec == nullptr) {
    // Matches the Python node's refusal text, which it prefixes with the name.
    return "cannot be changed while the node is running";
  }

  switch (spec->type) {
    case ParamType::String: {
      const std::string * text = std::get_if<std::string>(&value);
      if (text == nullptr) {
        return std::string("expects a string, got ") + value_kind(value);
      }
      params.*lookup(kStringFields, name) = *text;
      return "";
    }
    case ParamType::Bool: {
      bool result = false;
      if (const bool * b = std::get_if<bool>(&value)) {
        result = *b;
      } else if (const int64_t * i = std::get_if<int64_t>(&value)) {
        result = (*i != 0);
      } else if (const double * d = std::get_if<double>(&value)) {
        result = (*d != 0.0);
      } else {
        return "expects a bool, got a string";
      }
      params.*lookup(kBoolFields, name) = result;
      return "";
    }
    case ParamType::Int: {
      int result = 0;
      if (const bool * b = std::get_if<bool>(&value)) {
        result = *b ? 1 : 0;
      } else if (const int64_t * i = std::get_if<int64_t>(&value)) {
        result = static_cast<int>(*i);
      } else if (const double * d = std::get_if<double>(&value)) {
        if (!std::isfinite(*d)) {
          return "expects a finite integer, got a non-finite double";
        }
        result = static_cast<int>(*d);  // truncates toward zero, like Python int()
      } else {
        return "expects an integer, got a string";
      }
      params.*lookup(kIntFields, name) = result;
      return "";
    }
    case ParamType::Double: {
      double result = 0.0;
      if (const bool * b = std::get_if<bool>(&value)) {
        result = *b ? 1.0 : 0.0;
      } else if (const int64_t * i = std::get_if<int64_t>(&value)) {
        result = static_cast<double>(*i);
      } else if (const double * d = std::get_if<double>(&value)) {
        result = *d;
      } else {
        return "expects a double, got a string";
      }
      params.*lookup(kDoubleFields, name) = result;
      return "";
    }
  }
  return "cannot be changed while the node is running";
}

RansacParams ransac_params(const CommonParams & params)
{
  RansacParams out;
  out.distance_threshold = params.distance_threshold;
  out.max_iterations = params.max_iterations;
  out.min_iterations = params.min_iterations;
  out.confidence = params.confidence;
  out.max_tilt_deg = params.max_tilt_deg;
  out.min_inlier_ratio = params.min_inlier_ratio;
  out.max_scoring_points = params.max_scoring_points;
  return out;
}

}  // namespace ransac_seg_cpp
