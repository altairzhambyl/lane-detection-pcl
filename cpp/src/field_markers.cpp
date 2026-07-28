#include "ransac_seg_cpp/field_markers.hpp"

#include <array>
#include <vector>

#include "ransac_seg_cpp/markers.hpp"
#include "ransac_seg_cpp/planners.hpp"

namespace ransac_seg_cpp
{

visualization_msgs::msg::MarkerArray clear_field(
  const std::string & frame_id, const builtin_interfaces::msg::Time & stamp,
  const std::string & ns)
{
  visualization_msgs::msg::MarkerArray array;
  array.markers.push_back(delete_all(frame_id, stamp, ns));
  return array;
}

visualization_msgs::msg::MarkerArray banded_field(
  const std::string & frame_id, const builtin_interfaces::msg::Time & stamp,
  const std::string & ns, const CostGrid & grid, const std::vector<double> & values,
  const std::vector<uint8_t> & valid, int bands, double lifetime_sec)
{
  visualization_msgs::msg::MarkerArray array = clear_field(frame_id, stamp, ns);
  const std::vector<int32_t> band_of = normalised_bands(values, valid, bands);

  for (int band = 0; band < bands; ++band) {
    // Ascending (ix, iy), which is np.argwhere's order, so the cube list is the
    // same sequence of points on both sides.
    std::vector<std::array<int64_t, 2>> cells;
    for (int ix = 0; ix < grid.nx; ++ix) {
      for (int iy = 0; iy < grid.ny; ++iy) {
        if (band_of[grid.idx(ix, iy)] == band) {
          cells.push_back({ix, iy});
        }
      }
    }
    if (cells.empty()) {
      continue;
    }
    const std::vector<std::array<double, 3>> world = grid.cell_to_world_xyz(cells);
    std::vector<double> flat;
    flat.reserve(world.size() * 3);
    for (const std::array<double, 3> & point : world) {
      flat.insert(flat.end(), point.begin(), point.end());
    }
    array.markers.push_back(
      cube_list(
        frame_id, stamp, ns, band, flat.data(), world.size(), grid.resolution,
        band_color(band, bands), 0.05, lifetime_sec));
  }
  return array;
}

}  // namespace ransac_seg_cpp
