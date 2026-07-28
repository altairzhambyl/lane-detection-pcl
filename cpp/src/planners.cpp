#include "ransac_seg_cpp/planners.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>

namespace ransac_seg_cpp
{

namespace
{

constexpr double kInf = std::numeric_limits<double>::infinity();

std::string cell_text(Cell c)
{
  return "(" + std::to_string(c.first) + ", " + std::to_string(c.second) + ")";
}

/// The heuristic that is tight for a connectivity; throws for anything else.
///
/// Mirrors planners._heuristic, which is what makes astar reject an unsupported
/// connectivity before it ever looks at the start cell.
double heuristic_for(int connectivity, Cell a, Cell b, double resolution)
{
  if (connectivity == 8) {
    return octile(a, b, resolution);
  }
  if (connectivity == 4) {
    return manhattan(a, b, resolution);
  }
  throw std::invalid_argument(
          "connectivity must be 4 or 8, got " + std::to_string(connectivity));
}

/// A* open-set entry. Ordered by (f, ix, iy) ASCENDING.
///
/// Python pushes `(f, (ix, iy))` onto a heapq, so ties on f are broken by the
/// cell tuple. Comparing f alone would leave the choice among equal-f entries
/// to the heap's internal layout and pick a different — equally optimal, but
/// different — route than the reference implementation, which the differential
/// tests compare cell for cell.
struct OpenEntry
{
  double f;
  int ix;
  int iy;
};

struct OpenGreater
{
  bool operator()(const OpenEntry & a, const OpenEntry & b) const
  {
    if (a.f != b.f) {
      return a.f > b.f;
    }
    if (a.ix != b.ix) {
      return a.ix > b.ix;
    }
    return a.iy > b.iy;
  }
};

/// np.minimum: NaN in either operand propagates. std::min does not.
double nan_min(double a, double b)
{
  if (std::isnan(a) || std::isnan(b)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::min(a, b);
}

double nan_max(double a, double b)
{
  if (std::isnan(a) || std::isnan(b)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::max(a, b);
}

/// np.nanmax / np.nanmin over two values: skip a NaN operand entirely.
double skipnan_max(double a, double b)
{
  if (std::isnan(a)) {return b;}
  if (std::isnan(b)) {return a;}
  return std::max(a, b);
}

double skipnan_min(double a, double b)
{
  if (std::isnan(a)) {return b;}
  if (std::isnan(b)) {return a;}
  return std::min(a, b);
}

}  // namespace

double octile(Cell a, Cell b, double resolution)
{
  const double dx = std::abs(a.first - b.first);
  const double dy = std::abs(a.second - b.second);
  return resolution * ((dx + dy) + (kSqrt2 - 2.0) * std::min(dx, dy));
}

double manhattan(Cell a, Cell b, double resolution)
{
  const double dx = std::abs(a.first - b.first);
  const double dy = std::abs(a.second - b.second);
  return resolution * (dx + dy);
}

void validate_connectivity(int connectivity)
{
  if (connectivity != 4 && connectivity != 8) {
    throw std::invalid_argument(
            "connectivity must be 4 or 8, got " + std::to_string(connectivity));
  }
}

double step_length(const CostGrid & grid, Cell a, Cell b)
{
  const int dx = std::abs(a.first - b.first);
  const int dy = std::abs(a.second - b.second);
  if (dx > 1 || dy > 1 || (dx == 0 && dy == 0)) {
    throw std::invalid_argument(
            "cells " + cell_text(a) + " and " + cell_text(b) + " are not adjacent");
  }
  return grid.resolution * ((dx != 0 && dy != 0) ? kSqrt2 : 1.0);
}

double edge_cost(const CostGrid & grid, Cell a, Cell b, double step)
{
  // Widen to double before averaging: grid.cost is float32 storage, but every
  // accumulator in this file is double and the two must agree.
  const double ca = static_cast<double>(grid.cost[grid.idx(a.first, a.second)]);
  const double cb = static_cast<double>(grid.cost[grid.idx(b.first, b.second)]);
  return step * 0.5 * (ca + cb);
}

PathResult astar(const CostGrid & grid, Cell start, Cell goal, int connectivity)
{
  const double resolution = grid.resolution;
  // Probe the heuristic FIRST: an unsupported connectivity is a programming
  // error and must throw even when the start is untraversable, which would
  // otherwise short-circuit to ({}, inf) below.
  (void)heuristic_for(connectivity, start, goal, resolution);

  if (!grid.is_traversable(start.first, start.second) ||
    !grid.is_traversable(goal.first, goal.second))
  {
    return PathResult{{}, kInf};
  }

  const std::size_t n = static_cast<std::size_t>(grid.nx) * static_cast<std::size_t>(grid.ny);
  std::vector<double> best(n, kInf);
  std::vector<int32_t> parent(2 * n, -1);
  std::vector<uint8_t> closed(n, 0);

  const std::size_t start_k = grid.idx(start.first, start.second);
  const std::size_t goal_k = grid.idx(goal.first, goal.second);
  best[start_k] = 0.0;

  std::priority_queue<OpenEntry, std::vector<OpenEntry>, OpenGreater> queue;
  queue.push({heuristic_for(connectivity, start, goal, resolution), start.first, start.second});

  std::vector<Neighbor> nbs;
  while (!queue.empty()) {
    const OpenEntry entry = queue.top();
    queue.pop();
    const Cell current{entry.ix, entry.iy};
    const std::size_t ck = grid.idx(current.first, current.second);
    // Lazy deletion: stale duplicates stay in the heap and die here. Sound only
    // because the heuristic is consistent (see the header), so the first pop of
    // a cell already carries its final g value.
    if (closed[ck]) {
      continue;
    }
    closed[ck] = 1;

    if (current == goal) {
      std::vector<Cell> path;
      Cell node = goal;
      path.push_back(node);
      while (node != start) {
        const std::size_t k = grid.idx(node.first, node.second);
        node = Cell{parent[2 * k], parent[2 * k + 1]};
        path.push_back(node);
      }
      std::reverse(path.begin(), path.end());
      // The G value, not the f the goal was popped with.
      return PathResult{std::move(path), best[goal_k]};
    }

    const double current_g = best[ck];
    neighbors(grid, current.first, current.second, connectivity, nbs);
    for (const Neighbor & nb : nbs) {
      const std::size_t nk = grid.idx(nb.ix, nb.iy);
      if (closed[nk]) {
        continue;
      }
      const double tentative =
        current_g + edge_cost(grid, current, Cell{nb.ix, nb.iy}, nb.step_m);
      if (tentative < best[nk]) {  // strict: an equal-cost route never re-parents
        best[nk] = tentative;
        parent[2 * nk] = current.first;
        parent[2 * nk + 1] = current.second;
        queue.push(
          {tentative + heuristic_for(connectivity, Cell{nb.ix, nb.iy}, goal, resolution),
            nb.ix, nb.iy});
      }
    }
  }

  return PathResult{{}, kInf};
}

float DijkstraField::cost_at(int ix, int iy) const
{
  return cost[static_cast<std::size_t>(ix) * static_cast<std::size_t>(ny) + iy];
}

Cell DijkstraField::parent_at(int ix, int iy) const
{
  const std::size_t k = static_cast<std::size_t>(ix) * static_cast<std::size_t>(ny) + iy;
  return Cell{parent[2 * k], parent[2 * k + 1]};
}

DijkstraField dijkstra(const CostGrid & grid, Cell start, int connectivity)
{
  validate_connectivity(connectivity);

  const std::size_t n = static_cast<std::size_t>(grid.nx) * static_cast<std::size_t>(grid.ny);
  DijkstraField field;
  field.nx = grid.nx;
  field.ny = grid.ny;
  field.parent.assign(2 * n, -1);

  // Accumulate in double and narrow once at the end: rounding every partial sum
  // to float32 would make the field disagree with A*'s double total on the same
  // optimal path.
  std::vector<double> dist(n, kInf);

  if (!grid.is_traversable(start.first, start.second)) {
    // Empty state, never an exception: a start inside an obstacle is a normal
    // frame, not a bug.
    field.cost.assign(n, std::numeric_limits<float>::infinity());
    return field;
  }

  const std::size_t start_k = grid.idx(start.first, start.second);
  dist[start_k] = 0.0;
  field.parent[2 * start_k] = start.first;
  field.parent[2 * start_k + 1] = start.second;

  std::priority_queue<OpenEntry, std::vector<OpenEntry>, OpenGreater> queue;
  queue.push({0.0, start.first, start.second});

  std::vector<Neighbor> nbs;
  while (!queue.empty()) {
    const OpenEntry entry = queue.top();
    queue.pop();
    const Cell current{entry.ix, entry.iy};
    const std::size_t ck = grid.idx(current.first, current.second);
    // Lazy deletion; heapq exposes no decrease-key and neither does this heap.
    if (entry.f > dist[ck]) {
      continue;
    }
    neighbors(grid, current.first, current.second, connectivity, nbs);
    for (const Neighbor & nb : nbs) {
      const std::size_t nk = grid.idx(nb.ix, nb.iy);
      const double candidate =
        entry.f + edge_cost(grid, current, Cell{nb.ix, nb.iy}, nb.step_m);
      if (candidate < dist[nk]) {
        dist[nk] = candidate;
        field.parent[2 * nk] = current.first;
        field.parent[2 * nk + 1] = current.second;
        queue.push({candidate, nb.ix, nb.iy});
      }
    }
  }

  field.cost.resize(n);
  for (std::size_t k = 0; k < n; ++k) {
    field.cost[k] = static_cast<float>(dist[k]);
  }
  return field;
}

std::vector<int32_t> bfs(const CostGrid & grid, Cell start, int connectivity)
{
  validate_connectivity(connectivity);

  const std::size_t n = static_cast<std::size_t>(grid.nx) * static_cast<std::size_t>(grid.ny);
  std::vector<int32_t> hops(n, -1);
  if (!grid.is_traversable(start.first, start.second)) {
    return hops;
  }

  hops[grid.idx(start.first, start.second)] = 0;
  std::deque<Cell> queue{start};
  std::vector<Neighbor> nbs;
  while (!queue.empty()) {
    const Cell current = queue.front();
    queue.pop_front();
    const int32_t depth = hops[grid.idx(current.first, current.second)] + 1;
    neighbors(grid, current.first, current.second, connectivity, nbs);
    for (const Neighbor & nb : nbs) {
      const std::size_t nk = grid.idx(nb.ix, nb.iy);
      // Visited-on-enqueue, so nothing enters the queue twice.
      if (hops[nk] < 0) {
        hops[nk] = depth;
        queue.push_back(Cell{nb.ix, nb.iy});
      }
    }
  }
  return hops;
}

std::vector<int32_t> dfs(const CostGrid & grid, Cell start, int connectivity)
{
  validate_connectivity(connectivity);

  const std::size_t n = static_cast<std::size_t>(grid.nx) * static_cast<std::size_t>(grid.ny);
  std::vector<int32_t> order(n, -1);
  if (!grid.is_traversable(start.first, start.second)) {
    return order;
  }

  std::vector<Cell> stack{start};
  int32_t counter = 0;
  std::vector<Neighbor> nbs;
  while (!stack.empty()) {
    const Cell current = stack.back();
    stack.pop_back();
    const std::size_t ck = grid.idx(current.first, current.second);
    // The stack may hold the same cell several times; the first pop wins.
    if (order[ck] >= 0) {
      continue;
    }
    order[ck] = counter++;
    neighbors(grid, current.first, current.second, connectivity, nbs);
    for (const Neighbor & nb : nbs) {
      // Push order follows the fixed kSteps order, so the LAST neighbour
      // offered is the one explored first — the reference ordering.
      if (order[grid.idx(nb.ix, nb.iy)] < 0) {
        stack.push_back(Cell{nb.ix, nb.iy});
      }
    }
  }
  return order;
}

std::vector<Cell> reconstruct_path(const DijkstraField & field, Cell goal)
{
  if (goal.first < 0 || goal.first >= field.nx || goal.second < 0 || goal.second >= field.ny) {
    return {};
  }
  if (field.parent_at(goal.first, goal.second).first < 0) {
    return {};
  }

  std::vector<Cell> path{goal};
  Cell node = goal;
  while (true) {
    const Cell previous = field.parent_at(node.first, node.second);
    if (previous == node) {  // the source is its own parent
      break;
    }
    path.push_back(previous);
    node = previous;
  }
  std::reverse(path.begin(), path.end());
  return path;
}

double path_cost(const CostGrid & grid, const std::vector<Cell> & path)
{
  double total = 0.0;
  for (std::size_t i = 0; i + 1 < path.size(); ++i) {
    total += edge_cost(grid, path[i], path[i + 1], step_length(grid, path[i], path[i + 1]));
  }
  return total;
}

double path_length_m(const CostGrid & grid, const std::vector<Cell> & path)
{
  double total = 0.0;
  for (std::size_t i = 0; i + 1 < path.size(); ++i) {
    total += step_length(grid, path[i], path[i + 1]);
  }
  return total;
}

std::optional<std::pair<double, double>> pull_onto_grid(
  const CostGrid & grid, double x, double y, double axis_x, double axis_y)
{
  const double lo_x = grid.origin_x;
  const double lo_y = grid.origin_y;
  const double hi_x = lo_x + static_cast<double>(grid.nx) * grid.resolution;
  const double hi_y = lo_y + static_cast<double>(grid.ny) * grid.resolution;

  if (x >= lo_x && x <= hi_x && y >= lo_y && y <= hi_y) {
    return std::nullopt;
  }

  const double norm = std::hypot(axis_x, axis_y);
  if (norm < 1e-9) {
    return std::nullopt;
  }
  const double dir_x = -axis_x / norm;
  const double dir_y = -axis_y / norm;

  // Slab intersection with the divide-by-zero tolerated, exactly as numpy does
  // it under errstate(divide=ignore, invalid=ignore): an axis-aligned axis
  // gives 0/0 on the perpendicular component, so the reduction skips NaN rather
  // than the divisor being padded away.
  const double near_x = (lo_x - x) / dir_x;
  const double near_y = (lo_y - y) / dir_y;
  const double far_x = (hi_x - x) / dir_x;
  const double far_y = (hi_y - y) / dir_y;

  double entry = skipnan_max(nan_min(near_x, far_x), nan_min(near_y, far_y));
  const double exit_ = skipnan_min(nan_max(near_x, far_x), nan_max(near_y, far_y));
  entry = std::max(entry, 0.0);
  // Written in Python as `if not exit_ >= entry`, so a NaN comparison also
  // bails out; `exit_ < entry` would let a NaN through.
  if (!(exit_ >= entry)) {
    return std::nullopt;
  }

  // Step half a cell past the boundary so the landing point is inside a cell
  // rather than exactly on its edge, where floor() is ambiguous.
  const double travel = std::min(entry + 0.5 * grid.resolution, 0.5 * (entry + exit_));
  return std::make_pair(x + travel * dir_x, y + travel * dir_y);
}

std::optional<Cell> select_goal(
  const CostGrid & grid, double x, double y, double search_radius, double axis_x, double axis_y)
{
  const auto cell = snap_to_traversable(grid, x, y, search_radius);
  if (cell) {
    return cell;
  }

  const auto landing = pull_onto_grid(grid, x, y, axis_x, axis_y);
  if (!landing) {
    return std::nullopt;
  }
  return snap_to_traversable(grid, landing->first, landing->second, search_radius);
}

std::optional<Cell> furthest_reachable(
  const DijkstraField & field, const CostGrid & grid, double axis_x, double axis_y)
{
  const double norm = std::hypot(axis_x, axis_y);
  if (norm < 1e-9) {
    throw std::invalid_argument("fallback axis must have non-zero length");
  }
  const double dir_x = axis_x / norm;
  const double dir_y = axis_y / norm;

  bool found = false;
  double best_projection = 0.0;
  Cell best{0, 0};
  // ix outer, iy inner with a STRICT improvement test: ties keep the first cell
  // seen, i.e. the lexicographic minimum of (-projection, ix, iy).
  for (int ix = 0; ix < field.nx; ++ix) {
    for (int iy = 0; iy < field.ny; ++iy) {
      if (!std::isfinite(field.cost_at(ix, iy))) {
        continue;
      }
      const auto world = grid.cell_to_world(ix, iy);
      const double projection = world.first * dir_x + world.second * dir_y;
      if (!found || projection > best_projection) {
        found = true;
        best_projection = projection;
        best = Cell{ix, iy};
      }
    }
  }
  if (!found) {
    return std::nullopt;
  }
  return best;
}

std::vector<int32_t> normalised_bands(
  const std::vector<double> & values, const std::vector<uint8_t> & valid, int bands)
{
  if (valid.size() != values.size()) {
    throw std::invalid_argument("valid mask must match the value field in length");
  }
  bands = std::max(1, bands);

  std::vector<int32_t> index(values.size(), -1);
  bool any = false;
  double low = 0.0;
  double high = 0.0;
  for (std::size_t k = 0; k < values.size(); ++k) {
    // An infinite value inside `valid` is still invalid: it has no band.
    if (!valid[k] || !std::isfinite(values[k])) {
      continue;
    }
    if (!any) {
      any = true;
      low = values[k];
      high = values[k];
    } else {
      low = std::min(low, values[k]);
      high = std::max(high, values[k]);
    }
  }
  if (!any) {
    return index;
  }

  const bool flat = (high - low) < 1e-12;
  const double span = high - low;
  for (std::size_t k = 0; k < values.size(); ++k) {
    if (!valid[k] || !std::isfinite(values[k])) {
      continue;
    }
    if (flat) {
      index[k] = 0;
      continue;
    }
    // Truncation toward zero, then a clamp: without it the maximum value scales
    // to exactly `bands` and would land one band past the top.
    const int32_t raw = static_cast<int32_t>((values[k] - low) / span * bands);
    index[k] = std::min(std::max(raw, 0), bands - 1);
  }
  return index;
}

std::array<double, 4> band_color(int band, int bands, double alpha)
{
  bands = std::max(1, bands);
  // The max(1.0, bands - 1) guard makes a single-band field map to t = 0
  // instead of dividing by zero.
  double t = static_cast<double>(band) / std::max(1.0, static_cast<double>(bands - 1));
  t = std::min(std::max(t, 0.0), 1.0);
  return {t, 0.35 * (1.0 - std::fabs(2.0 * t - 1.0)) + 0.2, 1.0 - t, alpha};
}

}  // namespace ransac_seg_cpp
