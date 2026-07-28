# Architecture

Every stage below exists in code. Where a number appears it was measured on the
synthetic scenes described in the README, never on real sensor data.

## Pipeline Overview

```
LiDAR sensor  ──►  /sensor/lidar_front/points   (sensor_msgs/PointCloud2)
                              │
                              ▼
              ┌───────────────────────────────┐
              │ cloud_io.pointcloud2_to_xyz   │  strided numpy view, no per-point loop
              └───────────────┬───────────────┘
                              ▼
              ┌───────────────────────────────┐
              │ perception_node._range_gate   │  min_range..max_range, x > 0
              └───────────────┬───────────────┘
                              ▼
              ┌────────────────────────────────┐
              │ ground_plane.segment_ground    │  from-scratch RANSAC plane fit
              │  3-point hypotheses, tilt gate │  falls back to a fixed height and
              │  SVD refit on the consensus set│  warns when no plane clears the floor
              └───────────────┬────────────────┘
                              ▼
              ┌────────────────────────────────┐
              │ plane_filter.PlaneFilter       │  blends across frames, gates jumps,
              │                                │  holds the last plane through misses
              └───────────────┬────────────────┘
                  ┌───────────┴────────────┐
                  ▼                        ▼
      ground returns                non-ground returns
                  │                        │
                  │                        ▼
                  │            ┌───────────────────────────┐
                  │            │ cluster.cluster_obstacles │  voxel + KD-tree +
                  │            │                           │  connected components
                  │            └───────────┬───────────────┘
                  ▼                        │
      ┌───────────────────────┐            │
      │ lane.fit_lane         │            │
      │ centreline-frame fit, │            │
      │ percentile edges,     │            │
      │ spline resample       │            │
      └───────────┬───────────┘            │
                  │        ┌───────────────┘
                  │        ▼
                  │   ┌────────────────────────────────────┐
                  └──►│ grid.build_costmap                 │
                      │ tri-state: FREE / OCCUPIED /       │
                      │ INFLATED, EDT-based inflation,     │
                      │ float32 cost field                 │
                      └───────────────┬────────────────────┘
                                      ▼
                      ┌────────────────────────────────────┐
                      │ planners.py over the CostGrid      │
                      │ astar / dijkstra / bfs / dfs       │
                      └───────────────┬────────────────────┘
                                      ▼
                      ┌────────────────────────────────────┐
                      │ markers.py — every marker's z comes│
                      │ from the fitted plane, not z = 0   │
                      └────────────────────────────────────┘
```

## Modules

| Module | ROS import? | Responsibility |
|---|---|---|
| `cloud_io.py` | yes | `PointCloud2` ↔ numpy, both directions vectorised |
| `ground_plane.py` | no | RANSAC plane fit; `segment_ground()` returns `(mask, model_or_None)` |
| `plane_filter.py` | no | Temporal smoothing and jump rejection of the plane estimate |
| `cluster.py` | no | Voxel downsample → KD-tree pairs → connected components → `Cluster` list |
| `lane.py` | no | Road boundary fit in the centreline frame; always returns a `LaneFit` |
| `grid.py` | no | `CostGrid`, `build_costmap`, `snap_to_traversable`, `plane_z` |
| `planners.py` | no | `astar`, `dijkstra`, `bfs`, `dfs`, goal selection, colour banding |
| `markers.py` | yes | Marker constructors; empty geometry becomes an explicit clear, never an empty `ADD` |
| `params.py` | yes | The 24 shared ROS parameters, their defaults and their types |
| `perception_node.py` | yes | `PerceptionNode` base: decode → gate → segment → filter → rasterise → `on_frame` |
| `point_files.py` | no | `.bin` (KITTI) and `.pcd` readers for offline data |

Only four of the eleven import ROS, so the geometry and search code is testable
without a ROS installation — the numpy-only part of the suite runs on a bare
checkout.

## Nodes

Every node subclasses `PerceptionNode`, so they share one decode/segment path
and differ only in `on_frame`.

| Executable | Class | Node name | Publishes |
|---|---|---|---|
| `ransac_node` | `GroundSegmentationNode` | `ground_segmentation_node` | `/semantic_map/squares` (`Marker`, ns `ground`/`obstacles`/`inflation`), `/semantic_map/obstacles` (`MarkerArray`, one box per cluster) |
| `curve` | `LaneBoundaryNode` | `lane_boundary_node` | `/semantic_map/road_left`, `/semantic_map/road_right`, `/semantic_map/road_center` (`Marker`, LINE_STRIP) |
| `astar` | `AStarPlanner` | `astar_planner_node` | `/planning/astar_path` (`Marker`, LINE_STRIP) |
| `dijkstra` | `DijkstraPlanner` | `dijkstra_planner_node` | `/planning/dijkstra_cost` (`MarkerArray`, one CUBE_LIST per cost band) |
| `bfs` | `BFSPlanner` | `bfs_planner_node` | `/planning/bfs_hops` (`MarkerArray`, one CUBE_LIST per hop band) |
| `dfs` | `DFSBaseline` | `dfs_baseline_node` | `/planning/dfs_order` (`MarkerArray`, one CUBE_LIST per discovery-order band) |

All six subscribe to the `input_topic` parameter, `/sensor/lidar_front/points`
by default.

`dfs` is a baseline, not a fourth planner. Depth-first search reaches exactly
the set breadth-first search reaches — connectivity is a property of the graph,
not of the walk order — so it yields no path and no cost, only discovery order.
The node re-runs BFS on the same grid every second and logs whether the two
visited sets match, so the claim is measured rather than asserted.

## Data flow, stage by stage

1. **Decode.** `pointcloud2_to_xyz` reads the message as one strided numpy view.
   Measured on a 120k-point cloud this session: **1.6 ms**, against **206 ms**
   for the per-point `read_points` loop the nodes used to carry (identical
   output, max absolute difference 0). Every figure in this section is printed
   by `python3 tools/benchmarks.py`, which defines each scene in full.
2. **Range gate.** Returns outside `[min_range, max_range]` or behind the sensor
   (`x <= 0`) are dropped. Rear returns are a different piece of road; pooling
   them into one plane fit tilts it.
3. **Ground segmentation.** `segment_ground` fits a plane by RANSAC: minimal
   3-point hypotheses, degeneracy rejection, consensus scored against a capped
   random subsample, an adaptive iteration budget, a tilt gate so a wall cannot
   win over the road, and an SVD total-least-squares refit on the winning
   consensus set. On a synthetic 120k-point cloud at 10° slope with 35 %
   non-ground returns, measured this session: **recall 1.000 / precision 1.000
   in 16.0 ms**; the fixed `z < 0.2` cut on the same cloud gives **recall 0.717 /
   precision 0.744**, because at 10° the surface climbs past 0.2 m within about
   11 m of the sensor. When no plane clears the inlier floor the node warns and
   falls back to the fixed height rather than publishing nothing.
4. **Temporal filter.** `PlaneFilter` blends each new estimate into the held one
   and rejects tilt/offset jumps, holding the previous plane through a bounded
   run of misses before resetting.
5. **Clustering.** Non-ground returns are voxelised, linked with a KD-tree
   radius query and labelled by connected components, then gated on point count
   and z-extent. On a synthetic three-parked-cars scene: **3 clusters from 2700
   non-ground returns in 5.2 ms**, recovered extents 4.19 × 1.80 m against
   4.20 × 1.80 m planted.
6. **Costmap.** `build_costmap` rasterises ground *and* non-ground returns into
   a tri-state grid: `FREE` where ground was seen, `OCCUPIED` where enough
   non-ground returns landed (occupied always beats free), `INFLATED` within
   `robot_radius` of an obstacle, `UNKNOWN` elsewhere. A Euclidean distance
   transform in metres drives an exponential cost falloff out to
   `inflation_radius`; every finite cost is ≥ 1.0 so the A* heuristic stays
   admissible. The transform measures cell centre to cell centre, so it is
   shortened by half a cell diagonal before use: the return that made a cell
   `OCCUPIED` sits anywhere inside it, and taking the raw distance as clearance
   overstated it by up to `resolution × √2 / 2` — 0.35 m at the default 0.5 m
   resolution, against a 0.6 m robot radius. Measured on the parked-car scene:
   **2.7 ms** for an 81 × 19 grid (1132 free, 108 occupied, 51 inflated,
   248 unknown), free-cell cost spanning 1.000–1.870.
7. **Lane fit.** `fit_lane` seeds a heading by PCA, refines a polyline
   centreline against per-bin median lateral offsets, takes percentile edges per
   arc-length bin, truncates at gaps and resamples through a spline. It always
   returns a `LaneFit`; an invalid one carries a reason and the node publishes a
   clear marker instead of a boundary.
8. **Planning.** All four searches run over the same `CostGrid` and the same
   edge cost, `step_length_m × 0.5 × (cost[a] + cost[b])`. The start cell is
   always the ego origin. A* uses octile distance for 8-connectivity and
   Manhattan for 4 — never Euclidean, which is inadmissible on a grid.
   Measured on the parked-car costmap: **A\* total path cost 39.000000 and the
   Dijkstra cost field at the same goal 39.000000** (difference 0), and DFS and
   BFS visit the identical 1132 cells. A diagonal step is only offered when both
   cells it squeezes past are traversable; without that an 8-connected search
   walked through the shared corner of two diagonally touching blocked cells,
   i.e. straight through an unbroken diagonal wall.
9. **Markers.** Cell centres and path vertices are lifted onto the fitted plane
   through `plane_z`. At 10° over a 40 m sweep that is a 7 m correction; the
   previous hardcoded `z = 0.0` put every marker in the air. A layer with
   nothing to draw clears explicitly, because an `ADD` marker holding zero
   points reads as "no change" and leaves the previous frame on screen. The
   clear is a `DELETE` of that layer's own `(ns, id)`, not a `DELETEALL`:
   consumers act on `DELETEALL` globally, so clearing the empty obstacle layer
   used to erase the ground layer drawn on the same topic microseconds earlier.
   `box_array` is the exception — it owns its topic and publishes a variable
   number of ids, so it leads with a `DELETEALL` to retire the ids a busier
   frame left behind.

   Markers carry the incoming cloud's `header.stamp`, not the node's clock, so
   that bag replay under `use_sim_time` does not ask a viewer to look up a TF
   at two unrelated times.

## Parameters

All 24 shared parameters are declared by `params.py` on every node, so any of
them can be overridden per node from a launch file or the command line.

Twenty-one of them are also live: `ros2 param set` on a running node applies
them from the next sweep. `input_topic`, `marker_topic` and `qos_depth` are read
once while the node is built, and node-specific parameters are snapshotted into
node attributes; setting any of those on a live node is **refused with a
reason** rather than acknowledged and ignored.

| Parameter | Default | Meaning |
|---|---|---|
| `input_topic` | `/sensor/lidar_front/points` | Cloud topic subscribed to |
| `marker_topic` | per node | This node's marker topic |
| `qos_depth` | 10 | Queue depth, subscription and publishers |
| `frame_id_override` | `""` | Stamp markers with this frame instead of the cloud's |
| `resolution` | 0.5 | Costmap cell size, metres |
| `robot_radius` | 0.6 | Cells this close to an obstacle are untraversable |
| `inflation_radius` | 1.5 | Range over which proximity still raises cost |
| `cost_scaling` | 3.0 | Exponential falloff rate of the inflation cost |
| `min_ground_points` | 1 | Ground returns before a cell counts as free |
| `min_obstacle_points` | 3 | Non-ground returns before a cell counts as occupied |
| `min_range` | 0.5 | Drop returns closer than this |
| `max_range` | 60.0 | Drop returns beyond this |
| `marker_lifetime_sec` | 0.0 | 0 means markers never expire |
| `distance_threshold` | 0.20 | RANSAC inlier band, metres; chosen on SemanticKITTI 00/05 |
| `ray_fill` | True | Carve cells the beams crossed as free, closing gaps between rings |
| `ray_fill_bins` | 720 | Azimuth bins used when carving |
| `ray_fill_min_cells` | 2 | Ground cells an azimuth needs before it may carve |
| `max_iterations` | 200 | Hypothesis cap per frame |
| `min_iterations` | 12 | Hypotheses always drawn |
| `confidence` | 0.99 | Target probability of one all-inlier sample |
| `max_tilt_deg` | 25.0 | Reject planes tilted further than this from +z |
| `min_inlier_ratio` | 0.15 | Below this the fit counts as failed |
| `max_scoring_points` | 20000 | Subsample size used to score hypotheses |
| `fallback_height` | 0.2 | Fixed z cut used when no plane is found |
| `plane_filter_alpha` | 0.3 | Blend weight of a new estimate |
| `plane_max_tilt_jump_deg` | 5.0 | Reject estimates tilting more than this per frame |
| `plane_max_offset_jump` | 0.5 | Reject estimates moving more than this, metres |

Node-specific parameters: `min_ground_points_frame` (30, all nodes);
`marker_alpha`, `inflation_alpha`, `cluster_voxel_size`, `cluster_link_radius`,
`cluster_min_points`, `cluster_min_height`, `cluster_max_height`,
`obstacle_marker_topic` (ground segmentation); eleven `lane_*` tunables plus
`left_topic`/`right_topic`/`center_topic` (lane boundary); `goal_x`, `goal_y`,
`goal_search_radius`, `start_search_radius`, `connectivity`, `path_width`,
`goal_fallback_axis` (A*); `start_search_radius`, `connectivity`, `field_bands`
(Dijkstra, BFS, DFS).

## What is not here

- No ego-motion or odometry input, so the costmap is single-frame: it is rebuilt
  from scratch on every sweep and nothing is accumulated between them.
- No real-sensor validation. Every number on this page comes from synthetic
  clouds generated in-process.
- The two notebooks at the repository root are exploratory computer-vision work
  and are not part of this ROS 2 package; no `.py` file imports them.
