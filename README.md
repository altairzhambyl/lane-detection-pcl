![ROS 2](https://img.shields.io/badge/ROS_2-Humble-blue?logo=ros&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-scientific-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-splines-8CAAE6?logo=scipy&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-U--Net-FF6F00?logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-GPLv3-green?logo=gnu&logoColor=white)
![LiDAR](https://img.shields.io/badge/LiDAR-PointCloud2-orange)
![RANSAC](https://img.shields.io/badge/RANSAC-ground_plane-orange)

# Lane Detection PCL

**Real-time road surface segmentation and lane boundary detection using both LiDAR point clouds and Computer Vision** — with four path-planning algorithms running on the detected road grid, all visualized in RViz2.

Two detection pipelines in one repo:
- **LiDAR (PCL)** — ground-plane segmentation + grid-based path planning in ROS 2
- **Computer Vision** — U-Net road segmentation trained on camera feeds, with contour-based boundary extraction

---

## Problem

Autonomous vehicles need to understand road geometry in real time. Camera-based lane detection fails in poor lighting, rain, and fog. LiDAR provides reliable 3D geometry regardless of conditions — but turning a raw point cloud into "this is the road, these are the lanes, here's a driveable path" requires multiple processing stages that are hard to prototype quickly in ROS 2.

## Solution

**LiDAR pipeline** — ROS 2 package (`ransac_seg`) takes raw `PointCloud2` data from a front-facing LiDAR and:

1. **Segments the road surface** with RANSAC plane fitting (from-scratch, in `ransac_seg/ground_plane.py`)
2. **Fits smooth spline curves** to left and right lane boundaries
3. **Runs 4 path-planning algorithms** (A\*, BFS, DFS, Dijkstra) on the discretized road grid
4. **Publishes everything as RViz markers** for immediate visual feedback

**Computer Vision pipeline** — processes camera images to:

1. **Compares 4 classical detection methods** (Crop & Stretch, Canny + Hough, Adaptive Threshold, Otsu)
2. **Trains a U-Net** with Dice + BCE combined loss for road segmentation
3. **Extracts left/right track boundaries** via contour analysis on U-Net predictions
4. **Evaluates quantitatively** with Dice, IoU, Hausdorff distance metrics

---

## Demo

### Computer Vision Pipeline

#### Detection Methods Comparison
> 4 classical CV methods evaluated on middle-strip ROI: Crop & Stretch, Canny + Hough, Adaptive Threshold, Otsu.

![Detection Methods](docs/screenshots/cv_detection_methods_cell5_0.png)

#### U-Net Boundary Extraction
> Input camera image → U-Net predicted mask → Track boundaries (RED = left, BLUE = right, GREEN = drivable zone).

![Boundary Extraction](docs/screenshots/cv_boundary_extraction_cell16_0.png)

#### Training & Evaluation
> U-Net training curves (Loss, Dice, IoU) and validation accuracy metrics (confusion matrix, Dice/IoU distributions, Hausdorff distance).

| Training History | Accuracy Metrics |
|:---:|:---:|
| ![Training](docs/screenshots/cv_training_cell13_10.png) | ![Accuracy](docs/screenshots/cv_accuracy_eval_cell15_1.png) |

### LiDAR Pipeline

#### Road Surface Grid + Camera Feed
> Green grid tiles overlay the detected road surface. Front camera feed shown top-right for reference.

![Road Grid](docs/screenshots/road_grid.png)

### A* Path Planning on Road Grid
> Cyan grid = segmented road. Yellow line = A\* shortest path from vehicle to goal.

![A* Path](docs/screenshots/astar_path.png)

### Lane Boundary Detection
> Left: red lane boundary curves (spline fit). Right: LiDAR point cloud with road overlay.

![Lane Boundaries](docs/screenshots/lane_boundaries.png)

### Algorithm Comparison (A* / BFS / DFS)
> Three panels showing different path-planning algorithms running simultaneously on the same road grid.

![Algorithm Comparison](docs/screenshots/algorithms_comparison.png)

---

## Quick Start

```bash
# 1. Clone into your ROS 2 workspace
cd ~/ros2_ws/src && git clone https://github.com/kekeront/lane-detection-pcl.git

# 2. Install deps + build
cd ~/ros2_ws && rosdep install --from-paths src --ignore-src -r -y && colcon build --packages-select ransac_seg

# 3. Source and launch all nodes
source install/setup.bash && ros2 launch ransac_seg algorithms.launch.py
```

Then open **RViz2** and add Marker displays for `/semantic_map/squares`, `/semantic_map/road_left`, `/semantic_map/road_right`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SENSOR INPUTS                            │
│  LiDAR PointCloud2           Camera Images                  │
└──────┬───────────────────────────┬──────────────────────────┘
       │                           │
       ▼                           ▼
 ┌─────────────┐           ┌──────────────────┐
 │ ransac_node  │           │ Classical CV      │
 │ Ground-plane │           │ Canny/Hough/Otsu  │
 │ filter       │           │ Middle-strip ROI  │
 └──────┬───────┘           └────────┬─────────┘
        │                            │ training masks
   ┌────┴────┐                       ▼
   ▼         ▼               ┌──────────────┐
┌───────┐ ┌──────────┐       │ U-Net        │
│ curve │ │ Path     │       │ Dice+BCE loss│
│ Lane  │ │ Planners │       │ Mixed FP16   │
│ spline│ │ A*/BFS/  │       └──────┬───────┘
└───┬───┘ │ DFS/Dijk │              │
    │     └────┬─────┘              ▼
    ▼          ▼             ┌──────────────┐
  RViz2 Visualization       │ Boundary     │
                             │ Extraction   │
                             │ L/R contours │
                             └──────────────┘
```

> Full LiDAR architecture: [`docs/architecture.md`](docs/architecture.md)

---

## Key Features

### LiDAR (ROS 2)
- **RANSAC ground-plane segmentation** — from-scratch estimator (`ground_plane.py`): adaptive
  iteration budget from the inlier ratio, degenerate-sample rejection, a tilt gate so walls and
  vehicle flanks can't win over the road, and an SVD total-least-squares refit on the consensus set.
  On a 120k-point cloud with a 10° slope it recovers the surface at **1.00 recall / 1.00 precision
  in ~18 ms**, where the old fixed `z < 0.2` threshold manages 0.71 / 0.75. Falls back to the
  height filter (and warns) when no plane clears the inlier floor.
- **Spline lane boundaries** — `scipy.interpolate.UnivariateSpline` fits smooth left/right curves
- **4 path planners** — A\* (heuristic), BFS (shortest), DFS (exploratory), Dijkstra (uniform-cost)
- **Grid discretization** — continuous point cloud → 0.5m cells, shared across all nodes
- **Single launch file** — `ros2 launch ransac_seg algorithms.launch.py` starts everything

### Computer Vision
- **4 classical detection methods** — compared side-by-side on middle-strip ROI
- **U-Net segmentation** — trained with Dice + BCE combined loss, mixed-precision (FP16)
- **Contour-based boundary extraction** — left/right edge + drivable zone from predicted masks
- **Quantitative evaluation** — Dice, IoU, Hausdorff distance, confusion matrix on validation set

---

## ROS 2 Topics

### Subscribed

| Topic | Type | Description |
|-------|------|-------------|
| `/sensor/lidar_front/points` | `sensor_msgs/PointCloud2` | Raw LiDAR point cloud |

### Published

| Topic | Type | Description |
|-------|------|-------------|
| `/semantic_map/squares` | `visualization_msgs/Marker` | Road surface grid (CUBE_LIST) |
| `/semantic_map/road_left` | `visualization_msgs/Marker` | Left lane boundary (LINE_STRIP) |
| `/semantic_map/road_right` | `visualization_msgs/Marker` | Right lane boundary (LINE_STRIP) |

---

## Project Structure

```
lane-detection-pcl/
├── ransac_seg/                          # ROS 2 LiDAR pipeline
│   ├── ground_plane.py                  # RANSAC plane fitting (numpy, no ROS deps)
│   ├── ransac_node.py                   # Core road segmentation node
│   ├── curve.py                         # Lane boundary spline fitting
│   ├── astar.py                         # A* path planning
│   ├── bfs.py                           # BFS path planning
│   ├── dfs.py                           # DFS path planning
│   └── djikstra.py                      # Dijkstra path planning
├── track_boundary_detection_clean.ipynb  # CV pipeline (U-Net + classical)
├── checkpoint.ipynb                      # CV baseline experiments
├── launch/
│   └── algorithms.launch.py             # Launch all ROS 2 nodes
├── docs/
│   ├── architecture.md                  # LiDAR pipeline architecture
│   └── screenshots/                     # Demo visualizations
├── package.xml
├── setup.py
└── setup.cfg
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Middleware | ROS 2 (ament_python) |
| Language | Python 3.10+ |
| Point Cloud | sensor_msgs_py / PointCloud2 |
| Deep Learning | TensorFlow / Keras (U-Net) |
| Classical CV | OpenCV (Canny, Hough, Otsu) |
| Math | NumPy, SciPy |
| Visualization | RViz2 / Foxglove / Matplotlib |

---

## Dependencies

- **ROS 2** (Humble or later, `ament_python` build)
- `rclpy`, `sensor_msgs`, `sensor_msgs_py`, `visualization_msgs`, `geometry_msgs`
- `numpy`, `scipy`

---

## Usage

### Individual nodes

```bash
ros2 run ransac_seg ransac_node     # Road grid only
ros2 run ransac_seg curve           # Road grid + lane curves
ros2 run ransac_seg astar           # A* path on grid
ros2 run ransac_seg bfs             # BFS path on grid
ros2 run ransac_seg dfs             # DFS path on grid
ros2 run ransac_seg djikstra        # Dijkstra path on grid
```

### All at once

```bash
ros2 launch ransac_seg algorithms.launch.py
```

### Foxglove demo (no LiDAR required)

Generates a 60-frame MCAP recording of a drive over ground whose slope changes
underneath the sensor, running the real `segment_ground()` on every sweep:

```bash
python3 tools/foxglove_demo.py            # writes docs/ransac_demo/*.mcap
```

Open the `.mcap` in [Foxglove](https://app.foxglove.dev) and add a **3D panel**.
Topics:

| Topic | What it shows |
|-------|---------------|
| `/lidar/points` | Full cloud, green = road / red = obstacle **as classified by RANSAC** |
| `/lidar/legacy_points` | Same cloud classified by the old `z < 0.2` filter |
| `/ransac/plane` | The fitted plane as a translucent quad |
| `/ransac/stats` | Per-frame tilt, inlier ratio, recall, precision |
| `/semantic_map/squares` | The 0.5 m road grid the planners consume |

Scrub the timeline: as the slope steepens the RANSAC plane tracks it, while the
fixed threshold progressively misclassifies the road as obstacle. Averaged over
the 60 frames — **RANSAC 1.000 recall / 1.000 precision, legacy 0.418 / 0.850**.

### Live in Foxglove (no LiDAR either)

Streams the same scene into the real nodes instead of a file, so
`ransac_node` / `curve` run unmodified and publish their markers live:

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml   # shell 1
python3 tools/demo_publisher.py                          # shell 2
python3 -m ransac_seg.ransac_node                        # shell 3
python3 -m ransac_seg.curve                              # shell 4
```

Connect Foxglove to `ws://localhost:8765` (Open Connection → Foxglove
WebSocket), or open the deep link directly:

```bash
xdg-open 'foxglove://open?ds=foxglove-websocket&ds.url=ws%3A%2F%2Flocalhost%3A8765'
```

Sweeps publish at ~6 Hz rather than the requested 10 — the bottleneck is the
per-point Python `struct.pack` in the demo publisher (~14k points/frame), not
the segmentation, which takes ~18 ms.

![RANSAC vs fixed threshold](docs/screenshots/ransac_vs_threshold.png)

*Side view at 12° slope. Top: RANSAC follows the surface. Bottom: everything
past ~9 m sits above `z = 0.2` and is discarded as obstacle.*

---

## Limitations

- Ground-plane RANSAC assumes a sensor frame with +z roughly up (the tilt gate depends on it) and
  a single dominant surface — it will not split a road from an adjacent parallel ramp
- Validated on synthetic clouds only; no rosbag replay against real sensor data yet
- Path planners use uniform step cost (no terrain weighting)
- No obstacle avoidance layer — planners only consider road-surface cells
- Lane boundary detection assumes a single road corridor (no intersections)

---

## Roadmap

- [x] Replace Z-threshold with RANSAC plane fitting for robust segmentation
- [ ] Validate against recorded rosbags from a real LiDAR
- [ ] Add obstacle point clustering and avoidance constraints
- [ ] Weighted grid costs based on distance from lane center
- [ ] Support for multi-lane detection and intersections
- [ ] Integration with `nav2` for full autonomous navigation stack

---

## License

This project is licensed under the **GNU General Public License v3.0**. See [LICENSE](LICENSE) for details.
