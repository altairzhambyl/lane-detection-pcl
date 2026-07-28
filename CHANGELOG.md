# Changelog

## 2026-07-28 — Real-data benchmarks: SemanticKITTI accuracy, rosbag latency, threshold refit

**Situation**
- Every number in the repo came from clouds the repo generated. The synthetic scene reported
  RANSAC at recall 1.000 / precision 1.000, which measures the estimator against the exact model
  it assumes and proves nothing about a sensor.
- `tools/benchmarks.py`, which the README calls "a single command" that reproduces every figure,
  did not run: it never put the repo root on `sys.path`, so `import ransac_seg` failed.

**Options**
1. Keep the synthetic figures and label them harder.
2. Measure on the recorded drive only — real, but unlabelled, so no accuracy figure is possible.
3. Measure accuracy on SemanticKITTI (labelled, and the standard benchmark for this task) and
   latency on the recorded drive, and let each dataset answer only what it can support.

**Decisions**
- Took option 3. `tools/eval_semantickitti.py` scores IoU/precision/recall against the ground class
  set the published benchmarks use; `tools/bench_rosbag.py` times each stage on a real recording.
- The fixed-height baseline is swept over every threshold and reported at its best one, chosen on
  the test data itself — an oracle no deployed system gets.
- `distance_threshold` default 0.08 -> 0.20, chosen on sequences 00 and 05 and reported on all 11.
  Both the ROS parameter and `ground_plane.DEFAULT_DISTANCE_THRESHOLD` now carry it; two defaults
  for one quantity is how the eval script silently measured a configuration nobody runs.
- CLI defaults in the eval script follow the package constants rather than repeating them.

**Reasoning**
- 0.08 m was fitted to synthetic noise of 1 cm. Real ground is spread by camber, kerbs and gravel,
  so the band could not contain it: recall lost 21 points and 1046 of 23 201 frames failed to fit
  at all and silently fell back to a fixed height. This was a defect wearing a parameter's clothes.
- A baseline tuned to lose proves nothing, so the threshold baseline is given every advantage.
  Against it the honest claim is "on par", not "far better" — and that is what the README says.

**Outcomes** — every figure reproduced by the command printed beside it
- SemanticKITTI, 11 sequences, 23 201 frames: thr 0.08 **IoU 0.620** (prec 0.881, rec 0.676,
  p95 34.8 ms, 1046 frames with no plane) -> thr 0.20 **IoU 0.808** (prec 0.902, rec 0.886,
  p95 29.2 ms, 0 such frames). Oracle fixed height **0.797**; RANSAC wins 7 of 11 sequences.
- Recorded drive, 300 sweeps of 76 813 points: decode 0.6 / segment 26.1 / cluster 32.0 /
  costmap 9.8 / lane 18.6 / plan 6.5 ms at p50, **sum 93.6 ms** against a 100 ms budget. Routes on
  **281/300** frames (142 to the requested goal, 139 via fallback); 19 frames yield no start cell.
- Planner p95 is 87.3 ms on its own: an unreachable goal pays for a full Dijkstra field first.
- `tools/benchmarks.py` now runs as documented. Synthetic callback times shifted with the new
  threshold; **two** of six nodes now miss 100 ms, not three, and the README table is re-measured.
- `python3 -m pytest test/ -q` -> **176 passed, 1 skipped**.

**Lessons**
- A default tuned on synthetic data is a synthetic default. It survived a full adversarial review
  because every test that exercised it also used the cloud it was fitted to.
- Two constants for one quantity is a silent-drift generator: the eval script kept reporting 0.567
  after the package moved to 0.20, because argparse repeated the old value instead of importing it.
- Measuring `astar()` directly under-reported the pipeline, which falls back to the furthest
  reachable cell. Benchmarks have to replicate the caller, not the function.

**Next-step assumptions**
- One global plane cannot fit `sidewalk` and `terrain` at once; sector-wise fitting is the next
  real gain and is not written.
- The planner tail is the open budget problem, not segmentation.
- 0.20 m is this sensor's number. No auto-tuning exists.

## 2026-07-28 — Adversarial review fixes: planner corner-cutting, clearance, QoS, marker lifecycle

**Situation**
- Three reviewers audited the freshly integrated pipeline and demonstrated eleven defects by
  running the real modules. Two were blocking: an 8-connected search that walked through the
  shared corner of two diagonally touching blocked cells (a straight line *through* an unbroken
  diagonal wall), and a cloud subscription created from a bare depth, so it inherited rclpy's
  RELIABLE default and receives nothing from the BEST_EFFORT publisher every LiDAR driver offers.
- The docs carried a headline figure — legacy `z < 0.2` at recall 0.168 / precision 0.920 — that
  reproduces from nothing in the repo and contradicts the repo's own earlier CHANGELOG entry
  (0.714 / 0.745) for a scene described in identical words, 74 lines apart, neither retracted.
- README:161 pointed readers at CHANGELOG.md for "the exact commands"; CHANGELOG.md held two
  commands covering two of the ten figure groups.

**Options**
1. Fix the two blocking defects, note the rest (fastest; leaves a map that overstates its own
   clearance and a README nobody can check).
2. Fix the code and delete the unverifiable numbers from the docs (honest, but removes the only
   evidence the pipeline works).
3. Fix the code, then write the missing reproduction path — one script that prints every figure
   the docs quote — and rewrite the docs against its actual output.

**Decisions**
- Took option 3. New `tools/benchmarks.py` with six sections (decode, segmentation, clustering +
  costmap, lane, planners, node callbacks); every scene is defined in it in full.
- `grid.neighbors` offers a diagonal step only when both cells it squeezes past are traversable.
- Inflation clearance is the distance transform minus half a cell diagonal, so a cell left FREE is
  `robot_radius` clear of the obstacle *return*, not of the obstacle's cell centre.
- Over `MAX_CELLS` the map coarsens the resolution instead of rescaling nx/ny at a fixed one, which
  used to crop the map onto a corner of the observed ground and silently drop the rest.
- Fixed-id marker constructors clear with a `DELETE` of their own `(ns, id)`; only `box_array`,
  which owns its topic and publishes a variable id range, leads with a `DELETEALL`.
- The cloud subscription requests BEST_EFFORT; markers carry the cloud's `header.stamp`; an
  on-set-parameters callback applies the 21 live shared parameters and refuses the rest with a
  reason.

**Reasoning**
- Corner-cutting and centre-to-centre clearance are the same class of bug: the map describes cells,
  the vehicle occupies metres, and both places quietly swapped one for the other. Both fixes are
  conservative in the same direction — a corner the vehicle cannot fit through is not an edge, and
  a clearance the geometry does not guarantee is not clearance.
- `DELETEALL` was the wrong clear for a topic carrying three namespaces: viewers implement it as
  "wipe this display". The narrow `DELETE` is what the fixed-id layers actually meant.
- A wrong benchmark number is worse than no benchmark number, and a reproduction path that does not
  exist is the same failure as the "RANSAC-ready" claim, one layer down. The script had to be
  written before the docs could be rewritten.

**Outcomes** — every figure measured this session by the command shown, all on synthetic clouds
- `python3 -m pytest test/ -q` → **`171 passed, 1 skipped, 2 warnings`** (was 146 passed,
  1 skipped). 25 new tests, each confirmed to fail against the pre-fix code and pass after.
- `python3 tools/benchmarks.py --section segmentation`: RANSAC **recall 1.000 / precision 1.000 in
  16.0 ms** on the 120k / 10° / 35 %-outlier cloud; legacy `z < 0.2` **0.717 / 0.744**. The
  previously published 0.168 / 0.920 is retracted in place below.
- `--section decode`: `pointcloud2_to_xyz` **1.6 ms** vs **206 ms** for the `read_points` loop,
  max abs difference 0.
- `--section clustering`: 3 clusters from 2700 returns in **5.2 ms**, extents 4.19 × 1.80 m against
  4.20 × 1.80 m planted; costmap **2.7 ms**, 81 × 19 cells, 1132 free / 108 occupied / 51 inflated /
  248 unknown, free-cell cost 1.000–1.870.
- `--section lane`: straight 7.00 m road **6.713 m** (estimand 6.720 m); 6.00 m road at R = 60 m
  **5.749 m** (estimand 5.760 m).
- `--section planners`: A\* total cost **39.000000**, Dijkstra at the same goal **39.000000**;
  DFS and BFS visit the identical **1132** cells.
- `--section callbacks`, 120k-point sweep: A\* 44.0 ms, BFS 59.1 ms, DFS 74.0 ms, ground
  segmentation 105.3 ms, Dijkstra 134.9 ms, lane boundary 143.3 ms. **Three of six miss the 100 ms
  a 10 Hz sensor allows**, and the launch file runs all six over the same cloud. The README's
  "comfortable" claim came from a 14k-point scene and is now a table plus a Limitations bullet.
- Live check of the QoS fix, isolated `ROS_DOMAIN_ID=91`: **380 frames** processed in 4 s from a
  BEST_EFFORT publisher, against 0 before.
- `python3 tools/foxglove_demo.py --frames 20`: RANSAC 1.000 / 1.000, legacy 0.437 / 0.753, lane
  fitted 20/20 at median width 8.61 m, 2.6 clusters/frame, route planned 20/20 at median cost 24.3
  (was 23.9; the clearance fix inflates slightly more).
- `demo_publisher` tick re-measured: **55.1 ms**.
- TensorFlow and OpenCV badges removed from the README: nothing shippable imports either, and the
  disclosure sat below the fold.

**Lessons**
- A number nobody can regenerate rots into a claim. Both retracted figures survived a docs pass
  that was written *after* the measurements, because "measured this session" was recorded without
  the command that measured it. The command is the artefact, not the number.
- The three DELETE/DELETEALL defects, the corner cut and the centre-to-centre clearance were all
  found by running the module, not by reading it. Every one of them type-checks, lints and reads
  correctly.
- rclpy runs on-set-parameter callbacks for declarations too, so `has_parameter(name)` is the
  discriminator between "a subclass is declaring this" and "an operator is setting it".

**Next-step assumptions**
- Still no real-sensor numbers. The BEST_EFFORT fix means a real driver or `ros2 bag play` will now
  actually feed the nodes, which makes that the next thing to earn rather than the next blocker.
- The per-sweep budget is the open defect: a shared perception node publishing one decoded,
  segmented, rasterised frame to the five consumers removes roughly five sixths of the work.
  Not written.
- Node-specific parameters are refused at runtime rather than applied. If retuning `goal_x` or the
  cluster gates on a live node matters, they need the same treatment as the shared set.
- `tools/benchmarks.py` has no test coverage; it is a script whose output nobody asserts against.

## 2026-07-28 — Whole-pipeline integration: clustering, cost grid, planners, docs

**Situation**
- RANSAC landed, but everything downstream of it was still the original placeholder: six nodes each
  carrying their own per-point decode loop, a "grid" that was `unique(floor(xy/0.5))` drawn at
  `z = 0.0`, planners searching that set of ground cells with a uniform step cost, and no obstacle
  channel at all.
- `docs/architecture.md` still said "Ground-plane filter (RANSAC-ready)" and "Designed to be
  replaced with full RANSAC plane fitting" — the exact phrase the repo was criticised for — while
  the previous CHANGELOG entry claimed "README claims now match the code".
- `djikstra.py` was misspelled, the launch file started five of six nodes, `.github/workflows/`
  existed but was empty, and `test/test_nodes_end_to_end.py` covered three nodes via a `RoadSquares`
  alias and a `callback` method.

**Options**
1. Fix the docs only, leave the pipeline as it was (cheapest; leaves the same gap one layer down).
2. Adopt `nav2`'s costmap_2d and PCL/Open3D for clustering (mature, but the repo has no C++ and the
   point of the exercise is the geometry, not the integration).
3. Build the missing stages in numpy/scipy behind one shared `PerceptionNode`, then rewrite the docs
   against the finished system.

**Decisions**
- Took option 3, split across a foundation (`cloud_io`, `plane_filter`, `grid`, `markers`, `params`,
  `perception_node`) and three tracks (perception, planning, data/packaging), integrated here.
- One cost model everywhere: `edge(a,b) = step_m × 0.5 × (cost[a] + cost[b])`, implemented once in
  `planners.py`, so A\* and Dijkstra are comparable by construction.
- DFS is documented and shipped as a traversal-order baseline, not a fourth planner, and audits
  itself against BFS on live data every second.
- Markers place geometry with `plane_z`, and empty geometry becomes `DELETEALL` rather than an ADD
  holding zero points.
- `tools/foxglove_demo.py`'s scene narrowed from a 28 m open field to a 9 m road corridor, because
  the lane fit correctly rejects anything wider than `max_width` and the boundary half of the
  pipeline had nothing to show.
- The end-to-end test was rewritten against the real class names and now covers all six nodes,
  the blocked-ground case, the plane-projection regression and two consecutive frames.

**Reasoning**
- A costmap built only from ground returns cannot express an obstacle: the old planners saw a hole
  in the ground where a car was, which is indistinguishable from unobserved road. Occupied-beats-
  free plus inflation is the smallest change that makes "blocked" a state rather than an absence.
- A Euclidean heuristic on a grid graph is inadmissible; octile for 8-connectivity and Manhattan
  for 4 are the tight ones, and the A\*/Dijkstra agreement below is what proves it.
- Docs were rewritten last, after the measurements, because the previous entry's failure mode was
  writing the claim before the code.

**Outcomes** — every figure measured this session, all on synthetic clouds
- `python3 -m pytest test/ -q` → **`146 passed, 1 skipped, 2 warnings in 6.96s`** (was 22 passed,
  1 skipped; the two failures the tracks reported are gone).
- Decode, 120k-point cloud: `pointcloud2_to_xyz` **0.2 ms** vs **281 ms** for the per-point
  `read_points` loop, identical output (max abs diff 0).
- Segmentation, synthetic 120k cloud at 10° slope with 35 % non-ground returns: RANSAC **recall
  1.000 / precision 1.000 in 15.4 ms**; fixed `z < 0.2` **0.168 / 0.920** on the same cloud.
  **Retracted 2026-07-28 (see the entry above):** the legacy pair does not reproduce from anything
  in this repo. That cloud gives `z < 0.2` **0.717 / 0.744**, printed by
  `python3 tools/benchmarks.py --section segmentation`.
- Costmap, synthetic parked-car scene: **4.0 ms**, 80 × 16 cells, 933 free / 108 occupied /
  51 inflated / 188 unknown, free-cell cost 1.000–1.725.
- Clustering, same scene: **3 clusters from 2700 non-ground returns in 7.6 ms**, extents
  4.19 × 1.79 m against 4.2 × 1.8 m planted.
- Lane width: straight 7.00 m road **6.700 m** (estimand 6.720 m); 6.00 m road curving at R = 60 m
  **5.761 m** (estimand 5.760 m, error 0.001 m; 0.239 m against the nominal width).
- Planner agreement: A\* total path cost **23.500000**, Dijkstra cost field at the same goal
  **23.500000**, difference 0. DFS and BFS visit the identical **933** cells.
  Timings on that grid: A\* 1.8 ms, Dijkstra 19.1 ms, BFS 7.4 ms.
- Per-frame node callbacks on a 14k-point sweep: ground segmentation 16.6 ms, lane boundary 26.4 ms,
  A\* 10.5 ms, Dijkstra 31.5 ms, BFS 32.3 ms, DFS 26.4 ms.
- `python3 tools/foxglove_demo.py --frames 20`: RANSAC mean recall 1.000 / precision 1.000, legacy
  0.437 / 0.753, 0 fallback frames, lane fitted 20/20 frames at median width 8.61 m, 2.6 clusters
  per frame, route planned 20/20 frames at median cost 23.9. `demo_publisher` tick: **56.8 ms**.
- Tree reconciled: `djikstra.py` gone, all six `setup.py` entry points import and expose `main()`,
  the launch file starts all six under the names the nodes actually register
  (`ground_segmentation_node`, `lane_boundary_node`, and the four planner names), and `package.xml`
  gained `rcl_interfaces`, `builtin_interfaces` and `python3-matplotlib`.

**Lessons**
- A docs claim is only true once every file the README links to has been checked. The previous entry
  said the docs were fixed; `docs/architecture.md` still carried "RANSAC-ready" and "designed to be
  replaced", so the fix covered one file out of two and the overclaim survived in the deeper one —
  which is the file a reviewer reads when they want detail.
- Aliases kept "for the transitional test" (`RoadSquares`, `callback = _on_cloud`) let a stale test
  pass against a renamed API and hid the mismatch until the integration step. They are now redundant
  and should be deleted in a follow-up.
- Screenshots age faster than code. The four RViz captures predate clustering and the cost grid;
  they are now labelled as such rather than quietly implying current output.

**Next-step assumptions**
- No real-sensor accuracy numbers exist. The ingestion path (`tools/kitti_publisher.py`,
  `tools/replay_bag.py`, `ransac_seg/point_files.py`) is exercised and tested, but nothing has been
  run against a recorded vehicle dataset; that is the next claim to earn.
- No ego-motion or odometry, so the cost grid is single-frame — rebuilt per sweep, nothing
  accumulated. Any "map" language must wait for that.
- `tools/foxglove_demo.py`'s 60-frame averages have no test coverage; they are printed by a script
  nobody asserts against.
- `.github/workflows/ci.yml` has never executed. No badge, no status claim, until it has.

## 2026-07-28 — RANSAC ground-plane segmentation

**Situation**
- Package was named `ransac_seg`, README advertised "RANSAC-ready" ground-plane segmentation, and the
  Limitations section disclosed the truth — but RANSAC was never implemented.
- All six nodes shared the same placeholder: `road_pts = pts[pts[:,2] < 0.2]` under a
  `TODO: Apply your existing RANSAC road segmentation here` comment.
- A portfolio review flagged the naming/badge as an overclaim on first-glance read.

**Options**
1. Rename the package and drop the RANSAC framing (cheapest, removes the claim).
2. Call `pcl::SACSegmentation` or `open3d.segment_plane` (fast, but adds a heavy dep and
   demonstrates nothing — the repo has no C++ and no PCL linkage).
3. Implement RANSAC from scratch in numpy behind a small interface the nodes call.

**Decisions**
- Took option 3. New `ransac_seg/ground_plane.py`: minimal 3-point hypotheses, vectorised scoring,
  adaptive iteration budget, tilt gate, SVD total-least-squares refit on the consensus set.
- `segment_ground()` returns `(mask, model)`; `model is None` means the fixed-height fallback ran,
  so nodes can warn instead of silently degrading.
- Kept the fallback rather than failing closed — a perception node that stops publishing is worse
  than one that publishes a degraded surface and says so.

**Reasoning**
- A fixed z threshold only holds on flat ground with a level sensor; any slope or mount pitch makes
  it eat the road or keep the curb. That is the actual defect, not the missing library call.
- Scoring hypotheses against a capped random subset keeps the loop O(subset) per iteration while
  the consensus ratio stays a good estimate; the winner is re-scored against the full cloud.
- The tilt gate exists because a wall or truck flank fits a plane *better* than a road does — the
  geometric best fit is the wrong answer, and only the frame convention distinguishes them.

**Outcomes**
- 120k-point cloud, 10° slope, 35% obstacle returns: recall 1.000 / precision 1.000 in ~18 ms.
  Legacy `z < 0.2` on the same cloud: recall 0.714 / precision 0.745.
- 16 unit tests (slope recovery, wall rejection, degenerate/collinear/NaN input, determinism,
  subsample equivalence) + 4 end-to-end tests driving real node callbacks with a `PointCloud2`.
- Full suite: 22 passed, 1 skipped. Repo's own ament flake8/pep257 tests now pass — they were
  failing beforehand on `setup.py` and the launch file, both fixed.
- `package.xml` had `TODO:` description and license and declared no runtime deps; filled in.
- README claims now match the code; Limitations replaced with the real ones.

**Lessons**
- A disclosed limitation in a Limitations section does not offset a package name and a badge that
  claim the opposite — readers grep the headline claim, not the fine print.
- Writing the mask-alignment test first caught a real bug: dropping non-finite rows before sizing
  the output mask returned a mask the caller could not index with.

**Next-step assumptions**
- Synthetic validation only. The recall/precision numbers above do not transfer to a real sensor
  until this is replayed against a rosbag; that is the next task before any further claims.
- Obstacle clustering (the natural follow-on) should consume `~ground_mask` rather than re-segment.
