# Changelog

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
