#!/usr/bin/env python3
"""
Regenerate the golden files the C++ spline and lane tests validate against.

The C++ SmoothingSpline is a deliberate replacement for scipy's FITPACK-backed
UnivariateSpline (see spline.hpp), so the deviation is only acceptable if it is
measured: this script runs the *reference* Python on exactly the synthetic
roads test/test_lane.py builds and writes both the inputs and the scipy answers
out, so test_spline.cpp can report a real max-deviation instead of a hope.

It imports ransac_seg and test/test_lane.py read-only; it modifies nothing.
The only interference is a temporary monkeypatch of the module-private
``_smooth`` and ``_occupied_bins`` recorders inside this process.

Point clouds are written as raw little-endian float32 rather than text. Two
reasons, both load-bearing: the C++ ``fit_lane`` signature takes ``const float
*`` because that is how a PointCloud2 payload already sits in memory, so
feeding it float32 is what the production path does anyway and dumping float32
removes the input-precision difference from the comparison entirely; and the
text form of 120k xy pairs is ~6 MB per case against 0.96 MB binary.

Usage:  python3 cpp/tools/dump_golden.py [output_dir]
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)

from ransac_seg import lane as lane_mod  # noqa: E402
from ransac_seg.lane import fit_lane  # noqa: E402

TRUE_WIDTH = 6.00


def straight_road(n=60000, half_width=TRUE_WIDTH / 2.0, length=40.0, seed=1):
    """Return ground returns for a straight road of constant width."""
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [rng.uniform(0.0, length, n), rng.uniform(-half_width, half_width, n)]
    )


def curved_road(n=60000, half_width=TRUE_WIDTH / 2.0, length=40.0, seed=2):
    """Return ground returns straddling the centreline ``y = 0.01 x^2``."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, length, n)
    lateral = rng.uniform(-half_width, half_width, n)
    slope = 0.02 * t
    norm = np.hypot(slope, 1.0)
    return np.column_stack([t - lateral * slope / norm, 0.01 * t**2 + lateral / norm])


def obstacle_road():
    """Return the roadside-obstacle scene of test_lane.py, seed 3."""
    rng = np.random.default_rng(3)
    half = 3.5
    road = np.column_stack(
        [rng.uniform(0.0, 40.0, 120000), rng.uniform(-half, half, 120000)]
    )
    bx = rng.uniform(12.0, 17.0, 8000)
    by = rng.uniform(-6.5, -4.5, 8000)
    bz = rng.uniform(0.0, 1.5, 8000)
    leaked = bz < 0.03
    return np.vstack([road, np.column_stack([bx[leaked], by[leaked]])])


def gap_road():
    """Return the 14 m near run plus 10 m far run across a 16 m hole, seed 4."""
    rng = np.random.default_rng(4)
    near = np.column_stack(
        [rng.uniform(0.0, 14.0, 20000), rng.uniform(-3.0, 3.0, 20000)]
    )
    far = np.column_stack(
        [rng.uniform(30.0, 40.0, 20000), rng.uniform(-3.0, 3.0, 20000)]
    )
    return np.vstack([near, far])


def bend_road():
    """Return the 120 degree arc of test_lane.py, seed 5."""
    rng = np.random.default_rng(5)
    theta = rng.uniform(0.0, np.radians(120.0), 60000)
    radius = 20.0 + rng.uniform(-3.0, 3.0, 60000)
    return np.column_stack([radius * np.sin(theta), radius * (1.0 - np.cos(theta))])


def three_column_road():
    """Return the (N, 3) acceptance case: straight road with a zero z column."""
    xy = straight_road(n=20000)
    return np.column_stack([xy, np.zeros(xy.shape[0])])


# name -> (cloud builder, fit_lane keyword overrides)
CASES = [
    ("straight", straight_road, {}),
    ("curved", curved_road, {}),
    ("obstacle", obstacle_road, {}),
    ("gap", gap_road, {}),
    ("bend", bend_road, {}),
    ("straight_p60", straight_road, {"edge_percentile": 60.0}),
    ("straight_minw8", straight_road, {"min_width": 8.0}),
    ("straight_maxw4", straight_road, {"max_width": 4.0}),
    ("tiny", lambda: straight_road(n=10), {}),
    ("short_run", lambda: straight_road(n=2000, half_width=0.5, length=3.0), {}),
    ("xyz3", three_column_road, {}),
]

# Cases whose cloud is byte-identical to another case's.
ALIASES = {
    "straight_p60": "straight",
    "straight_minw8": "straight",
    "straight_maxw4": "straight",
}

DEFAULT_OPTS = {
    "bin_length": 2.0,
    "min_points_per_bin": 8,
    "edge_percentile": 2.0,
    "centerline_degree": 2,
    "robust_iterations": 3,
    "max_gap": 6.0,
    "smoothing_per_point": 0.02,
    "n_samples": 120,
    "min_width": 2.0,
    "max_width": 12.0,
    "min_bin_fill": 0.5,
}

OPT_ORDER = [
    "bin_length",
    "min_points_per_bin",
    "edge_percentile",
    "centerline_degree",
    "robust_iterations",
    "max_gap",
    "smoothing_per_point",
    "n_samples",
    "min_width",
    "max_width",
    "min_bin_fill",
]


def fmt(values):
    """Format a 1-D float array as round-trippable whitespace-separated text."""
    return " ".join("%.17g" % float(v) for v in np.asarray(values).reshape(-1))


def main():
    """Write the golden inputs and scipy answers, then print the file list."""
    out_dir = (
        sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "..", "test", "data")
    )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    smooth_calls = []
    occupied = []
    real_smooth = lane_mod._smooth
    real_occupied = lane_mod._occupied_bins

    def recording_smooth(s_bins, values, s_query, smoothing_per_point):
        out = real_smooth(s_bins, values, s_query, smoothing_per_point)
        smooth_calls.append(
            (
                np.asarray(s_bins, dtype=np.float64),
                np.asarray(values, dtype=np.float64),
                np.asarray(s_query, dtype=np.float64),
                float(smoothing_per_point),
                np.asarray(out, dtype=np.float64),
            )
        )
        return out

    def recording_occupied(*args, **kwargs):
        centres, groups, counts = real_occupied(*args, **kwargs)
        occupied.append((np.asarray(centres), np.asarray(counts)))
        return centres, groups, counts

    written = []
    spline_lines = []
    lane_lines = []
    n_spline_cases = 0

    lane_mod._smooth = recording_smooth
    lane_mod._occupied_bins = recording_occupied
    try:
        for name, builder, overrides in CASES:
            cloud = np.ascontiguousarray(builder(), dtype=np.float32)
            cols = cloud.shape[1]
            # The option-variant cases re-fit the very same cloud, so they share
            # one file rather than three identical 0.48 MB copies.
            rel = "road_%s.f32" % ALIASES.get(name, name)
            if rel not in written:
                cloud.tofile(os.path.join(out_dir, rel))
                written.append(rel)

            smooth_calls.clear()
            occupied.clear()
            opts = dict(DEFAULT_OPTS)
            opts.update(overrides)
            # float32 in, float64 maths: exactly what the C++ signature sees.
            fit = fit_lane(cloud.astype(np.float64), **opts)

            lane_lines.append("case %s" % name)
            lane_lines.append("cloud %s %d %d" % (rel, cloud.shape[0], cols))
            lane_lines.append(
                "opts " + " ".join("%.17g" % float(opts[k]) for k in OPT_ORDER)
            )
            lane_lines.append("valid %d" % (1 if fit.valid else 0))
            lane_lines.append("reason %s" % (fit.reason if fit.reason else "-"))
            lane_lines.append("n %d" % fit.width.shape[0])
            lane_lines.append("centerline " + fmt(fit.centerline))
            lane_lines.append("left " + fmt(fit.left))
            lane_lines.append("right " + fmt(fit.right))
            lane_lines.append("width " + fmt(fit.width))
            lane_lines.append("arclength " + fmt(fit.arclength))

            if smooth_calls:
                n_spline_cases += 1
                final_centres, final_counts = occupied[-1]
                spline_lines.append("case %s %d" % (name, len(smooth_calls)))
                spline_lines.append("counts " + fmt(final_counts.astype(np.float64)))
                for x, y, q, spp, out in smooth_calls:
                    spline_lines.append(
                        "call %d %d %.17g" % (x.shape[0], q.shape[0], spp)
                    )
                    spline_lines.append("x " + fmt(x))
                    spline_lines.append("y " + fmt(y))
                    spline_lines.append("q " + fmt(q))
                    spline_lines.append("out " + fmt(out))
    finally:
        lane_mod._smooth = real_smooth
        lane_mod._occupied_bins = real_occupied

    spline_path = os.path.join(out_dir, "golden_spline.txt")
    with open(spline_path, "w") as handle:
        handle.write("cases %d\n" % n_spline_cases)
        handle.write("\n".join(spline_lines) + "\n")
    written.append("golden_spline.txt")

    lane_path = os.path.join(out_dir, "golden_lane.txt")
    with open(lane_path, "w") as handle:
        handle.write("cases %d\n" % len(CASES))
        handle.write("\n".join(lane_lines) + "\n")
    written.append("golden_lane.txt")

    print("wrote to %s:" % out_dir)
    for rel in written:
        size = os.path.getsize(os.path.join(out_dir, rel))
        print("  %-28s %9d bytes" % (rel, size))


if __name__ == "__main__":
    main()
