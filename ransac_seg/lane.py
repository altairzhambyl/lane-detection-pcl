#!/usr/bin/env python3
"""
Lane boundary fitting in the frame of a fitted road centreline.

The previous method took min/max y per integer-x column and splined the result.
Three things break it. Min/max is the least robust estimator there is, so a
single parked car drags the boundary out to meet it and the spline compensates
with an undershoot *into* the drivable lane. A fixed x axis means the extents
are measured along the sensor's x, not across the road, so any curve inflates
the reported width and a road bending past 90 degrees becomes multivalued.
And a hardcoded spline smoothing budget produces different geometry on the
same road depending only on how many range bins the sweep happened to fill.

So: fit a centreline first, measure everything perpendicular to it, use
percentiles instead of extrema, and scale the smoothing budget with the sample
count. Pure numpy/scipy — no ROS import.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.spatial import cKDTree

# Vertex spacing of the internal centreline polyline. Fine enough that the
# nearest-vertex projection below quantises arc length by half of it, coarse
# enough that a 60 m road stays a few hundred vertices.
_CURVE_SPACING = 0.25
_MAX_CURVE_VERTICES = 4000


@dataclass(frozen=True)
class LaneFit:
    """A lane boundary pair expressed in the frame of the fitted centreline."""

    centerline: np.ndarray
    left: np.ndarray
    right: np.ndarray
    width: np.ndarray
    arclength: np.ndarray
    valid: bool
    reason: str


def _empty(reason):
    """Build an empty, invalid fit carrying a machine-readable cause."""
    return LaneFit(
        centerline=np.zeros((0, 2), dtype=np.float64),
        left=np.zeros((0, 2), dtype=np.float64),
        right=np.zeros((0, 2), dtype=np.float64),
        width=np.zeros(0, dtype=np.float64),
        arclength=np.zeros(0, dtype=np.float64),
        valid=False,
        reason=reason,
    )


def _as_xy(ground_xy):
    """Coerce ``(N, 2)`` or ``(N, 3)`` input to finite ``(N, 2)`` float xy."""
    arr = np.asarray(ground_xy, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    arr = arr.reshape(-1, arr.shape[-1])[:, :2]
    return arr[np.isfinite(arr).all(axis=1)]


def _principal_axis(xy):
    """Return the mean-centred cloud's dominant direction, oriented towards +x."""
    centred = xy - xy.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axis = vt[0]
    # A heading pointing backwards would run arc length from far to near and
    # mirror the left/right labels, so pin it to the sensor's forward axis.
    if axis[0] < 0.0:
        axis = -axis
    return axis / np.linalg.norm(axis)


def _resample(curve):
    """Resample a polyline to uniform vertex spacing, returning ``(K, 2)``."""
    steps = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(cumulative[-1])
    if total <= 0.0:
        return curve
    count = int(np.clip(round(total / _CURVE_SPACING) + 1, 2, _MAX_CURVE_VERTICES))
    targets = np.linspace(0.0, total, count)
    return np.column_stack(
        [
            np.interp(targets, cumulative, curve[:, 0]),
            np.interp(targets, cumulative, curve[:, 1]),
        ]
    )


def _tangents(curve):
    """Return unit tangents and cumulative arc length of a polyline."""
    tangent = np.gradient(curve, axis=0)
    norm = np.linalg.norm(tangent, axis=1)
    norm[norm < 1e-12] = 1.0
    tangent = tangent / norm[:, None]
    steps = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    arclength = np.concatenate([[0.0], np.cumsum(steps)])
    return tangent, arclength


def _project(curve, xy):
    """
    Project points onto a polyline, returning ``(s, d)`` in its local frame.

    ``s`` is true arc length along the curve and ``d`` is the signed lateral
    offset, positive to the left of the direction of travel. Measuring the
    offset against the local tangent rather than a global axis is what keeps
    the width honest once the road bends.
    """
    tangent, arclength = _tangents(curve)
    # Every point is projected on every pass, so this is the hot loop of the
    # whole fit; the query is embarrassingly parallel and scipy will thread it.
    _, nearest = cKDTree(curve).query(xy, workers=-1)
    delta = xy - curve[nearest]
    tang = tangent[nearest]
    along = delta[:, 0] * tang[:, 0] + delta[:, 1] * tang[:, 1]
    lateral = -delta[:, 0] * tang[:, 1] + delta[:, 1] * tang[:, 0]
    return arclength[nearest] + along, lateral


def _bin_edges(s, bin_length):
    """Return the bin index per sample and the bin centres present."""
    origin = float(s.min())
    index = np.floor((s - origin) / float(bin_length)).astype(np.int64)
    index -= index.min()
    centres = origin + (np.arange(index.max() + 1) + 0.5) * float(bin_length)
    return index, centres


def _grouped(values, index, n_bins):
    """Return per-bin views of ``values`` as a list of arrays (empty allowed)."""
    order = np.argsort(index, kind="stable")
    sorted_index = index[order]
    starts = np.searchsorted(sorted_index, np.arange(n_bins), side="left")
    stops = np.searchsorted(sorted_index, np.arange(n_bins), side="right")
    ordered = values[order]
    return [ordered[a:b] for a, b in zip(starts, stops)]


def _occupied_bins(curve, xy, bin_length, min_points_per_bin, min_bin_fill):
    """
    Bin the lateral residuals by arc length and keep the fully observed bins.

    The bins at the two ends of a sweep are only partially covered by returns,
    so their percentiles describe a sliver of road rather than a cross-section
    of it and taper the reported width. Requiring a fraction of the typical
    bin population drops them instead of letting them bend the fit.
    """
    s, d = _project(curve, xy)
    index, centres = _bin_edges(s, bin_length)
    groups = _grouped(d, index, centres.shape[0])
    counts = np.array([g.shape[0] for g in groups])

    populated = counts[counts >= int(min_points_per_bin)]
    if populated.size == 0:
        return np.zeros(0), [], np.zeros(0)
    floor = max(int(min_points_per_bin), float(min_bin_fill) * np.median(populated))
    keep = [i for i, c in enumerate(counts) if c >= floor]
    return centres[keep], [groups[i] for i in keep], counts[keep]


def _refine_centerline(
    curve, xy, bin_length, degree, iterations, min_points_per_bin, min_bin_fill
):
    """
    Push a centreline polyline towards the median of the returns around it.

    Each pass re-projects onto the current curve (so ``s`` is arc length, not a
    fixed-axis coordinate), takes the MEDIAN lateral offset per bin — a
    roadside car is a one-sided outlier and would drag a mean — and offsets the
    curve by a low-degree polynomial fitted to those medians.
    """
    for _ in range(int(iterations)):
        centres, groups, counts = _occupied_bins(
            curve, xy, bin_length, min_points_per_bin, min_bin_fill
        )
        if centres.shape[0] < int(degree) + 1:
            break
        medians = np.array([np.median(g) for g in groups])
        # A bin's median is noisier the fewer points it holds, so weight the
        # fit by sqrt(count); an unweighted fit lets the thinnest bin at the
        # far end of the sweep steer the whole curve.
        coeffs = np.polyfit(centres, medians, int(degree), w=np.sqrt(counts))

        tangent, arclength = _tangents(curve)
        # Never evaluate the polynomial outside the span it was fitted over:
        # extrapolating a quadratic past the last bin whips the curve ends out
        # and the next iteration then fits that error, diverging.
        offset = np.polyval(coeffs, np.clip(arclength, centres[0], centres[-1]))
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
        curve = _resample(curve + offset[:, None] * normal)
    return curve


def _longest_run(centres, max_gap):
    """Return the slice of bin centres forming the longest gap-free run."""
    if centres.shape[0] == 0:
        return slice(0, 0)
    breaks = np.flatnonzero(np.diff(centres) > float(max_gap)) + 1
    starts = np.concatenate([[0], breaks])
    stops = np.concatenate([breaks, [centres.shape[0]]])
    # Longest by metres covered, not by bin count: a run of three bins spanning
    # 30 m is more road than five bins spanning 10 m.
    spans = centres[stops - 1] - centres[starts]
    best = int(np.argmax(spans))
    return slice(int(starts[best]), int(stops[best]))


def _smooth(s_bins, values, s_query, smoothing_per_point):
    """Spline-smooth a boundary offset with a budget that scales with bin count."""
    n_bins = s_bins.shape[0]
    # UnivariateSpline's `s` is a total squared-residual budget, so a hardcoded
    # value means a different amount of smoothing per sample as the bin count
    # changes with range sparsity. Scaling it keeps the geometry comparable.
    budget = float(smoothing_per_point) * n_bins
    degree = int(np.clip(n_bins - 1, 1, 3))
    spline = UnivariateSpline(s_bins, values, s=budget, k=degree)
    return spline(s_query)


def fit_lane(
    ground_xy,
    *,
    bin_length=2.0,
    min_points_per_bin=8,
    edge_percentile=2.0,
    centerline_degree=2,
    robust_iterations=3,
    max_gap=6.0,
    smoothing_per_point=0.02,
    n_samples=120,
    min_width=2.0,
    max_width=12.0,
    min_bin_fill=0.5,
):
    """
    Fit left/right lane boundaries in the frame of a fitted road centreline.

    ``ground_xy`` is the already range-gated ground return set, ``(N, 2)`` or
    ``(N, 3)``. Boundaries are the ``edge_percentile`` and
    ``100 - edge_percentile`` percentiles of the lateral residual per arc
    length bin: percentiles tolerate one-sided contamination up to
    ``edge_percentile`` of a bin's population, which is exactly what min/max
    could not survive. The price is a known bias: on a road of uniform return
    density the percentile edge sits ``edge_percentile`` per cent of the width
    inside the true one, so the reported width is low by twice that fraction.

    Always returns a LaneFit. An invalid one carries a machine-readable
    ``reason`` and may hold empty arrays, so a caller can publish a clear
    marker instead of guessing.
    """
    xy = _as_xy(ground_xy)
    if xy.shape[0] < 3 * int(min_points_per_bin):
        return _empty("too_few_points")

    axis = _principal_axis(xy)
    mean = xy.mean(axis=0)
    along = (xy - mean) @ axis
    curve = _resample(
        np.array([mean + float(along.min()) * axis, mean + float(along.max()) * axis])
    )
    curve = _refine_centerline(
        curve,
        xy,
        bin_length,
        centerline_degree,
        robust_iterations,
        min_points_per_bin,
        min_bin_fill,
    )

    bin_s, groups, _ = _occupied_bins(
        curve, xy, bin_length, min_points_per_bin, min_bin_fill
    )
    if bin_s.shape[0] < 3:
        return _empty("gap")

    lo = np.array([np.percentile(g, float(edge_percentile)) for g in groups])
    hi = np.array([np.percentile(g, 100.0 - float(edge_percentile)) for g in groups])

    # Never let a spline invent road across an empty stretch: cut at the gap
    # and keep the longer contiguous run instead.
    run = _longest_run(bin_s, max_gap)
    bin_s, lo, hi = bin_s[run], lo[run], hi[run]
    if bin_s.shape[0] < 3:
        return _empty("gap")

    samples = np.linspace(float(bin_s[0]), float(bin_s[-1]), int(n_samples))
    left_d = _smooth(bin_s, hi, samples, smoothing_per_point)
    right_d = _smooth(bin_s, lo, samples, smoothing_per_point)

    tangent, arclength = _tangents(curve)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    centre_xy = np.column_stack(
        [
            np.interp(samples, arclength, curve[:, 0]),
            np.interp(samples, arclength, curve[:, 1]),
        ]
    )
    normal_xy = np.column_stack(
        [
            np.interp(samples, arclength, normal[:, 0]),
            np.interp(samples, arclength, normal[:, 1]),
        ]
    )
    norm = np.linalg.norm(normal_xy, axis=1)
    norm[norm < 1e-12] = 1.0
    normal_xy = normal_xy / norm[:, None]

    left = centre_xy + left_d[:, None] * normal_xy
    right = centre_xy + right_d[:, None] * normal_xy
    width = left_d - right_d

    reason = ""
    if np.any(width <= 0.0):
        reason = "crossed"
    elif np.any(width < float(min_width)) or np.any(width > float(max_width)):
        reason = "width_out_of_range"

    return LaneFit(
        centerline=centre_xy,
        left=left,
        right=right,
        width=width,
        arclength=samples,
        valid=reason == "",
        reason=reason,
    )
