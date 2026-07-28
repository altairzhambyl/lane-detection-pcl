#!/usr/bin/env python3
"""
RANSAC ground-plane segmentation for LiDAR point clouds.

Splits a raw point cloud into road surface and everything else by fitting a
plane with RANSAC instead of thresholding on a fixed z height. A fixed
threshold only works when the sensor is perfectly level and the road is flat;
on a slope, a banked corner, or with any mounting pitch it either eats the
road or keeps the curb.

The estimator is plain numpy: hypotheses come from minimal 3-point samples,
scoring is vectorised over the cloud, and the winning consensus set is refit
by total least squares (SVD) so the final plane uses every inlier rather than
the three points that happened to be drawn.

Frame assumption: coordinates are sensor/base frame with +z roughly up. The
tilt gate relies on this to reject walls and truck sides, which otherwise fit
a plane far better than the road does.
"""

from dataclasses import dataclass

import numpy as np

# Defaults tuned for a front-facing automotive LiDAR, distances in metres.
DEFAULT_DISTANCE_THRESHOLD = 0.20
DEFAULT_MAX_ITERATIONS = 200
DEFAULT_MIN_ITERATIONS = 12
DEFAULT_CONFIDENCE = 0.99
DEFAULT_MAX_TILT_DEG = 10.0
DEFAULT_MIN_INLIER_RATIO = 0.15
DEFAULT_MAX_SCORING_POINTS = 20000

_MIN_SAMPLE = 3


@dataclass(frozen=True)
class PlaneModel:
    """A fitted plane ``normal . x + offset = 0`` with unit, +z-oriented normal."""

    normal: np.ndarray
    offset: float
    inlier_ratio: float
    iterations: int
    refined: bool

    def distance_to(self, points):
        """Return the perpendicular distance from each point to the plane."""
        return np.abs(self.signed_distance_to(points))

    def signed_distance_to(self, points):
        """Return signed distance per point; positive is above the plane."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        return pts @ self.normal + self.offset

    @property
    def tilt_deg(self):
        """Return the angle in degrees between the plane normal and +z."""
        return float(np.degrees(np.arccos(np.clip(self.normal[2], -1.0, 1.0))))


def _as_points(points):
    """Coerce input to an ``(N, 3)`` float array, keeping every input row."""
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.empty((0, 3), dtype=float)
    return pts.reshape(-1, pts.shape[-1])[:, :3]


def _finite_view(points):
    """
    Return ``(all_rows, finite_rows, finite_mask)`` for a raw cloud.

    Masks handed back to callers must index their original array, so the
    row count of the input is kept even when NaN returns are dropped.
    """
    raw = _as_points(points)
    finite = np.isfinite(raw).all(axis=1)
    return raw, raw[finite], finite


def _orient_up(normal, offset):
    """Flip the plane so its normal points along +z, keeping the same surface."""
    if normal[2] < 0.0:
        return -normal, -offset
    return normal, offset


def _plane_from_sample(sample):
    """Fit a plane through three points, or return ``None`` if degenerate."""
    p0, p1, p2 = sample
    normal = np.cross(p1 - p0, p2 - p0)
    norm = float(np.linalg.norm(normal))
    # Scale the degeneracy test by the sample's own extent so it is unit-free:
    # three near-collinear or near-coincident points give a tiny cross product.
    scale = float(np.linalg.norm(p1 - p0) * np.linalg.norm(p2 - p0))
    if scale <= 0.0 or norm < 1e-6 * scale:
        return None
    normal = normal / norm
    return _orient_up(normal, -float(normal @ p0))


def _refit_total_least_squares(inliers):
    """Refit a plane to all inliers by SVD (total least squares)."""
    if inliers.shape[0] < _MIN_SAMPLE:
        return None
    centroid = inliers.mean(axis=0)
    # Smallest right-singular vector of the centred cloud is the direction of
    # least variance, i.e. the plane normal. Minimises perpendicular error,
    # unlike an ordinary least-squares fit of z on (x, y).
    _, singular_values, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
    if singular_values[-1] > singular_values[0] * 0.99:
        # Points fill a sphere rather than a sheet; no meaningful plane.
        return None
    normal = vh[-1]
    return _orient_up(normal, -float(normal @ centroid))


def _required_iterations(inlier_ratio, confidence, cap):
    """Return how many samples are still needed to hit ``confidence``."""
    if inlier_ratio <= 0.0:
        return cap
    all_inliers = inlier_ratio**_MIN_SAMPLE
    if all_inliers >= 1.0:
        return 1
    denominator = np.log1p(-all_inliers)
    if denominator >= 0.0:
        return cap
    return int(min(cap, np.ceil(np.log1p(-confidence) / denominator)))


def fit_plane_ransac(
    points,
    distance_threshold=DEFAULT_DISTANCE_THRESHOLD,
    max_iterations=DEFAULT_MAX_ITERATIONS,
    min_iterations=DEFAULT_MIN_ITERATIONS,
    confidence=DEFAULT_CONFIDENCE,
    max_tilt_deg=DEFAULT_MAX_TILT_DEG,
    min_inlier_ratio=DEFAULT_MIN_INLIER_RATIO,
    max_scoring_points=DEFAULT_MAX_SCORING_POINTS,
    refine=True,
    seed=0,
):
    """
    Fit a ground plane with RANSAC.

    Returns ``(model, inlier_mask)``. ``model`` is ``None`` when no plane meets
    ``min_inlier_ratio`` within the tilt gate, in which case the mask is all
    ``False``. The mask always indexes the caller's original array.
    """
    raw, pts, finite = _finite_view(points)
    total = pts.shape[0]
    empty_mask = np.zeros(raw.shape[0], dtype=bool)
    if total < _MIN_SAMPLE:
        return None, empty_mask

    rng = np.random.default_rng(seed)
    cos_limit = float(np.cos(np.radians(max_tilt_deg)))

    # Hypotheses are scored against a fixed random subset on large clouds:
    # consensus size is a ratio, so a few thousand points estimate it well and
    # the loop stays O(subset) instead of O(N) per iteration. The winning
    # plane is re-scored against every point at the end.
    if 0 < max_scoring_points < total:
        scoring_idx = rng.choice(total, size=max_scoring_points, replace=False)
        scoring_pts = pts[scoring_idx]
    else:
        scoring_pts = pts

    best_plane = None
    best_count = 0
    budget = max(int(min_iterations), 1)
    iterations = 0

    while iterations < min(budget, max_iterations):
        iterations += 1
        sample = pts[rng.choice(total, size=_MIN_SAMPLE, replace=False)]
        plane = _plane_from_sample(sample)
        if plane is None:
            continue
        normal, offset = plane
        if normal[2] < cos_limit:
            # Wall, vehicle flank or kerb face rather than road surface.
            continue

        count = int(
            np.count_nonzero(
                np.abs(scoring_pts @ normal + offset) <= distance_threshold
            )
        )
        if count > best_count:
            best_count = count
            best_plane = (normal, offset)
            needed = _required_iterations(
                count / scoring_pts.shape[0], confidence, max_iterations
            )
            budget = max(min_iterations, min(max_iterations, needed))

    if best_plane is None:
        return None, empty_mask

    normal, offset = best_plane
    mask = np.abs(pts @ normal + offset) <= distance_threshold
    refined = False

    if refine:
        better = _refit_total_least_squares(pts[mask])
        if better is not None and better[0][2] >= cos_limit:
            refined_mask = np.abs(pts @ better[0] + better[1]) <= distance_threshold
            # Keep the refit only if it explains at least as much of the cloud;
            # a refit skewed by a sloped shoulder can lose inliers.
            if np.count_nonzero(refined_mask) >= np.count_nonzero(mask):
                normal, offset = better
                mask = refined_mask
                refined = True

    inlier_count = int(np.count_nonzero(mask))
    inlier_ratio = inlier_count / total
    if inlier_ratio < min_inlier_ratio:
        return None, empty_mask

    model = PlaneModel(
        normal=normal,
        offset=float(offset),
        inlier_ratio=float(inlier_ratio),
        iterations=iterations,
        refined=refined,
    )

    if finite.all():
        return model, mask
    # Non-finite rows were dropped for the fit; re-align onto the input rows.
    full_mask = np.zeros(raw.shape[0], dtype=bool)
    full_mask[finite] = mask
    return model, full_mask


def segment_ground(
    points, distance_threshold=DEFAULT_DISTANCE_THRESHOLD, fallback_height=0.2, **kwargs
):
    """
    Split a cloud into road surface and the rest.

    Returns ``(ground_mask, model)``. When RANSAC finds no plausible plane —
    a cloud that is mostly obstacle, or too sparse — this falls back to the
    old fixed-height filter and returns ``model=None`` so callers can tell the
    difference and log it.
    """
    raw = _as_points(points)
    if raw.shape[0] == 0:
        return np.zeros(0, dtype=bool), None

    model, mask = fit_plane_ransac(
        points, distance_threshold=distance_threshold, **kwargs
    )
    if model is not None:
        return mask, model

    # NaN height must never pass as road, so send it to +inf, not to 0.
    heights = np.nan_to_num(raw[:, 2], nan=np.inf, posinf=np.inf, neginf=np.inf)
    return heights < fallback_height, None
