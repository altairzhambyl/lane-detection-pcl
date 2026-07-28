#!/usr/bin/env python3
"""
Unit tests for the centreline-frame lane boundary fit.

Every scenario the old min/max-per-x-column fitter was measured to fail on has
a test here: curvature, a parked car, a range gap. Pure numpy/scipy with no ROS
import, so they run on a bare checkout.

Note on the expected width: percentile edges are inset by construction. On a
road of uniform return density the 2nd percentile sits 2% of the width inside
the true edge, so a 6.00 m road reads 6.00 * (1 - 2 * 0.02) = 5.76 m. The
assertions below compare against that estimand and separately bound the total
deviation from the nominal width.
"""

import numpy as np
from ransac_seg.lane import LaneFit, fit_lane

TRUE_WIDTH = 6.00
# Inset of the default 2nd/98th percentile edges on a uniform-density road.
EXPECTED_WIDTH = TRUE_WIDTH * (1.0 - 2.0 * 0.02)


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
    # Offset perpendicular to the centreline, so the width is constant in the
    # road frame rather than measured along a fixed y axis.
    return np.column_stack([t - lateral * slope / norm, 0.01 * t**2 + lateral / norm])


def test_straight_road_width():
    """A straight constant-width road is measured to the percentile estimand."""
    fit = fit_lane(straight_road())

    assert fit.valid and fit.reason == ""
    # Measured this session: 0.018 m against the estimand, 0.258 m against the
    # nominal 6.00 m, the difference being the percentile inset.
    assert np.abs(fit.width - EXPECTED_WIDTH).max() < 0.15
    assert np.abs(fit.width - TRUE_WIDTH).max() < 0.35
    assert fit.left.shape == fit.right.shape == fit.centerline.shape
    assert np.all(np.diff(fit.arclength) > 0.0)


def test_curved_road_width():
    """Curvature no longer inflates the width; extents follow the centreline."""
    fit = fit_lane(curved_road())

    assert fit.valid
    # Measured this session across three seeds: 0.034-0.067 m against the
    # estimand, 0.274-0.307 m against the nominal 6.00 m. The fixed-x-column
    # fitter this replaced was reported at 8.06 m on the same geometry by the
    # audit that motivated the rewrite.
    assert np.abs(fit.width - EXPECTED_WIDTH).max() < 0.15
    assert np.abs(fit.width - TRUE_WIDTH).max() < 0.35


def test_curved_road_centerline_follows_the_arc():
    """The fitted centreline tracks ``y = 0.01 x^2``, not the chord."""
    fit = fit_lane(curved_road())
    x = fit.centerline[:, 0]
    residual = np.abs(fit.centerline[:, 1] - 0.01 * x**2)
    assert residual.max() < 0.5


def test_roadside_obstacle_does_not_drag_the_boundary():
    """A parked block leaking into the ground band cannot pull the edge out."""
    rng = np.random.default_rng(3)
    half = 3.5
    road = np.column_stack(
        [rng.uniform(0.0, 40.0, 120000), rng.uniform(-half, half, 120000)]
    )
    # An 8000-point block beside the road, 0-1.5 m tall; only the returns
    # inside the plane's inlier band reach the lane fitter. That leak is ~1%
    # of an affected bin, i.e. below edge_percentile, which is the property
    # under test — heavier contamination degrades the fit.
    bx = rng.uniform(12.0, 17.0, 8000)
    by = rng.uniform(-6.5, -4.5, 8000)
    bz = rng.uniform(0.0, 1.5, 8000)
    leaked = bz < 0.03
    ground = np.vstack([road, np.column_stack([bx[leaked], by[leaked]])])

    fit = fit_lane(ground)
    assert fit.valid

    x = fit.right[:, 0]
    window = (x >= 7.0) & (x <= 19.0)
    error = np.abs(fit.right[window, 1] + half)
    # Measured this session: 0.125 m. The min-per-column estimator on the same
    # cloud is off by 2.994 m over the same span.
    assert error.max() < 0.30

    column = np.floor(ground[:, 0] / 2.0).astype(int)
    worst_min_based = max(
        abs(ground[column == c, 1].min() + half)
        for c in np.unique(column)
        if 7.0 <= (c + 0.5) * 2.0 <= 19.0
    )
    assert worst_min_based > 1.0


def test_gap_is_not_bridged():
    """A range gap truncates the fit instead of splining across empty road."""
    rng = np.random.default_rng(4)
    near = np.column_stack(
        [rng.uniform(0.0, 14.0, 20000), rng.uniform(-3.0, 3.0, 20000)]
    )
    far = np.column_stack(
        [rng.uniform(30.0, 40.0, 20000), rng.uniform(-3.0, 3.0, 20000)]
    )

    fit = fit_lane(np.vstack([near, far]))

    assert fit.valid
    # The longer run (14 m of near road) survives; nothing is drawn over the
    # 16 m hole, which is what a spline through both runs would have done.
    assert fit.centerline[:, 0].max() < 16.0
    assert fit.centerline[:, 0].min() < 4.0


def test_too_few_points():
    """Fewer than three bins worth of returns is reported, not fitted."""
    fit = fit_lane(straight_road(n=10))
    assert isinstance(fit, LaneFit)
    assert not fit.valid and fit.reason == "too_few_points"
    assert fit.centerline.shape == (0, 2) and fit.width.shape == (0,)


def test_short_run_reports_gap():
    """A road covering fewer than three bins cannot support a fit."""
    fit = fit_lane(straight_road(n=2000, half_width=0.5, length=3.0))
    assert not fit.valid and fit.reason == "gap"


def test_crossed_boundaries_are_rejected():
    """Boundaries that cross give reason 'crossed' rather than negative width."""
    # Percentiles above the median swap the two edges, which is the only way
    # to make a boundary pair cross on synthetic data.
    fit = fit_lane(straight_road(), edge_percentile=60.0)
    assert not fit.valid and fit.reason == "crossed"


def test_width_gate():
    """A width outside the configured band invalidates the fit."""
    assert fit_lane(straight_road(), min_width=8.0).reason == "width_out_of_range"
    assert fit_lane(straight_road(), max_width=4.0).reason == "width_out_of_range"


def test_sharp_bend_does_not_raise_or_return_garbage():
    """A road bending past 90 degrees stays finite and single valued."""
    rng = np.random.default_rng(5)
    theta = rng.uniform(0.0, np.radians(120.0), 60000)
    radius = 20.0 + rng.uniform(-3.0, 3.0, 60000)
    arc = np.column_stack([radius * np.sin(theta), radius * (1.0 - np.cos(theta))])

    fit = fit_lane(arc)

    # A degree-2 centreline cannot represent a 120 degree bend, so the fit is
    # expected to fail its width gate — but it must fail by reporting, never by
    # raising or by emitting a multivalued curve.
    assert isinstance(fit, LaneFit)
    assert np.all(np.isfinite(fit.centerline))
    assert np.all(np.isfinite(fit.width))
    assert np.all(np.diff(fit.arclength) > 0.0)


def test_three_column_input_is_accepted():
    """An (N, 3) ground cloud is sliced to xy rather than rejected."""
    xy = straight_road(n=20000)
    xyz = np.column_stack([xy, np.zeros(xy.shape[0])])
    assert fit_lane(xyz).valid


def test_empty_input():
    """An empty cloud returns an invalid fit, not an exception."""
    fit = fit_lane(np.zeros((0, 3)))
    assert not fit.valid and fit.reason == "too_few_points"
