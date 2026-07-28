#!/usr/bin/env python3
"""Tests for the temporal plane filter and its innovation gate."""

import numpy as np
from ransac_seg.ground_plane import PlaneModel
from ransac_seg.plane_filter import PlaneFilter


def make_plane(tilt_deg=0.0, offset=1.7, inlier_ratio=0.8, iterations=20):
    """Build a PlaneModel tilted by ``tilt_deg`` about the y axis."""
    tilt = np.radians(tilt_deg)
    normal = np.array([-np.sin(tilt), 0.0, np.cos(tilt)])
    return PlaneModel(
        normal=normal / np.linalg.norm(normal),
        offset=float(offset),
        inlier_ratio=inlier_ratio,
        iterations=iterations,
        refined=True,
    )


def test_first_estimate_is_taken_verbatim():
    """With no history there is nothing to blend against."""
    filt = PlaneFilter()
    model = make_plane(3.0)

    out = filt.update(model)
    assert out is model
    assert filt.last is model


def test_constant_plane_converges_and_stays():
    """Repeating one estimate settles exactly on it and does not drift."""
    filt = PlaneFilter(alpha=0.3)
    model = make_plane(4.0, offset=1.9)

    for _ in range(50):
        out = filt.update(model)

    np.testing.assert_allclose(out.normal, model.normal, atol=1e-9)
    assert abs(out.offset - model.offset) < 1e-6

    settled = filt.update(model)
    np.testing.assert_allclose(settled.normal, out.normal, atol=1e-12)


def test_twenty_degree_outlier_frame_is_rejected():
    """A wildly different fit is a bad fit, not a road that moved."""
    filt = PlaneFilter(alpha=0.5, max_tilt_jump_deg=5.0)
    good = make_plane(0.0)
    filt.update(good)

    out = filt.update(make_plane(20.0))
    assert out is filt.last
    np.testing.assert_allclose(out.normal, good.normal)


def test_offset_gate_rejects_a_vertical_jump():
    """A plane that teleports vertically is gated even when its tilt matches."""
    filt = PlaneFilter(max_offset_jump=0.5)
    filt.update(make_plane(0.0, offset=1.7))

    out = filt.update(make_plane(0.0, offset=3.0))
    assert out.offset == 1.7


def test_none_holds_then_resets_after_the_miss_budget():
    """Missing fits hold the last plane, then give up rather than go stale."""
    filt = PlaneFilter(reset_after_misses=3)
    good = filt.update(make_plane(2.0))

    assert filt.update(None) is good
    assert filt.update(None) is good
    assert filt.update(None) is None
    assert filt.last is None


def test_reset_clears_state():
    """An explicit reset forgets the accumulated plane."""
    filt = PlaneFilter()
    filt.update(make_plane(1.0))
    filt.reset()
    assert filt.last is None


def test_filtering_reduces_variance_of_a_noisy_sequence():
    """The whole point of the filter: less frame-to-frame swing."""
    rng = np.random.default_rng(11)
    filt = PlaneFilter(alpha=0.3)

    raw_tilts, filtered_tilts = [], []
    for _ in range(30):
        model = make_plane(tilt_deg=2.0 + rng.normal(0.0, 0.4))
        raw_tilts.append(model.tilt_deg)
        filtered_tilts.append(filt.update(model).tilt_deg)

    # Skip the warm-up frames: the first estimate is taken verbatim, so the
    # filter has not yet had anything to average against.
    raw_var = float(np.var(raw_tilts[5:]))
    filtered_var = float(np.var(filtered_tilts[5:]))
    assert filtered_var < raw_var
    assert filtered_var < 0.5 * raw_var


def test_gated_misses_also_expire():
    """A run of rejected fits eventually resets, so a real change can take."""
    filt = PlaneFilter(max_tilt_jump_deg=2.0, reset_after_misses=2)
    filt.update(make_plane(0.0))

    filt.update(make_plane(20.0))
    assert filt.update(make_plane(20.0)) is None

    fresh = make_plane(20.0)
    assert filt.update(fresh) is fresh
