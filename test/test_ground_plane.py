#!/usr/bin/env python3
"""
Unit tests for RANSAC ground-plane segmentation.

Pure numpy — these run without a ROS 2 install.
"""

import numpy as np
import pytest
from ransac_seg.ground_plane import PlaneModel, fit_plane_ransac, segment_ground


def make_plane_cloud(
    normal, offset, n_inliers=3000, n_outliers=1500, noise=0.01, extent=25.0, seed=7
):
    """Build a cloud of plane points plus obstacle clutter above the plane."""
    rng = np.random.default_rng(seed)
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    # Two in-plane basis vectors, so inliers are sampled on the plane itself.
    seed_vec = np.array([1.0, 0.0, 0.0])
    if abs(normal @ seed_vec) > 0.9:
        seed_vec = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, seed_vec)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    coeffs = rng.uniform(-extent, extent, size=(n_inliers, 2))
    origin = -offset * normal
    inliers = origin + coeffs[:, :1] * u + coeffs[:, 1:] * v
    inliers += rng.normal(0.0, noise, size=inliers.shape)

    # Outliers sit clearly above the surface: cars, poles, foliage.
    outliers = origin + rng.uniform(-extent, extent, size=(n_outliers, 2)) @ np.vstack(
        [u, v]
    )
    outliers += normal * rng.uniform(0.6, 4.0, size=(n_outliers, 1))

    cloud = np.vstack([inliers, outliers])
    truth = np.zeros(cloud.shape[0], dtype=bool)
    truth[:n_inliers] = True

    shuffle = rng.permutation(cloud.shape[0])
    return cloud[shuffle], truth[shuffle]


def test_recovers_level_plane():
    """A flat road at z = -1.7 is recovered to centimetre accuracy."""
    cloud, truth = make_plane_cloud([0.0, 0.0, 1.0], 1.7)
    model, mask = fit_plane_ransac(cloud)

    assert model is not None
    assert model.tilt_deg < 1.0
    assert model.offset == pytest.approx(1.7, abs=0.02)
    # Recall and precision against the known ground truth.
    assert np.count_nonzero(mask & truth) / np.count_nonzero(truth) > 0.95
    assert np.count_nonzero(mask & ~truth) == 0


def test_recovers_sloped_plane_where_z_threshold_fails():
    """On a 10 degree slope the fixed z filter breaks and RANSAC does not."""
    normal = np.array([np.sin(np.radians(10.0)), 0.0, np.cos(np.radians(10.0))])
    cloud, truth = make_plane_cloud(normal, 1.7)

    model, mask = fit_plane_ransac(cloud)
    assert model is not None
    assert model.tilt_deg == pytest.approx(10.0, abs=1.0)
    assert np.count_nonzero(mask & truth) / np.count_nonzero(truth) > 0.95

    # The old behaviour, for contrast: a fixed height cannot follow the slope.
    legacy = cloud[:, 2] < 0.2
    legacy_recall = np.count_nonzero(legacy & truth) / np.count_nonzero(truth)
    assert legacy_recall < 0.95


def test_normal_is_always_oriented_up():
    """Sign of the fitted normal does not depend on sample draw order."""
    cloud, _ = make_plane_cloud([0.0, 0.0, -1.0], -1.7)
    model, _ = fit_plane_ransac(cloud)
    assert model is not None
    assert model.normal[2] > 0.0


def test_refit_beats_minimal_sample():
    """Total-least-squares refit is applied and tightens the fit."""
    cloud, _ = make_plane_cloud([0.0, 0.02, 1.0], 1.7, noise=0.02)
    model, _ = fit_plane_ransac(cloud, refine=True)
    assert model is not None and model.refined

    coarse, _ = fit_plane_ransac(cloud, refine=False)
    assert model.inlier_ratio >= coarse.inlier_ratio


def test_vertical_wall_is_rejected_by_tilt_gate():
    """A dominant vertical surface must not be mistaken for the road."""
    rng = np.random.default_rng(3)
    wall = np.column_stack(
        [
            np.full(4000, 5.0) + rng.normal(0.0, 0.01, 4000),
            rng.uniform(-20.0, 20.0, 4000),
            rng.uniform(0.0, 6.0, 4000),
        ]
    )
    model, mask = fit_plane_ransac(wall, min_inlier_ratio=0.1)
    assert model is None
    assert not mask.any()


def test_returns_none_when_no_plane_dominates():
    """Uniform noise has no consensus set worth reporting."""
    rng = np.random.default_rng(11)
    blob = rng.uniform(-10.0, 10.0, size=(2000, 3))
    model, mask = fit_plane_ransac(blob)
    assert model is None
    assert not mask.any()


@pytest.mark.parametrize(
    "cloud",
    [
        np.empty((0, 3)),
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    ],
)
def test_degenerate_inputs_are_safe(cloud):
    """Too few points returns no model and a correctly sized empty mask."""
    model, mask = fit_plane_ransac(cloud)
    assert model is None
    assert mask.shape == (cloud.shape[0],)
    assert not mask.any()


def test_collinear_points_do_not_produce_a_plane():
    """Three collinear points are degenerate and must not define a plane."""
    line = np.column_stack(
        [
            np.linspace(0.0, 10.0, 500),
            np.zeros(500),
            np.zeros(500),
        ]
    )
    model, _ = fit_plane_ransac(line)
    assert model is None


def test_non_finite_rows_are_dropped_and_mask_stays_aligned():
    """Non-finite rows are ignored by the fit but never marked as ground."""
    cloud, truth = make_plane_cloud([0.0, 0.0, 1.0], 1.7, n_inliers=800, n_outliers=400)
    cloud = np.vstack([cloud, [np.nan, np.nan, np.nan], [np.inf, 0.0, 0.0]])

    model, mask = fit_plane_ransac(cloud)
    assert model is not None
    assert mask.shape == (cloud.shape[0],)
    assert not mask[-1] and not mask[-2]
    assert np.count_nonzero(mask) == pytest.approx(np.count_nonzero(truth), rel=0.1)


def test_is_deterministic_for_a_given_seed():
    """Same cloud and seed produce an identical segmentation."""
    cloud, _ = make_plane_cloud([0.01, 0.0, 1.0], 1.7)
    first, mask_a = fit_plane_ransac(cloud, seed=42)
    second, mask_b = fit_plane_ransac(cloud, seed=42)
    assert np.array_equal(mask_a, mask_b)
    assert first.normal.tolist() == second.normal.tolist()


def test_subsampled_scoring_matches_full_scoring():
    """Scoring on a subset finds the same plane as scoring on everything."""
    cloud, _ = make_plane_cloud([0.0, 0.05, 1.0], 1.7, n_inliers=12000, n_outliers=6000)
    subsampled, _ = fit_plane_ransac(cloud, max_scoring_points=2000)
    full, _ = fit_plane_ransac(cloud, max_scoring_points=0)
    assert subsampled is not None and full is not None
    assert float(subsampled.normal @ full.normal) > 0.999
    assert subsampled.offset == pytest.approx(full.offset, abs=0.02)


def test_segment_ground_returns_plane_on_normal_input():
    """The node-facing helper reports the model it used."""
    cloud, truth = make_plane_cloud([0.0, 0.0, 1.0], 1.7)
    mask, model = segment_ground(cloud)
    assert isinstance(model, PlaneModel)
    assert np.count_nonzero(mask & truth) / np.count_nonzero(truth) > 0.95


def test_segment_ground_falls_back_when_ransac_fails():
    """With no plane to find, callers still get the legacy height filter."""
    rng = np.random.default_rng(5)
    blob = rng.uniform(-10.0, 10.0, size=(2000, 3))
    mask, model = segment_ground(blob, fallback_height=0.2)
    assert model is None
    assert np.array_equal(mask, blob[:, 2] < 0.2)


def test_signed_distance_sign_convention():
    """Points above the plane have positive signed distance."""
    cloud, _ = make_plane_cloud([0.0, 0.0, 1.0], 1.7)
    model, _ = fit_plane_ransac(cloud)
    assert model.signed_distance_to([[0.0, 0.0, 0.0]])[0] > 0.0
    assert model.signed_distance_to([[0.0, 0.0, -3.0]])[0] < 0.0
    assert model.distance_to([[0.0, 0.0, -3.0]])[0] == pytest.approx(1.3, abs=0.05)
