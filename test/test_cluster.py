#!/usr/bin/env python3
"""
Unit tests for the voxel-linkage obstacle clustering.

Pure numpy/scipy with no ROS import, so they run on a bare checkout.
"""

import numpy as np
import pytest
from ransac_seg.cluster import Cluster, cluster_obstacles, voxel_downsample


def block(rng, centre, size, count):
    """Return ``count`` uniformly sampled points inside an axis-aligned block."""
    half = np.asarray(size, dtype=np.float64) / 2.0
    centre = np.asarray(centre, dtype=np.float64)
    return rng.uniform(centre - half, centre + half, size=(count, 3))


def test_voxel_downsample_collapses_and_maps_back():
    """Representatives are per-voxel centroids and inverse indexes into them."""
    points = np.array(
        [
            [0.01, 0.01, 0.01],
            [0.05, 0.05, 0.05],
            [5.00, 5.00, 5.00],
        ]
    )
    reps, inverse = voxel_downsample(points, voxel_size=0.2)

    assert reps.shape == (2, 3)
    assert inverse.shape == (3,)
    assert inverse[0] == inverse[1] and inverse[0] != inverse[2]
    np.testing.assert_allclose(reps[inverse[0]], [0.03, 0.03, 0.03])


def test_voxel_downsample_empty():
    """An empty cloud downsamples to empty arrays rather than raising."""
    reps, inverse = voxel_downsample(np.zeros((0, 3)))
    assert reps.shape == (0, 3)
    assert inverse.shape == (0,)


def test_voxel_downsample_survives_a_huge_extent():
    """A cloud spanning more cells than an int64 index takes the general path."""
    points = np.array([[0.0, 0.0, 0.0], [1e12, 1e12, 1e12], [1e12, 1e12, 1e12]])
    reps, inverse = voxel_downsample(points, voxel_size=0.2)

    assert reps.shape == (2, 3)
    assert inverse.tolist() == [0, 1, 1]


def test_three_separated_blocks():
    """Well-separated blocks give one cluster each with accurate centroids."""
    rng = np.random.default_rng(7)
    truth = [(2.0, 0.0, 1.0), (12.0, 6.0, 1.0), (20.0, -8.0, 1.0)]
    parts = [block(rng, c, (1.0, 1.0, 1.6), 400) for c in truth]
    points = np.vstack(parts)

    clusters = cluster_obstacles(points)

    assert len(clusters) == 3
    for want in truth:
        best = min(clusters, key=lambda c: np.linalg.norm(c.centroid - np.array(want)))
        assert np.linalg.norm(best.centroid - np.array(want)) < 0.2
    assert sum(c.count for c in clusters) == points.shape[0]


def test_close_blocks_merge():
    """Blocks 0.3 m apart are inside the 0.6 m link radius and become one."""
    rng = np.random.default_rng(11)
    left = block(rng, (5.0, 0.0, 1.0), (1.0, 1.0, 1.6), 300)
    right = block(rng, (5.0, 1.3, 1.0), (1.0, 1.0, 1.6), 300)

    clusters = cluster_obstacles(np.vstack([left, right]))

    assert len(clusters) == 1
    assert clusters[0].count == 600


def test_speck_rejected_by_min_points():
    """A five-point speck never reaches the minimum cluster size."""
    rng = np.random.default_rng(3)
    speck = block(rng, (4.0, 0.0, 1.0), (0.2, 0.2, 1.0), 5)
    assert cluster_obstacles(speck, min_points=12) == []


def test_pancake_rejected_by_min_height():
    """A 5 cm tall slab is ground scrap, not an obstacle."""
    rng = np.random.default_rng(5)
    pancake = block(rng, (6.0, 0.0, 0.02), (2.0, 2.0, 0.05), 500)
    assert cluster_obstacles(pancake, min_height=0.15) == []


def test_tall_artefact_rejected_by_max_height():
    """A column taller than max_height is an overhang artefact, not an obstacle."""
    rng = np.random.default_rng(9)
    column = block(rng, (8.0, 0.0, 3.0), (0.5, 0.5, 6.0), 800)
    assert cluster_obstacles(column, max_height=4.0) == []


def test_empty_input_returns_empty_list():
    """An empty cloud yields no clusters and does not raise."""
    assert cluster_obstacles(np.zeros((0, 3))) == []


def test_indices_index_the_input_array():
    """Cluster indices address the input rows and lie inside the reported box."""
    rng = np.random.default_rng(13)
    points = np.vstack(
        [
            block(rng, (3.0, 0.0, 1.0), (1.0, 1.0, 1.6), 300),
            block(rng, (15.0, 5.0, 1.0), (1.0, 1.0, 1.6), 300),
        ]
    )

    for cluster in cluster_obstacles(points):
        member = points[cluster.indices]
        assert member.shape[0] == cluster.count
        assert np.all(member >= cluster.min_bound - 1e-9)
        assert np.all(member <= cluster.max_bound + 1e-9)


def test_indices_survive_non_finite_rows():
    """Indices address the caller's rows even when NaN returns were dropped."""
    rng = np.random.default_rng(23)
    points = np.vstack(
        [
            np.full((5, 3), np.nan),
            block(rng, (3.0, 0.0, 1.0), (1.0, 1.0, 1.6), 300),
        ]
    )

    clusters = cluster_obstacles(points)

    assert len(clusters) == 1
    assert clusters[0].indices.min() >= 5
    assert np.all(np.isfinite(points[clusters[0].indices]))


def test_clusters_sorted_by_count_and_truncated():
    """Survivors come back largest first and truncated to max_clusters."""
    rng = np.random.default_rng(17)
    points = np.vstack(
        [
            block(rng, (3.0, 0.0, 1.0), (1.0, 1.0, 1.6), 100),
            block(rng, (10.0, 0.0, 1.0), (1.0, 1.0, 1.6), 400),
            block(rng, (17.0, 0.0, 1.0), (1.0, 1.0, 1.6), 250),
        ]
    )

    clusters = cluster_obstacles(points)
    assert [c.count for c in clusters] == [400, 250, 100]

    assert [c.count for c in cluster_obstacles(points, max_clusters=2)] == [400, 250]


def test_link_radius_below_lattice_diagonal_raises():
    """A radius shorter than the voxel diagonal would split single objects."""
    rng = np.random.default_rng(19)
    points = block(rng, (4.0, 0.0, 1.0), (1.0, 1.0, 1.6), 200)

    with pytest.raises(ValueError, match="fragment"):
        cluster_obstacles(points, voxel_size=0.5, link_radius=0.6)


def test_extent_and_height_properties():
    """Extent is the bounding-box size and height its vertical component."""
    cluster = Cluster(
        centroid=np.zeros(3),
        min_bound=np.array([-1.0, -2.0, 0.0]),
        max_bound=np.array([1.0, 2.0, 1.5]),
        count=4,
        indices=np.arange(4, dtype=np.int32),
    )
    np.testing.assert_allclose(cluster.extent, [2.0, 4.0, 1.5])
    assert cluster.height == pytest.approx(1.5)
