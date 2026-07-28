#!/usr/bin/env python3
"""
Temporal low-pass filter over successive RANSAC plane estimates.

RANSAC redraws its minimal samples every frame, so consecutive fits of the
same physical road differ slightly. That jitter moves the plane vertically at
range by an amount comparable to the inlier band itself, which makes ground
membership flicker for far-field points and makes any downstream rasterisation
strobe. Smoothing the normal and offset across frames costs nothing and keeps
the map stable.

An innovation gate sits in front of the blend: a fit that disagrees violently
with the accumulated plane is a bad fit (a wall, a truck flank, a frame where
the road left the field of view), not a real road change, so it is dropped
rather than averaged in. Pure numpy; no ROS import, so planners and CI can
use it without a ROS install.
"""

import numpy as np

from ransac_seg.ground_plane import PlaneModel


def _tilt_between_deg(a, b):
    """Return the angle in degrees between two unit normals."""
    return float(np.degrees(np.arccos(np.clip(float(a @ b), -1.0, 1.0))))


class PlaneFilter:
    """Low-pass filter over successive PlaneModel estimates with an innovation gate."""

    def __init__(
        self,
        alpha=0.3,
        max_tilt_jump_deg=5.0,
        max_offset_jump=0.5,
        reset_after_misses=5,
    ):
        """Configure blend weight, gate thresholds and the miss budget."""
        self.alpha = float(alpha)
        self.max_tilt_jump_deg = float(max_tilt_jump_deg)
        self.max_offset_jump = float(max_offset_jump)
        self.reset_after_misses = int(reset_after_misses)
        self._last = None
        self._misses = 0

    @property
    def last(self):
        """Return the most recent filtered plane, or ``None``."""
        return self._last

    def reset(self):
        """Forget the accumulated plane and the miss counter."""
        self._last = None
        self._misses = 0

    def update(self, model):
        """
        Fold one estimate into the filter and return the plane to use.

        ``None`` (or a gated-out estimate) holds the last good plane and counts
        as a miss; after ``reset_after_misses`` consecutive misses the filter
        gives up and returns ``None`` so callers fall back rather than steering
        by a stale road.
        """
        if model is None:
            return self._miss()

        if self._last is None:
            self._last = model
            self._misses = 0
            return self._last

        tilt_jump = _tilt_between_deg(self._last.normal, model.normal)
        offset_jump = abs(model.offset - self._last.offset)
        if tilt_jump > self.max_tilt_jump_deg or offset_jump > self.max_offset_jump:
            return self._miss()

        blended = (1.0 - self.alpha) * self._last.normal + self.alpha * model.normal
        norm = float(np.linalg.norm(blended))
        if norm < 1e-9:
            # Opposed normals cannot be averaged; trust the fresh measurement.
            return self._accept(model)

        self._last = PlaneModel(
            normal=blended / norm,
            offset=(1.0 - self.alpha) * self._last.offset + self.alpha * model.offset,
            inlier_ratio=model.inlier_ratio,
            iterations=model.iterations,
            refined=model.refined,
        )
        self._misses = 0
        return self._last

    def _accept(self, model):
        """Take a model verbatim as the new filter state."""
        self._last = model
        self._misses = 0
        return self._last

    def _miss(self):
        """Count a missing or rejected estimate and return the held plane."""
        self._misses += 1
        if self._misses >= self.reset_after_misses:
            self.reset()
        return self._last
