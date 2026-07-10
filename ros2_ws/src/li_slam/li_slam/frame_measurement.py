#!/usr/bin/env python3
"""
frame_measurement.py
──────────────────────
Phase 6 — Observer Interface.

This is THE boundary between the frontend (ROS2, OpenCV, camera-specific
code) and the observer backend (pure Section-4.2 math, no OpenCV, no
ROS2). Nothing on either side should reach across this boundary directly
— the frontend builds a FrameMeasurement and hands it over; the observer
only ever sees FrameMeasurement objects.

Deliberately defined with plain dataclasses (no numpy subclassing, no
ROS message types) so it's trivially serializable/testable and doesn't
drag ROS2 or OpenCV imports into the observer package.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class LandmarkMeasurement:
    """
    One landmark's measurement for a single frame. Mirrors the spec from
    the roadmap doc, minus fields the observer doesn't need directly
    (e.g. raw descriptor is kept for downstream re-identification/
    debugging, not because the observer math uses it).
    """
    id:               int
    pixel:            tuple                  # (u, v)
    descriptor:       np.ndarray              # (32,) uint8, ORB
    depth:            float                   # meters
    position_camera:  np.ndarray | None       # (X, Y, Z), camera frame
    track_age:        int
    confidence:       float                   # 0..1, see LandmarkManager._estimate_quality


@dataclass
class FrameMeasurement:
    """
    Everything the observer needs from a single image frame:
      - which landmarks were seen, where, and how confidently
      - which IMU samples arrived since the last frame
      - the camera intrinsics in effect for this frame

    gyro_samples / accel_samples are lists of (timestamp, value) tuples,
    chronologically ordered, value shape (3,). They may be empty if no
    IMU messages arrived in this window (shouldn't normally happen given
    IMU rates >> camera rate, but the observer's propagation step should
    handle an empty list gracefully rather than assuming a fixed count).
    """
    timestamp:         float
    landmarks:         list                   # list[LandmarkMeasurement]
    gyro_samples:      list                   # list[(float, np.ndarray)]
    accel_samples:     list                   # list[(float, np.ndarray)]
    camera_intrinsics: object                  # CameraIntrinsics or None

    def num_landmarks(self) -> int:
        return len(self.landmarks)

    def __repr__(self):
        return (f"FrameMeasurement(t={self.timestamp:.3f}, "
                f"landmarks={len(self.landmarks)}, "
                f"gyro={len(self.gyro_samples)}, "
                f"accel={len(self.accel_samples)})")
