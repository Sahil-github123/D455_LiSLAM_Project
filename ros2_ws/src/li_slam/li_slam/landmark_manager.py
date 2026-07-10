#!/usr/bin/env python3
"""
landmark_manager.py
────────────────────
Owns the Landmark data structure and its lifecycle (Phase 1 cleanup,
Phase 2 depth integration, Phase 3 full landmark fields).

This module has ZERO ROS2 dependencies and ZERO OpenCV display calls —
it only knows about pixels, descriptors, depth, and 3D geometry. This
keeps it testable in isolation and reusable if you ever swap the camera
driver or move to a different frontend.

Identity model (unchanged from before, now extended)
──────────────────────────────────────────────────────
A Landmark is tracked while KLT can follow it (`tracked=True`). When KLT
loses it, it goes dormant (`tracked=False`) but stays in the dictionary
for REID_SEARCH_RADIUS / STALE_TIMEOUT, during which a new detection can
re-claim its identity via descriptor + spatial matching. This is what
prevents the same physical point from getting a new ID every time it's
briefly occluded or leaves frame.

Phase 2 adds: every active landmark also carries depth and a 3D camera-
frame position, computed from the latest depth image + intrinsics.

Phase 3 adds: track_age, num_observations, descriptor_quality, active
flag (distinct from tracked — see below), valid_depth flag.
"""

import time
import numpy as np
import cv2


# ──────────────────────────────────────────────────────────────────────────────
# Landmark
# ──────────────────────────────────────────────────────────────────────────────

class Landmark:
    """
    A persistent map entity — the SLAM-relevant unit. One id for its
    entire life, whether currently tracked, dormant, or pending pruning.

    Fields
    ------
    id                 : int, unique for the node's lifetime
    descriptor         : (32,) uint8 ORB descriptor, refreshed on each
                         successful re-identification (not every frame —
                         while KLT-tracked we don't recompute it, since
                         we already know the identity)
    pixel              : (u, v) float — current or last-known image position
    depth              : float, meters. 0.0 / NaN if invalid
    position_camera    : (X, Y, Z) ndarray in the camera optical frame,
                         or None if depth is currently invalid
    track_age          : int, frames survived since first detected
                         (NOT reset on a brief loss+reacquire — this is
                         the landmark's total lifetime, distinct from
                         "frames continuously tracked")
    num_observations   : int, times this landmark has been freshly
                         measured (KLT update OR re-identification)
    tracked            : bool, is KLT actively following it RIGHT NOW
    active             : bool, is it eligible to be reported to the
                         observer this frame (tracked AND valid_depth)
    last_seen          : float, time.time() of last update
    valid_depth        : bool, whether `depth`/`position_camera` are usable
    descriptor_quality : float in [0,1], rough confidence signal — see
                         `_estimate_quality()`. Cheap heuristic, not a
                         replacement for proper covariance, but useful
                         for gating which landmarks the observer should
                         trust most when memory/compute is tight.
    """
    __slots__ = (
        'id', 'descriptor', 'pixel',
        'depth', 'position_camera', 'valid_depth',
        'track_age', 'num_observations',
        'tracked', 'active', 'last_seen',
        'descriptor_quality',
    )

    def __init__(self, landmark_id: int, descriptor: np.ndarray, pixel: tuple):
        self.id                 = landmark_id
        self.descriptor         = descriptor
        self.pixel              = pixel

        self.depth              = 0.0
        self.position_camera    = None
        self.valid_depth        = False

        self.track_age          = 1
        self.num_observations   = 1

        self.tracked            = True
        self.active             = False     # set True once depth is valid
        self.last_seen           = time.time()

        self.descriptor_quality = 1.0

    def __repr__(self):
        return (f"Landmark(id={self.id}, pixel={self.pixel}, "
                f"depth={self.depth:.2f}, tracked={self.tracked}, "
                f"age={self.track_age})")


# ──────────────────────────────────────────────────────────────────────────────
# Camera intrinsics (small helper struct — Phase 2.3)
# ──────────────────────────────────────────────────────────────────────────────

class CameraIntrinsics:
    """
    Pinhole intrinsics needed to back-project (u,v,depth) -> (X,Y,Z).

    Populate this from the D455's /camera/camera/color/camera_info topic
    (CameraInfo.k = [fx, 0, cx, 0, fy, cy, 0, 0, 1]) rather than
    hardcoding — intrinsics differ per device and per resolution.
    """
    __slots__ = ('fx', 'fy', 'cx', 'cy', 'depth_scale')

    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 depth_scale: float = 0.001):
        """
        depth_scale: multiply raw depth image values by this to get meters.
        RealSense D455 typically publishes depth in millimeters as uint16,
        so depth_scale=0.001 converts mm -> m. Confirm against your actual
        depth encoding (check the topic's encoding field).
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale = depth_scale

    def backproject(self, u: float, v: float, raw_depth: float) -> tuple:
        """
        (u, v, raw_depth_value) -> (X, Y, Z) in camera frame, meters.
        Standard pinhole back-projection:
            Z = depth
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
        """
        z = raw_depth * self.depth_scale
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return (x, y, z)


# ──────────────────────────────────────────────────────────────────────────────
# Landmark Manager
# ──────────────────────────────────────────────────────────────────────────────

class LandmarkManager:
    """
    Owns every Landmark and orchestrates:

      1. mark_tracked() / mark_lost()    — called per-point from KLT results
      2. integrate_new_detections()      — re-identify or create, for the
                                            (small) set of freshly detected
                                            points during a redetect pass
      3. update_depth()                  — Phase 2: stamp every currently
                                            tracked landmark with depth +
                                            3D camera-frame position
      4. prune_stale() / prune_inactive() — Phase 1 cleanup: drop dead
                                            landmarks so the dict doesn't
                                            grow unboundedly over a long
                                            session
      5. get_active_landmarks()          — Phase 6 prep: the exact list
                                            the FrameMeasurement consumes

    Matching is intentionally restricted in scope (see integrate_new_
    detections) so cost stays bounded by "how much tracking churned this
    frame", not by total accumulated landmark count.
    """

    MATCH_DISTANCE_THRESH = 64       # Hamming distance gate (ORB, 0-256)
    REID_SEARCH_RADIUS    = 30.0     # px — spatial gate for re-identification
    STALE_TIMEOUT         = 60.0      # s  — prune a lost landmark after this long
    MIN_TRACK_AGE_ACTIVE  = 2        # frames — require at least this much
                                      # history before reporting to the
                                      # observer, to suppress one-frame
                                      # detection noise

    def __init__(self):
        self.landmarks: dict = {}
        self._next_id:  int = 0
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # ------------------------------------------------------------------ public:
    #                          creation / tracking lifecycle

    def create(self, descriptor: np.ndarray, pixel: tuple) -> int:
        lm = Landmark(self._next_id, descriptor, pixel)
        self.landmarks[self._next_id] = lm
        lm_id = self._next_id
        self._next_id += 1
        return lm_id

    def mark_tracked(self, landmark_id: int, pixel: tuple):
        """Called every frame for every landmark KLT successfully followed."""
        lm = self.landmarks.get(landmark_id)
        if lm is None:
            return
        lm.pixel            = pixel
        lm.track_age       += 1
        lm.num_observations += 1
        lm.tracked           = True
        lm.last_seen         = time.time()

    def mark_lost(self, landmark_id: int):
        """Called when KLT drops a previously-tracked landmark."""
        lm = self.landmarks.get(landmark_id)
        if lm is not None:
            lm.tracked = False
            lm.active  = False

    # ------------------------------------------------------------------ public:
    #                          re-identification / new detections

    def integrate_new_detections(self, keypoints: list,
                                  descriptors: np.ndarray) -> list:
        """
        Given freshly-detected keypoints (already masked to exclude regions
        near actively-tracked landmarks), either re-identify each as a
        returning landmark or register it as new.

        Returns list of (landmark_id, pixel, is_new).
        """
        if descriptors is None or len(descriptors) == 0:
            return []

        results = []

        candidate_ids = [lid for lid, lm in self.landmarks.items()
                          if not lm.tracked]

        best_for_query = {}
        if candidate_ids:
            cand_descs = np.array(
                [self.landmarks[lid].descriptor for lid in candidate_ids],
                dtype=np.uint8)
            cand_pixels = np.array(
                [self.landmarks[lid].pixel for lid in candidate_ids],
                dtype=np.float32)

            matches = self._matcher.match(descriptors, cand_descs)
            for m in matches:
                if m.distance > self.MATCH_DISTANCE_THRESH:
                    continue
                kp_xy   = np.array(keypoints[m.queryIdx].pt, dtype=np.float32)
                cand_xy = cand_pixels[m.trainIdx]
                if np.linalg.norm(kp_xy - cand_xy) > self.REID_SEARCH_RADIUS:
                    continue
                prev = best_for_query.get(m.queryIdx)
                if prev is None or m.distance < prev.distance:
                    best_for_query[m.queryIdx] = m

        claimed_landmark_ids = set()

        for i, (kp, des) in enumerate(zip(keypoints, descriptors)):
            m = best_for_query.get(i)
            if m is not None and candidate_ids[m.trainIdx] not in claimed_landmark_ids:
                lm_id = candidate_ids[m.trainIdx]
                lm = self.landmarks[lm_id]
                lm.descriptor          = des
                lm.pixel               = kp.pt
                lm.tracked              = True
                lm.num_observations    += 1
                lm.last_seen            = time.time()
                lm.descriptor_quality   = self._estimate_quality(m.distance)
                claimed_landmark_ids.add(lm_id)
                results.append((lm_id, kp.pt, False))
            else:
                new_id = self.create(des, kp.pt)
                results.append((new_id, kp.pt, True))

        return results

    # ------------------------------------------------------------------ public:
    #                          Phase 2 — depth integration

    def update_depth(self, depth_image: np.ndarray,
                      intrinsics: CameraIntrinsics):
        """
        Stamp every currently-tracked landmark with depth and 3D camera-
        frame position. Call this once per frame, after `_propagate()`
        but before reporting active landmarks to the observer.

        depth_image : raw depth frame (same convention as the camera
                      driver publishes — typically uint16 mm for D455)
        intrinsics  : CameraIntrinsics for back-projection

        A landmark whose pixel falls outside the depth image bounds, or
        whose raw depth value is 0 (D455's "invalid" sentinel), is marked
        valid_depth=False and active=False — it still exists and is still
        tracked, it's just not usable as a 3D measurement this frame.
        """
        h, w = depth_image.shape[:2]

        for lm in self.landmarks.values():
            if not lm.tracked:
                continue
            # print(lm.pixel)

            u, v = lm.pixel
            ui, vi = int(round(u)), int(round(v))
            # print(ui, vi)

            if not (0 <= ui < w and 0 <= vi < h):
                lm.valid_depth = False
                lm.active = False
                continue

            raw_depth = depth_image[vi, ui]
            if raw_depth == 0:
                lm.valid_depth = False
                lm.active = False
                continue
            # print(raw_depth)

            x, y, z = intrinsics.backproject(u, v, float(raw_depth))
            lm.depth = z
            lm.position_camera = np.array([x, y, z], dtype=np.float32)
            lm.valid_depth = True
            lm.active = lm.track_age >= self.MIN_TRACK_AGE_ACTIVE

    # ------------------------------------------------------------------ public:
    #                          cleanup (Phase 1) / reporting (Phase 6 prep)

    def prune_stale(self):
        """Drop landmarks that have been lost (untracked) for too long."""
        now = time.time()
        stale = [lid for lid, lm in self.landmarks.items()
                 if not lm.tracked and (now - lm.last_seen) > self.STALE_TIMEOUT]
        for lid in stale:
            del self.landmarks[lid]

    def tracked_count(self) -> int:
        return sum(1 for lm in self.landmarks.values() if lm.tracked)

    def active_count(self) -> int:
        return sum(1 for lm in self.landmarks.values() if lm.active)

    def get_active_landmarks(self) -> list:
        """
        Returns the list of Landmark objects eligible to be sent to the
        observer this frame: currently tracked, valid depth, and past
        the minimum-age noise-suppression threshold.

        This is the exact set Phase 6's FrameMeasurement will consume.
        """
        return [lm for lm in self.landmarks.values() if lm.active]

    # ----------------------------------------------------------------- private

    @staticmethod
    def _estimate_quality(match_distance: float) -> float:
        """
        Crude descriptor-quality heuristic: 0 at the match-distance gate,
        1.0 at a perfect (distance=0) match. Not a statistically rigorous
        confidence — just enough signal to let the observer (eventually)
        downweight shakier re-identifications if desired.
        """
        thresh = LandmarkManager.MATCH_DISTANCE_THRESH
        return max(0.0, 1.0 - (match_distance / thresh))
