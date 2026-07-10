#!/usr/bin/env python3
"""
imu_buffer.py
──────────────
Phase 4 — IMU integration (buffering stage only; integration/propagation
math belongs to the observer backend, Phase 8, not here).

Responsibility of this module: collect (timestamp, gyro, accel) samples
as they arrive asynchronously from the gyro/accel topics, and on demand
hand back exactly the samples that fall between two image timestamps,
then discard everything older than that window.

Why two deques, not one
────────────────────────
The D455 publishes gyro and accel on separate topics at separate rates
(commonly gyro ~200-400Hz, accel ~63-250Hz depending on profile). They
are NOT guaranteed to arrive interleaved or at matching timestamps. We
buffer them independently and only align them by timestamp at extraction
time, rather than trying to force-pair them at callback time.

Why deque
──────────
O(1) append on the right, O(1) popleft on the left. We append on every
IMU callback (potentially several hundred Hz combined) and trim from the
left every image frame (~30-60Hz) — a deque is the right structure for
both ends being hot.
"""

import bisect
from collections import deque
import numpy as np


class IMUSample:
    """A single timestamped IMU reading (gyro OR accel, not both)."""
    __slots__ = ('timestamp', 'value')

    def __init__(self, timestamp: float, value: np.ndarray):
        self.timestamp = timestamp          # seconds, float
        self.value     = value              # ndarray shape (3,)


class IMUBuffer:
    """
    Buffers gyro and accel samples and extracts the window relevant to
    the interval between two consecutive image frames.

    Usage
    -----
        buf = IMUBuffer()
        # in gyro_callback:
        buf.add_gyro(t, np.array([wx, wy, wz]))
        # in accel_callback:
        buf.add_accel(t, np.array([ax, ay, az]))

        # in rgb_callback, once per frame:
        gyro_samples, accel_samples = buf.extract_window(prev_image_t, curr_image_t)
        # gyro_samples / accel_samples: list[IMUSample], chronological

    Samples strictly older than the window's lower bound are dropped on
    every extraction call, so memory stays bounded by "however much IMU
    data arrives in roughly one frame period", not by total session time.
    """

    def __init__(self, max_buffer_seconds: float = 2.0):
        """
        max_buffer_seconds: hard safety cap. If image frames stop arriving
        (e.g. RGB callback stalls) IMU data would otherwise accumulate
        unboundedly; this caps it regardless of frame timing.
        """
        self._gyro:  deque = deque()
        self._accel: deque = deque()
        self._max_buffer_seconds = max_buffer_seconds

    # ------------------------------------------------------------------ public

    def add_gyro(self, timestamp: float, gyro: np.ndarray):
        self._gyro.append(IMUSample(timestamp, gyro))
        self._trim(self._gyro, timestamp)

    def add_accel(self, timestamp: float, accel: np.ndarray):
        self._accel.append(IMUSample(timestamp, accel))
        self._trim(self._accel, timestamp)

    def extract_window(self, t_start: float, t_end: float):
        """
        Returns (gyro_samples, accel_samples): all samples with
        t_start < timestamp <= t_end, chronologically ordered.

        Samples with timestamp <= t_start are then dropped from the
        buffer (they belong to a window that has already been consumed
        by an earlier frame). Samples with timestamp > t_end are left
        in place — they belong to the NEXT window.

        Uses bisect since both deques are append-only / time-ordered by
        construction (IMU samples arrive in order), so timestamps are
        sorted and binary search is valid.
        """
        gyro_samples  = self._extract_and_trim(self._gyro,  t_start, t_end)
        accel_samples = self._extract_and_trim(self._accel, t_start, t_end)
        return gyro_samples, accel_samples

    def clear(self):
        self._gyro.clear()
        self._accel.clear()

    # ----------------------------------------------------------------- private

    def _extract_and_trim(self, buf: deque, t_start: float, t_end: float):
        timestamps = [s.timestamp for s in buf]

        lo = bisect.bisect_right(timestamps, t_start)   # first idx with t > t_start
        hi = bisect.bisect_right(timestamps, t_end)      # first idx with t > t_end

        window = list(buf)[lo:hi]

        # Drop everything up to and including t_start — it belongs to a
        # window that's already been (or will never be) consumed.
        for _ in range(lo):
            buf.popleft()

        return window

    def _trim(self, buf: deque, latest_timestamp: float):
        """Safety trim: drop anything older than max_buffer_seconds."""
        cutoff = latest_timestamp - self._max_buffer_seconds
        while buf and buf[0].timestamp < cutoff:
            buf.popleft()
