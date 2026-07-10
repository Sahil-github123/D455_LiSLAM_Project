#!/usr/bin/env python3
"""
data_logger.py
───────────────
Writes every sensor stream to a separate CSV file for inspection.

Files created in LOG_DIR (default: ~/slam_logs/):

    imu_gyro.csv        — every gyro sample, one row per callback
    imu_accel.csv       — every accel sample, one row per callback
    depth_stats.csv     — per-frame depth statistics (not the full image)
    landmarks.csv       — per-frame × per-landmark rows, full fields
    frame_summary.csv   — one row per RGB frame, aggregate counts + timing

All files are line-buffered and flushed once per RGB frame, so they are
readable in a text editor / tail -f during a live run without waiting for
the process to exit.
"""

import csv
import os


LOG_DIR = os.path.expanduser("~/slam_logs")

# ── column definitions ─────────────────────────────────────────────────────

_COLS = {
    "imu_gyro": [
        "timestamp", "wx", "wy", "wz",
    ],
    "imu_accel": [
        "timestamp", "ax", "ay", "az",
    ],
    "depth_stats": [
        "frame_id", "timestamp",
        "valid_pixels", "invalid_pixels", "valid_ratio",
        "depth_min_m", "depth_max_m", "depth_mean_m", "depth_std_m",
    ],
    "landmarks": [
        "frame_id", "timestamp",
        "landmark_id",
        "pixel_u", "pixel_v",
        "depth_m",
        "cam_x", "cam_y", "cam_z",
        "track_age", "num_observations", "confidence",
        "valid_depth", "tracked", "active",
    ],
    "frame_summary": [
        "frame_id", "timestamp",
        "total_landmarks", "active_landmarks", "tracked_landmarks",
        "gyro_samples_in_window", "accel_samples_in_window",
        "redetected", "fps",
    ],
}


class DataLogger:
    """
    Instantiate once in D455Interface.__init__().
    Call log_gyro / log_accel in their respective callbacks.
    Call log_depth_frame / log_landmarks / log_frame_summary in _rgb_callback.
    Call close() (or use as context manager) on node shutdown.
    """

    def __init__(self, log_dir: str = LOG_DIR):
        os.makedirs(log_dir, exist_ok=True)
        self._log_dir  = log_dir
        self._frame_id = 0

        # Open all files explicitly; store (file_handle, csv_writer) pairs
        self._files: dict = {}
        for name, cols in _COLS.items():
            path = os.path.join(log_dir, f"{name}.csv")
            fh   = open(path, "w", newline="", buffering=1)   # line-buffered
            w    = csv.writer(fh)
            w.writerow(cols)
            self._files[name] = (fh, w)

        print(f"[DataLogger] Writing CSVs to {log_dir}")

    # ───────────────────────────────────────── convenience writer shorthand

    def _w(self, name: str):
        return self._files[name][1]

    # ─────────────────────────────────────────────────── public log methods

    def log_gyro(self, timestamp: float, gyro):
        """
        Call in _gyro_callback, immediately after extracting wx/wy/wz.

            self._logger.log_gyro(t, gyro)
        """
        self._w("imu_gyro").writerow([
            f"{timestamp:.9f}",
            f"{gyro[0]:.8f}", f"{gyro[1]:.8f}", f"{gyro[2]:.8f}",
        ])

    def log_accel(self, timestamp: float, accel):
        """
        Call in _accel_callback, immediately after extracting ax/ay/az.

            self._logger.log_accel(t, accel)
        """
        self._w("imu_accel").writerow([
            f"{timestamp:.9f}",
            f"{accel[0]:.8f}", f"{accel[1]:.8f}", f"{accel[2]:.8f}",
        ])

    def log_depth_frame(self, timestamp: float, depth_image):
        """
        Call in _depth_callback after storing self._latest_depth,
        passing the raw uint16 depth numpy array.

            self._logger.log_depth_frame(t, self._latest_depth)

        Logs statistics only — not the full image — so the file stays small.
        """
        import numpy as np

        valid   = depth_image[depth_image > 0]
        total   = depth_image.size
        n_valid = len(valid)
        n_inv   = total - n_valid
        ratio   = n_valid / total if total > 0 else 0.0

        scale = 0.001       # D455 publishes uint16 mm; convert to metres
        if n_valid > 0:
            d_min  = float(valid.min())  * scale
            d_max  = float(valid.max())  * scale
            d_mean = float(valid.mean()) * scale
            d_std  = float(valid.std())  * scale
        else:
            d_min = d_max = d_mean = d_std = 0.0

        self._w("depth_stats").writerow([
            self._frame_id, f"{timestamp:.9f}",
            n_valid, n_inv, f"{ratio:.6f}",
            f"{d_min:.4f}", f"{d_max:.4f}", f"{d_mean:.4f}", f"{d_std:.4f}",
        ])

    def log_landmarks(self, timestamp: float, landmark_manager):
        """
        Call in _rgb_callback AFTER update_depth() has run, so that
        depth / position_camera fields are already populated.

            self._logger.log_landmarks(timestamp, self._lm_mgr)
        """
        w = self._w("landmarks")
        for lm in landmark_manager.landmarks.values():
            if lm.position_camera is not None:
                cx, cy, cz = (f"{lm.position_camera[0]:.6f}",
                               f"{lm.position_camera[1]:.6f}",
                               f"{lm.position_camera[2]:.6f}")
            else:
                cx = cy = cz = ""

            # print(lm.id, lm.depth, lm.valid_depth, lm.position_camera)
            
            w.writerow([
                self._frame_id, f"{timestamp:.9f}",
                lm.id,
                f"{lm.pixel[0]:.3f}", f"{lm.pixel[1]:.3f}",
                f"{lm.depth:.4f}",
                cx, cy, cz,
                lm.track_age,
                lm.num_observations,
                f"{lm.descriptor_quality:.4f}",
                int(lm.valid_depth),
                int(lm.tracked),
                int(lm.active),
            ])

    def log_frame_summary(self, timestamp: float, landmark_manager,
                          gyro_count: int, accel_count: int,
                          redetected: bool, fps: float):
        """
        Call at the END of _rgb_callback, after everything else.
        Also increments the internal frame counter and flushes all files.

            self._logger.log_frame_summary(
                timestamp, self._lm_mgr,
                len(gyro_samples), len(accel_samples),
                redetected, fps
            )
        """
        self._w("frame_summary").writerow([
            self._frame_id, f"{timestamp:.9f}",
            len(landmark_manager.landmarks),
            landmark_manager.active_count(),
            landmark_manager.tracked_count(),
            gyro_count,
            accel_count,
            int(redetected),
            f"{fps:.3f}",
        ])

        self._frame_id += 1
        self._flush_all()   # flush once per frame — cheap at 30-60 Hz

    def close(self):
        self._flush_all()
        for fh, _ in self._files.values():
            fh.close()
        print(f"[DataLogger] All files closed. Logs in: {self._log_dir}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ────────────────────────────────────────────────────────────── private

    def _flush_all(self):
        for fh, _ in self._files.values():
            fh.flush()
