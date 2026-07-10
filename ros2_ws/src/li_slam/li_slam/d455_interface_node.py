#!/usr/bin/env python3
"""
d455_interface_node.py
────────────────────────
ROS2 frontend node. Owns ONLY:
  - subscriptions / message conversion
  - the KLT tracking loop
  - calling into LandmarkManager (landmark_manager.py) for identity/depth
  - calling into IMUBuffer (imu_buffer.py) for IMU windowing
  - visualisation
  - assembling FrameMeasurement (Phase 6) and handing it off

No SLAM math lives here. No observer math lives here (that's
observer/observer_state.py, observer/observer_propagation.py, etc. —
Phase 7+, intentionally a separate package so the frontend never imports
anything observer-related and vice versa).
"""

import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, Imu, CameraInfo
from cv_bridge import CvBridge

from li_slam.landmark_manager import LandmarkManager, CameraIntrinsics
from li_slam.imu_buffer import IMUBuffer
from li_slam.frame_measurement import FrameMeasurement, LandmarkMeasurement
# from li_slam.pandas_logger import PandasLogger
from li_slam.mydata_logger import DataLogger


# ──────────────────────────────────────────────────────────────────────────────
# ROS2 Node
# ──────────────────────────────────────────────────────────────────────────────

class D455Interface(Node):
    """
    Per-frame pipeline:
      1. KLT-propagate all currently-tracked landmarks (cheap, every frame).
      2. If tracked count dips below threshold, masked ORB redetect +
         LandmarkManager re-identification/creation (rare, bounded cost).
      3. Stamp every tracked landmark with depth + 3D position
         (LandmarkManager.update_depth).
      4. Extract the IMU samples that arrived since the last frame
         (IMUBuffer.extract_window).
      5. Assemble a FrameMeasurement and hand it to the observer-facing
         callback (currently just logged; Phase 7+ will consume it).
      6. Visualise.
    """

    MIN_TRACKED_FEATURES = 400      # redetect when active landmarks fall below this
    EXCLUSION_RADIUS      = 10      # px — mask radius around tracked points

    def __init__(self):
        super().__init__('d455_interface')

        self._bridge  = CvBridge()
        self._orb     = cv2.ORB_create(nfeatures=500)
        self._lm_mgr  = LandmarkManager()
        self._imu_buf = IMUBuffer()

        # Optical flow state
        self._prev_gray:   np.ndarray | None = None
        self._prev_points: np.ndarray | None = None   # (N,1,2) float32
        self._prev_ids:    list = []
        self._prev_image_time: float | None = None

        # Latest depth frame, kept but not processed until needed
        # (Phase 2.1 — store, don't block on it)
        self._latest_depth: np.ndarray | None = None

        # Camera intrinsics — populated from CameraInfo on first message.
        # Do NOT hardcode these; they vary by device/resolution.
        self._intrinsics: CameraIntrinsics | None = None

        self._prev_time = time.time()   # for FPS display only
        
        # # 🔴 ADD THIS: Initialize pandas logger
        # self._logger = PandasLogger(log_dir="d455_logs")
        # self._frame_counter = 0
        self._my_logger = DataLogger(log_dir="d455_logs")
        
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(Image, "/camera/camera/color/image_raw",
                                 self._rgb_callback, 10)
        self.create_subscription(Image,
                                 "/camera/camera/aligned_depth_to_color/image_raw",
                                 self._depth_callback, 10)
        self.create_subscription(CameraInfo,
                                 "/camera/camera/color/camera_info",
                                 self._camera_info_callback, 10)
        self.create_subscription(Imu, "/camera/camera/gyro/sample",
                                 self._gyro_callback, sensor_qos)
        self.create_subscription(Imu, "/camera/camera/accel/sample",
                                 self._accel_callback, sensor_qos)

        # print("self =", self)
        # print("self.get_logger =", self.get_logger)
        # print("type(self.get_logger()) =", type(self.get_logger()))
        # self.get_logger().info("D455 Interface started")
        self.get_logger().info("D455 Interface started with logging")
        

    # ══════════════════════════════════════════════════════════════════ private:
    #                              tracking pipeline

    def _bootstrap(self, gray: np.ndarray):
        """First frame: detect once, register every keypoint as a new landmark."""
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        if descriptors is None or len(descriptors) == 0:
            return

        points, ids = [], []
        for kp, des in zip(keypoints, descriptors):
            lm_id = self._lm_mgr.create(des, kp.pt)
            points.append(kp.pt)
            ids.append(lm_id)

        self._prev_gray   = gray.copy()
        self._prev_points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        self._prev_ids    = ids

    def _propagate(self, gray: np.ndarray):
        """Run KLT on all currently-tracked points; update LandmarkManager."""
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_points, None)

        good_new, good_old = [], []
        survived_points, survived_ids = [], []

        for i, st in enumerate(status.flatten()):
            lm_id = self._prev_ids[i]
            if st == 0:
                self._lm_mgr.mark_lost(lm_id)
                continue

            new_px = tuple(next_pts[i].ravel())
            self._lm_mgr.mark_tracked(lm_id, new_px)

            good_new.append(next_pts[i])
            good_old.append(self._prev_points[i])
            survived_points.append(next_pts[i])
            survived_ids.append(lm_id)

        good_new = np.array(good_new, dtype=np.float32)
        good_old = np.array(good_old, dtype=np.float32)
        survived_points = (np.array(survived_points, dtype=np.float32).reshape(-1, 1, 2)
                          if survived_points else np.empty((0, 1, 2), dtype=np.float32))

        return good_new, good_old, survived_points, survived_ids

    def _build_exclusion_mask(self, gray: np.ndarray, occupied_pixels: list) -> np.ndarray:
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        for x, y in occupied_pixels:
            cv2.circle(mask, (int(x), int(y)), self.EXCLUSION_RADIUS, 0, -1)
        return mask

    def _redetect_and_integrate(self, gray: np.ndarray, occupied_pixels: list):
        mask = self._build_exclusion_mask(gray, occupied_pixels)
        keypoints, descriptors = self._orb.detectAndCompute(gray, mask)
        if descriptors is None or len(descriptors) == 0:
            return [], []

        results = self._lm_mgr.integrate_new_detections(keypoints, descriptors)
        points = [pixel for (_id, pixel, _is_new) in results]
        ids    = [_id for (_id, _pixel, _is_new) in results]
        return points, ids

    # ══════════════════════════════════════════════════════════════════ private:
    #                              ROS2 callbacks

    def _rgb_callback(self, msg: Image):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._bootstrap(gray)
            self._prev_image_time = timestamp
            return

        good_new, good_old, survived_points, survived_ids = self._propagate(gray)

        redetected = False
        if len(survived_ids) < self.MIN_TRACKED_FEATURES:
            redetected = True
            occupied = [tuple(pt.ravel()) for pt in survived_points]
            new_points, new_ids = self._redetect_and_integrate(gray, occupied)

            if new_points:
                new_points_arr = np.array(new_points, dtype=np.float32).reshape(-1, 1, 2)
                survived_points = (np.concatenate([survived_points, new_points_arr], axis=0)
                                   if survived_points.shape[0] > 0 else new_points_arr)
                survived_ids = survived_ids + new_ids

        self._prev_gray   = gray.copy()
        self._prev_points = survived_points
        self._prev_ids    = survived_ids

        # Phase 2: stamp depth + 3D position on every currently tracked landmark
        if self._latest_depth is not None and self._intrinsics is not None:
            self._lm_mgr.update_depth(self._latest_depth, self._intrinsics)
        
        # print("Depth frame exists:", self._latest_depth is not None)
        # print("Intrinsics exist:", self._intrinsics is not None)

        self._lm_mgr.prune_stale()

        # Phase 4: pull IMU samples that arrived since the previous image
        gyro_samples, accel_samples = self._imu_buf.extract_window(
            self._prev_image_time, timestamp)
        self._prev_image_time = timestamp

        # Phase 6: assemble the frame measurement and hand it off
        frame_measurement = self._build_frame_measurement(
            timestamp, gyro_samples, accel_samples)
        self._on_frame_measurement(frame_measurement)

        # Logging
        now = time.time()
        fps = 1.0 / max(now - self._prev_time, 1e-6)
        self._my_logger.log_landmarks(timestamp, self._lm_mgr)
        self._my_logger.log_frame_summary(
            timestamp,
            self._lm_mgr,
            len(gyro_samples),
            len(accel_samples),
            redetected,
            fps
        )
        
        self._visualize(frame, good_new, good_old, redetected)
        

    def _depth_callback(self, msg: Image):
        # Phase 2.1: store only, no processing here. Processing happens
        # lazily in update_depth(), driven by the RGB frame's landmark set.
        self._latest_depth = self._bridge.imgmsg_to_cv2(
            msg, desired_encoding='passthrough')

        depth_vis = cv2.convertScaleAbs(self._latest_depth, alpha=0.03)
        cv2.imshow("Depth", depth_vis)
        cv2.waitKey(1)
        
        # print(f"Depth callback: shape={self._latest_depth.shape}")
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._my_logger.log_depth_frame(timestamp, self._latest_depth)
        

    def _camera_info_callback(self, msg: CameraInfo):
        if self._intrinsics is not None:
            return   # only need to read this once; intrinsics don't change
        fx, fy = msg.k[0], msg.k[4]
        cx, cy = msg.k[2], msg.k[5]
        self._intrinsics = CameraIntrinsics(fx, fy, cx, cy, depth_scale=0.001)
        self.get_logger().info(
            f"Camera intrinsics set: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
        

    def _gyro_callback(self, msg: Imu):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        gyro = np.array([msg.angular_velocity.x,
                         msg.angular_velocity.y,
                         msg.angular_velocity.z], dtype=np.float64)
        self._imu_buf.add_gyro(t, gyro)
        # print(f"Gyro callback: t={t:.6f}, gyro={gyro}")
        self._my_logger.log_gyro(t, gyro)
        

    def _accel_callback(self, msg: Imu):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        accel = np.array([msg.linear_acceleration.x,
                          msg.linear_acceleration.y,
                          msg.linear_acceleration.z], dtype=np.float64)
        self._imu_buf.add_accel(t, accel)
        # print(f"Accel callback: t={t:.6f}, accel={accel}")
        self._my_logger.log_accel(t, accel)

    # ══════════════════════════════════════════════════════════════════ private:
    #                       Phase 6 — FrameMeasurement assembly

    def _build_frame_measurement(self, timestamp: float,
                                  gyro_samples: list, accel_samples: list
                                 ) -> FrameMeasurement:
        active_landmarks = self._lm_mgr.get_active_landmarks()

        landmark_measurements = [
            LandmarkMeasurement(
                id=lm.id,
                pixel=lm.pixel,
                descriptor=lm.descriptor,
                depth=lm.depth,
                position_camera=lm.position_camera,
                track_age=lm.track_age,
                confidence=lm.descriptor_quality,
            )
            for lm in active_landmarks
        ]

        return FrameMeasurement(
            timestamp=timestamp,
            landmarks=landmark_measurements,
            gyro_samples=[(s.timestamp, s.value) for s in gyro_samples],
            accel_samples=[(s.timestamp, s.value) for s in accel_samples],
            camera_intrinsics=self._intrinsics,
        )

    def _on_frame_measurement(self, fm: FrameMeasurement):
        """
        Hand-off point to the observer backend. Currently a no-op /
        logging stub — Phase 7+ will replace this with a call into
        observer/observer_state.py + observer_propagation.py +
        observer_correction.py.

        Deliberately kept as a single method so wiring in the real
        observer later is a one-line change here, not a refactor of
        the frontend.
        """
        # # 🔴 ADD THIS: Log frame to pandas DataFrames
        # self._frame_counter += 1
        # self._logger.log_frame(self._frame_counter, fm, self._observer_state)
        
        # # Print occasional summary to console
        # if self._frame_counter % 10 == 0:
        #     self.get_logger().info(
        #         f"Frame {self._frame_counter}: {len(fm.landmarks)} landmarks, "
        #         f"pos=({self._observer_state.x[0]:.2f}, "
        #         f"{self._observer_state.x[1]:.2f}, "
        #         f"{self._observer_state.x[2]:.2f})"
        #     )
        pass  # TODO(Phase 7): self._observer.process(fm)

    # ══════════════════════════════════════════════════════════════════ private:
    #                              visualisation

    def _visualize(self, frame: np.ndarray,
                   good_new: np.ndarray, good_old: np.ndarray,
                   redetected: bool):
        vis = frame.copy()

        if good_new is not None and len(good_new) > 0:
            for new, old in zip(good_new, good_old):
                x_new, y_new = new.ravel()
                x_old, y_old = old.ravel()
                cv2.line(vis, (int(x_old), int(y_old)), (int(x_new), int(y_new)),
                         (0, 255, 0), 1)
                cv2.circle(vis, (int(x_new), int(y_new)), 3, (0, 0, 255), -1)

        now = time.time()
        fps = 1.0 / max(now - self._prev_time, 1e-6)
        self._prev_time = now

        label = "REDETECTING" if redetected else f"Tracked: {len(good_new)}"
        color = (0, 0, 255) if redetected else (0, 255, 0)
        cv2.putText(vis, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(vis, f"FPS: {fps:.1f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(vis,
                    f"Landmarks total: {len(self._lm_mgr.landmarks)}  "
                    f"active: {self._lm_mgr.active_count()}",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

        cv2.imshow("ORB Features", vis)
        cv2.waitKey(1)
    
    # 🔴 ADD THIS: Clean shutdown to save CSV files
    def destroy_node(self):
        """Save all logged data before shutting down."""
        if hasattr(self, '_my_logger'):
            self._my_logger.close()
        super().destroy_node()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = D455Interface()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    # try:
    #     rclpy.spin(node)
    # except KeyboardInterrupt:
    #     node.get_logger().info("Shutting down...")
    # finally:
    #     node.destroy_node()
    #     rclpy.shutdown()


if __name__ == '__main__':
    main()
