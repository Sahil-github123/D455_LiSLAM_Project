#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu

from cv_bridge import CvBridge
import cv2
import time
import numpy as np

from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import HistoryPolicy

# TODO: Later, we should do using this
# @dataclass
# class TrackingResult:
    # good_new: np.ndarray
    # good_old: np.ndarray
    # redetected: bool


class Landmark:
    def __init__(self, landmark_id, descriptor, pixel):

        self.id = landmark_id
        self.descriptor = descriptor
        self.pixel = pixel
        self.observations = 1
        self.age = 0        # Start with zero / one ?
        self.last_seen = time.time()


# Because in the paper you're implementing,
# the observer estimates landmarks (persistent map entities),
# while optical flow produces tracks (temporary image measurements).
# Mixing those concepts into one dictionary quickly becomes messy.
class FeatureTrack:
    pass


class LandmarkManager:
    pass


class D455Interface(Node):

    def __init__(self):

        self.prev_time = time.time()

        super().__init__('d455_interface')

        self.bridge = CvBridge()

        self.rgb_count = 0
        self.depth_count = 0

        self.min_tracking_features = 200        # When tracking becomes weak we re-run ORB (Tracked Features < Threshold)

        # To solve the QoS warning
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10 )

        # ORB feature detector
        self.orb = cv2.ORB_create( nfeatures=1000 )

        # Optical Flow Tracking Variables
        self.prev_gray = None
        self.prev_points = None
        
        # Landmarks storage
        self.landmarks = {}
        self.next_landmark_id = 0
        self.bf = cv2.BFMatcher( cv2.NORM_HAMMING, crossCheck=True )

        #--------------------------- Subscriptions ---------------------------#
        self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.rgb_callback,
            10
        )
        self.create_subscription(
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",
            self.depth_callback,
            10
        )
        self.create_subscription(
            Imu,
            "/camera/camera/gyro/sample",
            self.gyro_callback,
            sensor_qos          # Earlier kept 50
        )
        self.create_subscription(
            Imu,
            "/camera/camera/accel/sample",
            self.accel_callback,
            sensor_qos          # Earlier kept 50
        )

        self.get_logger().info("D455 Interface Started")


    def detect_orb_features(self, gray):
        keypoints, descriptors = \
            self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors


    def track_features(self, gray):
        # Optical Flow Tracking
        # First frame
        initialize = False
        if self.prev_gray is None:
            keypoints = self.orb.detect(gray, None)
            
            points = np.array(
                [kp.pt for kp in keypoints],
                dtype=np.float32
            ).reshape(-1,1,2)

            self.prev_gray = gray.copy()
            self.prev_points = points
            initialize = True
            return None, None, False, initialize

        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray,
            self.prev_points, None
        )

        # Keep Only Good Tracks
        good_new = next_points[status.flatten() == 1]
        good_old = self.prev_points[status.flatten() == 1]
        # Meaning: status = 1 = Feature successfully tracked  ,  status = 0 = Feature lost

        # Re-Detection (Because lost features never come back)
        # if len(good_new) < 200:
        redetect = False
        tracked_count = len(good_new)
        if tracked_count < self.min_tracking_features:
            keypoints = self.orb.detect(gray, None)

            new_points = np.array(
                [kp.pt for kp in keypoints],
                dtype = np.float32
            ).reshape(-1,1,2)

            self.prev_points = new_points
            self.prev_gray = gray.copy()
            redetect = True
            
        return good_new, good_old, redetect, initialize


    def visualize_features(self, frame, good_new, good_old, redetect):
        # Draw Tracks/Motion Vectors
        frame_features = frame.copy()

        for new, old in zip(good_new, good_old):
            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()

            cv2.line(frame_features,
                     (int(x_old), int(y_old)), (int(x_new), int(y_new)),
                     (0,255,0), 2
            )
            cv2.circle(frame_features, (int(x_new), int(y_new)),
                        3,  (0,0,255),  -1
            )

        # Redetecting Message
        if redetect:
            cv2.putText(frame_features, "REDETECTING", (20,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2
            )
            cv2.imshow("ORB Features", frame_features)
            cv2.waitKey(1)
            return   # Return cause mixing them in the same frame gets messy, so: Detect, Store, Wait for next frame, Track again.

        # Display
        current = time.time()
        fps = 1.0/(current-self.prev_time)
        self.prev_time = current

        cv2.putText(
            frame_features,
            # f"Features: {len(keypoints)}",
            f"Tracked: {len(good_new)}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )
        cv2.putText(
            frame_features,
            f"FPS: {fps:.1f}",
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,0,0),
            2
        )
        cv2.imshow("ORB Features", frame_features)
        # cv2.imshow("RGB",frame)
        cv2.waitKey(1)


    def update_landmarks(self, descriptors, keypoints):
        if descriptors is None:
            return

        for kp, des in zip(keypoints, descriptors):
            landmark = Landmark(self.next_landmark_id, des, kp.pt)

            self.landmarks[self.next_landmark_id] = landmark
            
            self.next_landmark_id += 1


    def rgb_callback(self, msg):

        self.rgb_count += 1

        # Receive the RGB image and convert it to OpenCV format
        frame = self.bridge.imgmsg_to_cv2( msg, desired_encoding = 'bgr8' )

        # Convert to Grayscale for Feature Detection
        gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )

        # Detect ORB Features
        keypoints, descriptors = self.detect_orb_features(gray)

        good_new, good_old, redetect, initialize = self.track_features(gray)

        if initialize:      return

        self.visualize_features(frame, good_new, good_old, redetect)

        if redetect:        return
        
        self.update_landmarks(descriptors, keypoints)
        print(f"Landmarks Count: {len(self.landmarks)}")

        # Update For Next Frame     # Store previous frame
        self.prev_gray = gray.copy()
        self.prev_points = good_new.reshape(-1, 1, 2)

        ''' # Phase 2 (Old)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is not None:
            print(descriptors.shape)
            # This confirms 32-bit ORB descriptors are available
        frame_features = cv2.drawKeypoints(frame, keypoints, None)
        '''


    def depth_callback(self,msg):
        self.depth_count += 1
        depth = self.bridge.imgmsg_to_cv2( msg, desired_encoding = 'passthrough' )
        depth_vis = cv2.convertScaleAbs(depth, alpha = 0.03 )
        cv2.imshow("Depth", depth_vis)
        cv2.waitKey(1)


    def gyro_callback(self,msg):
        # print("GYRO RECEIVED")
        pass


    def accel_callback(self,msg):
        # print("ACCEL RECEIVED")
        pass


def main():

    rclpy.init()

    node = D455Interface()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
    