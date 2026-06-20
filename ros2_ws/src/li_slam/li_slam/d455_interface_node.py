#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu

from cv_bridge import CvBridge

import cv2
import time
import numpy as np


class D455Interface(Node):

    def __init__(self):

        self.prev_time = time.time()

        super().__init__('d455_interface')

        self.bridge = CvBridge()

        self.rgb_count = 0
        self.depth_count = 0

        # ORB feature detector
        self.orb = cv2.ORB_create(
            nfeatures=1000
        )

        # Optical Flow Tracking Variables
        self.prev_gray = None
        self.prev_points = None

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
            50
        )

        self.create_subscription(
            Imu,
            "/camera/camera/accel/sample",
            self.accel_callback,
            50
        )

        self.get_logger().info("D455 Interface Started")


    def rgb_callback(self,msg):

        self.rgb_count += 1

        frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        ''' Phase 2
        keypoints, descriptors = self.orb.detectAndCompute(
            gray,
            None
        )
        if descriptors is not None:
            print(descriptors.shape)
            # This confirms 32-bit ORB descriptors are available

        frame_features = cv2.drawKeypoints(
            frame,
            keypoints,
            None
        )
        '''

        # Optical Flow Tracking
        # First frame
        if self.prev_gray is None:
            keypoints = self.orb.detect(
                gray,
                None
            )
            points = np.array(
                [kp.pt for kp in keypoints],
                dtype=np.float32
            ).reshape(-1,1,2)

            self.prev_gray = gray.copy()
            self.prev_points = points
            return

        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_points,
            None
        )

        # Keep Only Good Tracks
        good_new = next_points[status.flatten() == 1]
        good_old = self.prev_points[status.flatten() == 1]
        # Meaning: status = 1 --> Feature successfully tracked  ,  status = 0 --> Feature lost

        # Draw Tracks/Motion Vectors
        frame_features = frame.copy()

        for new, old in zip(good_new, good_old):
            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()

            cv2.line(
                frame_features,
                (int(x_old),int(y_old)),
                (int(x_new),int(y_new)),
                (0,255,0),
                2
            )

            cv2.circle(
                frame_features,
                (int(x_new),int(y_new)),
                3,
                (0,0,255),
                -1
            )
        
        # Re-Detection (Because lost features never come back)
        # if len(good_new) < 200:
            
        
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

        cv2.imshow(
            "ORB Features",
            frame_features
        )
        # cv2.imshow("RGB",frame)

        # Update For Next Frame
        self.prev_gray = gray.copy()
        self.prev_points = good_new.reshape(-1, 1, 2)

        cv2.waitKey(1)


    def depth_callback(self,msg):

        self.depth_count += 1

        depth = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='passthrough'
        )

        depth_vis = cv2.convertScaleAbs(
            depth,
            alpha=0.03
        )

        cv2.imshow(
            "Depth",
            depth_vis
        )

        cv2.waitKey(1)


    def gyro_callback(self,msg):
        pass


    def accel_callback(self,msg):
        pass


def main():

    rclpy.init()

    node = D455Interface()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
    
