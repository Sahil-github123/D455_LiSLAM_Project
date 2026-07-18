import numpy as np
from dataclasses import dataclass

from .math_utils import skew, project_to_so3, so3_exp, rotation_from_two_vectors
from .observer_state import ObserverState
from li_slam.frame_measurement import FrameMeasurement


@dataclass
class IMUEvent:
    timestamp: float
    gyro: np.ndarray
    accel: np.ndarray


class Propagator:
    '''Propagates the observer state forward in time using IMU measurements.'''

    def __init__(self):
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.accel_init_samples = []
        # A single RGB frame may contain only one or two accelerometer samples. Instead, initialize using a stationary window.
        self.initialized = False
        self.init_accel_samples = []
        self.init_gyro_samples = []
        
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3, dtype=np.float64)

    def propagate(self, state: ObserverState, frame: FrameMeasurement):
        '''Propagate the observer state forward in time using the IMU samples from the frame measurement.'''
        if not frame.accel_samples:
            return
        
        if not self.initialized:
            self.init_accel_samples.extend(frame.accel_samples)
            self.init_gyro_samples.extend(frame.gyro_samples)
            if len(self.init_accel_samples) < 100:
                return

            self.initialize_from_stationary(state, self.init_accel_samples, self.init_gyro_samples)
            self.initialized = True
            print("Observer orientation initialized from gravity")
            print("Observer initialized from stationary IMU samples")
            
            return   # Skip propagation during initialization
        
        # print(f'state.R: {state.R}')
        
        # # Verify stationary accelerometer magnitude
        # for _, accel in frame.accel_samples:
        #     self.accel_init_samples.append(accel.copy())
        #     if len(self.accel_init_samples) >= 100:
        #         mean_accel = np.mean(self.accel_init_samples, axis=0)
        #         print("Mean stationary accel:",    mean_accel,
        #               "norm:",        np.linalg.norm(mean_accel)   )
        #         self.accel_init_samples.clear()
        #     else:
        #         print(f"Collecting stationary accel samples: {len(self.accel_init_samples)}/100")
        
        # This means: if either sensor has no measurements for this frame interval, skip propagation rather than trying to integrate with fake zero measurements.
        if not frame.gyro_samples:
            return
        
        events = self._build_timeline(frame.gyro_samples, frame.accel_samples)
        if len(events) < 2:
            return
        
        # print(f"{len(events)} IMU events")
        # for e in events:
        #     print(e.timestamp, e.gyro, e.accel)
        
        for prev, curr in zip(events[:-1], events[1:]):

            dt = curr.timestamp - prev.timestamp
            if dt <= 0.0:
                continue

            # --- 1. Orientation propagation ---
            corrected_gyro = prev.gyro - self.gyro_bias
            dtheta = corrected_gyro * dt
            state.R = (state.R  @  so3_exp(dtheta) )
            state.R = project_to_so3(state.R)

            # --- 2. Acceleration in world frame ---
            corrected_accel = prev.accel - self.accel_bias
            a_world = (state.R @ corrected_accel + self.gravity )
            # a_world = (state.R @ prev.accel + self.gravity )

            # --- 3. Position propagation ---
            state.x += state.v * dt

            # --- 4. Velocity propagation ---
            state.v += a_world * dt

            # print(  "a_body:", prev.accel,
            #         "a_world:", a_world,
            #         "norm:", np.linalg.norm(a_world) )
            print(
                f"dt={dt:.5f} | "
                f"|a_world|={np.linalg.norm(a_world):.5f} | "
                f"|gyro|={np.linalg.norm(prev.gyro):.5f} | "    # This will let us distinguish accelerometer bias from gyro-induced attitude drift
                f"|v|={np.linalg.norm(state.v):.5f} | "
                f"|x|={np.linalg.norm(state.x):.5f}"
            )
    
    
    def _build_timeline(self, gyro_samples, accel_samples):
        '''Build a timeline of IMU events from gyro and accel samples.'''

        events = []
        # latest_gyro = np.zeros(3, dtype=np.float64)
        # latest_accel = np.zeros(3, dtype=np.float64)
        latest_gyro = None
        latest_accel = None
        gyro_idx = 0
        accel_idx = 0

        while gyro_idx < len(gyro_samples) or accel_idx < len(accel_samples):

            next_gyro = (gyro_samples[gyro_idx][0] 
                    if gyro_idx < len(gyro_samples)
                    else np.inf
            )
            next_accel = (accel_samples[accel_idx][0]
                    if accel_idx < len(accel_samples)
                    else np.inf
            )

            if next_gyro <= next_accel:
                timestamp, gyro = gyro_samples[gyro_idx]
                latest_gyro = gyro.copy()
                gyro_idx += 1

            else:
                timestamp, accel = accel_samples[accel_idx]
                latest_accel = accel.copy()
                accel_idx += 1
            
            # Only create an event once both sensors have valid values
            if latest_gyro is not None and latest_accel is not None:
                events.append(
                    IMUEvent(timestamp, latest_gyro.copy(), latest_accel.copy() )
                )

        return events
    
    def initialize_from_stationary(
        self,
        state: ObserverState,
        accel_samples, gyro_samples
    ):
        """
        Initialize body-to-world orientation using stationary
        accelerometer measurements.

        Assumes accelerometer measures specific force.
        """

        if not accel_samples:
            return False

        accel_values = np.array(
            [accel for _, accel in accel_samples],
            dtype=np.float64
        )
        gyro_values = np.array(
            [gyro for _, gyro in gyro_samples],
            dtype=np.float64
        )

        mean_accel = np.mean(accel_values, axis=0)

        # Specific force at rest is approximately -gravity
        gravity_body = -mean_accel

        gravity_world = np.array(
            [0.0, 0.0, -1.0],
            dtype=np.float64
        )

        state.R = rotation_from_two_vectors(
            gravity_body,
            gravity_world
        )

        state.R = project_to_so3(state.R)
        
        # Expected stationary accelerometer measurement
        expected_accel = ( state.R.T @ (-self.gravity) )
        self.accel_bias = mean_accel - expected_accel
        
        self.gyro_bias = np.mean(gyro_values, axis=0)
        
        print("Mean accel:", mean_accel)
        print("Expected accel:", expected_accel)
        print("Estimated accel bias:", self.accel_bias)
        print("Estimated gyro bias:", self.gyro_bias)

        return True
