# Landmark-Inertial SLAM with Intel RealSense D455

An implementation of **Landmark-Inertial Simultaneous Localization and Mapping (LiSLAM)** using an RGB-D camera and IMU, based on the observer proposed in:

> **Synchronous Observer Design for Landmark-Inertial SLAM with Almost-Global Convergence**

The project focuses on translating the nonlinear observer formulation from simulation/theory to a **real RGB-D + IMU platform**, with a practical visual frontend for continuous landmark tracking.

---

## System Pipeline

```text
                    Intel RealSense D455
                            │
                 ┌──────────┴──────────┐
                 │                     │
              RGB-D                   IMU
                 │                Accel + Gyro
                 ▼                     │
         ORB Feature Detection         │
                 │                     │
                 ▼                     │
          KLT Feature Tracking         │
                 │                     │
                 ▼                     │
       Depth-based 3D Landmarks        │
                 │                     │
                 └──────────┬──────────┘
                            ▼
                 Landmark-Inertial Observer
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
             State Estimate       Landmark Map
```

---

## Observer Formulation

The system estimates the IMU pose, velocity, and landmark positions using inertial propagation and landmark-based corrections.

The observer state is represented as

$$
\hat{x} =
(\hat{R},\hat{p},\hat{v},\hat{p}_1,\ldots,\hat{p}_N)
$$

where:

* \(R \in SO(3)\): attitude
* \(p \in \mathbb{R}^3\): position
* \(v \in \mathbb{R}^3\): velocity
* \(p_i \in \mathbb{R}^3\): \(i\)-th landmark position

The inertial propagation is driven by angular velocity and acceleration measurements:

$$
\dot{\hat{R}} = \hat{R}(\omega_m-\hat{b}_\omega)^\wedge
$$

$$
\dot{\hat{p}} = \hat{v}
$$

$$
\dot{\hat{v}} =
g+\hat{R}(a_m-\hat{b}_a)
$$

with landmark observations providing the correction term.

The visual measurements are expressed through the estimated relative landmark position:

$$
y_i = \hat{R}^{T}(p_i-p)
$$

and the resulting landmark residuals are used to correct the propagated state.

The observer gains \(k_R,\;k_v,\;k_p,\;k_x\) control the correction dynamics.

---

## What Was Implemented

The theoretical observer was integrated with a practical RGB-D/IMU frontend consisting of:

* ORB-based feature detection
* KLT optical-flow tracking between frames
* Automatic feature re-detection when tracking degrades
* Depth-based 3D landmark initialization
* Landmark management and feature-landmark association
* Real-time IMU propagation
* Nonlinear observer correction
* Stationary IMU initialization using gravity
* ROS 2 integration and experimental data logging

---

## Main Contribution / Improvements

The primary focus of this implementation is **bridging the gap between the theoretical LiSLAM observer and real sensor data**.

Compared with a purely theoretical/simulation implementation, the system incorporates practical mechanisms required for operation on a real RGB-D/IMU platform:

### 1. Continuous Visual Tracking

Instead of independently detecting landmarks in every frame, previously observed features are propagated using KLT optical flow. ORB detection is triggered when the number of tracked features falls below a threshold.

### 2. Direct RGB-D Landmark Initialization

The D455 provides depth for each visual feature, allowing landmarks to be initialized directly in 3D without monocular triangulation.

### 3. Landmark Management

A landmark manager maintains feature-to-landmark associations and distinguishes currently active observations from the persistent landmark map.

### 4. Real-World Sensor Integration

The observer was adapted to handle real IMU and RGB-D measurements, including timestamp-based propagation, sensor noise, feature tracking failures, and depth measurement errors.

---

## Results

The system was successfully demonstrated on the D455 with:

* Continuous visual feature tracking
* 3D landmark generation
* IMU-based state propagation
* Landmark-based observer correction
* Online map construction

A representative development run produced:

```text
Map size:          ~1250
Active landmarks: ~138
Mean residual:      0.191 m
Median residual:    0.083 m
Maximum residual:   1.838 m
```

These are development results and are not intended as formal benchmark values.

---

## Limitations & Future Work

Current limitations include landmark re-identification after features leave and re-enter the field of view, depth/IMU noise, feature tracking failures, and sensor synchronization.

Future work includes:

* Robust landmark re-identification
* Improved feature-landmark association
* IMU bias estimation
* Quantitative trajectory evaluation
* Ground-truth benchmarking
* Comparison with established VIO/SLAM systems
* Improved real-time performance

---

## Reference

**Synchronous Observer Design for Landmark-Inertial SLAM with Almost-Global Convergence**

The observer formulation and convergence properties are based on the above work; this repository focuses on its implementation and adaptation to a real RGB-D/IMU system.
