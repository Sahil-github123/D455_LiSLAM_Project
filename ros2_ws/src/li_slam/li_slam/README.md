# Observer Backend (Phase 7+)

This package is intentionally **empty of OpenCV/ROS2 dependencies**. It
implements Section 4.2 of the LI-SLAM paper as pure mathematics, operating
only on `FrameMeasurement` objects (see `../frame_measurement.py`).

## Planned files

- `observer_state.py`
  `ObserverState` — holds `R̂` (SO(3)), `v̂`, `x̂`, `{p̂ᵢ}` (landmark
  position estimates), and bias estimates if/when added. Maps directly
  to the `X̂ = (R̂, V̂) ∈ SEₙ₊₂(3)` representation in Section 3.3/4.1.

- `observer_propagation.py`
  Implements the dynamics `Ẋ = XU + GX + [N,X]` (eq. 11/14) using only
  gyro/accel — the IMU-only "predict" step, no landmarks involved. Pure
  function of `(ObserverState, gyro, accel, dt) -> ObserverState`.

- `observer_correction.py`
  Implements the correction terms `Δ`, `Γ` (eqs. 20-22, Theorem 1) using
  landmark measurements `yᵢ = R̂ᵀ(p̂ᵢ - x̂)` vs. the actual measured
  `yᵢ` from `FrameMeasurement.landmarks[i].position_camera`. This is the
  "update" step.

- `lie_algebra.py`
  Small shared utilities: `so3_hat`/`so3_vee`, SEₙ₊₂(3) composition,
  the `Π`, `C`, `SN` constant matrices from Section 3.3, etc. — kept
  separate so propagation/correction stay readable.

## Why this is a separate package from the frontend

`d455_interface_node.py` should never need to know `kR`, `kv`, `kx`, `kp`
exist. The observer should never need to know what an `Image` ROS message
or a `cv2.KeyPoint` is. The only thing crossing the boundary is
`FrameMeasurement` (`../frame_measurement.py`) flowing one direction, and
eventually `ObserverState` (pose/velocity/landmark estimates) flowing back
out for whatever consumes them (RViz visualization, logging, downstream
planning, etc.).

This isn't implemented yet — `D455Interface._on_frame_measurement()` is
currently a no-op stub with a `# TODO(Phase 7)` marker showing exactly
where the wiring goes.
