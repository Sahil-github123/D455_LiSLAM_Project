import numpy as np
from .observer_state import ObserverState
from .propagator import Propagator
from .math_utils import skew, so3_exp, project_to_so3


class Observer:
    """
    Section 4.2 / Theorem 1 observer.

    process() runs, once per image frame:
      1. IMU prediction  (propagator.py — Eq. 2 dynamics, unchanged)
      2. New-landmark initialization
      3. Landmark residuals  y_i - yhat_i
      4. Measurement correction (Theorem 1): feeds the residuals back
         into R, v, x and every mapped landmark via the correction
         terms Omega_delta, W_delta.

    Step 4 is the part that was missing — without it this class was a
    pure IMU integrator with no feedback, and would drift without bound.
    """

    def __init__(self, kR: float = 2.0, kv: float = 2.0,
                 kx: float = 1.0, kp: float = 4.0):
        """
        Gains as used in the paper's own simulation (Section 5):
        kv=2.0, kx=1.0, kp=4.0, kR=2.0. Required: kR>0, kv>0, kp>0,
        kp + n*kx > 0 (n = number of landmarks used in the correction
        this frame).
        """
        self.state = ObserverState()
        self.propagator = Propagator()

        self.kR = kR
        self.kv = kv
        self.kx = kx
        self.kp = kp

        self._prev_frame_t = None   # for dt between frames
        self._dt_history = []       # recent frame dt's, to flag outliers

        # Rolling diagnostics history — see observer_diagnostics.py for
        # plotting. Kept as plain lists of python floats/small arrays so
        # this stays cheap to append to at frame rate; call
        # save_diagnostics() periodically on long-running sessions to
        # flush to disk instead of holding everything in memory.
        self.diag = {
            't':               [],
            'dt':              [],
            'num_landmarks':   [],
            'map_size':        [],
            'residual_mean':   [],
            'residual_median': [],
            'residual_max':    [],
            'correction_applied': [],
            'num_substeps':    [],
            'omega_delta_norm': [],
            'dR_angle':        [],   # rotation change this frame, radians
            'dv':              [],   # |v_new - v_old|
            'dx':              [],   # |x_new - x_old|
            'dP_mean':         [],   # mean |p_i_new - p_i_old| over corrected landmarks
            'x':               [],   # (3,) estimated position, for trajectory plot
            'v_norm':          [],
        }

    def process(self, frame):
        # dt since the previous frame — needed to integrate the
        # correction terms, which (unlike propagate()) only update once
        # per frame rather than once per IMU sample.
        dt = None
        if self._prev_frame_t is not None:
            dt = frame.timestamp - self._prev_frame_t
        self._prev_frame_t = frame.timestamp

        # Flag anomalously large frame gaps (dropped/stalled frames).
        # These are a common root cause of the "whole-pose glitches for
        # one frame" pattern — a bad/huge dt feeding propagate() or the
        # correction step — so make them visible instead of silent.
        if dt is not None and dt > 0.0:
            self._dt_history.append(dt)
            if len(self._dt_history) > 5:
                recent_median = np.median(self._dt_history[-30:])
                if dt > 3.0 * recent_median:
                    print(f"[Observer] WARNING: frame dt={dt:.4f}s is "
                          f">3x the recent median ({recent_median:.4f}s) "
                          f"— likely a dropped/stalled frame at "
                          f"t={frame.timestamp:.3f}")

        # 1. IMU prediction
        self.propagator.propagate(self.state, frame)

        # 2. Initialize unseen landmarks - ONLY after propagator is initialized
        residuals = []
        if self.propagator.initialized:
            self._initialize_new_landmarks(frame)
            # 3. Compute residuals for known landmarks
            residuals = self._compute_landmark_residuals(frame)

        residual_mean = residual_median = residual_max = float('nan')
        if residuals:
            residual_vectors = np.array([r[3] for r in residuals])
            residual_norms = np.linalg.norm(residual_vectors, axis=1)
            residual_mean = float(np.mean(residual_norms))
            residual_median = float(np.median(residual_norms))
            residual_max = float(np.max(residual_norms))

        # 4. Measurement correction (Theorem 1 / Section 4.2). Skipped
        # until the propagator has an initial orientation/bias estimate,
        # until we have at least one landmark residual to correct with,
        # and on the very first frame (no dt yet to integrate over).
        correction_diag = None
        if (self.propagator.initialized and residuals
                and dt is not None and dt > 0.0):
            correction_diag = self._apply_correction(residuals, dt)

        # --- record diagnostics for this frame ---
        d = self.diag
        d['t'].append(frame.timestamp)
        d['dt'].append(dt if dt is not None else float('nan'))
        d['num_landmarks'].append(len(residuals))
        d['map_size'].append(len(self.state.landmarks))
        d['residual_mean'].append(residual_mean)
        d['residual_median'].append(residual_median)
        d['residual_max'].append(residual_max)
        d['correction_applied'].append(correction_diag is not None)
        d['num_substeps'].append(correction_diag['num_substeps'] if correction_diag else 0)
        d['omega_delta_norm'].append(correction_diag['omega_delta_norm'] if correction_diag else float('nan'))
        d['dR_angle'].append(correction_diag['dR_angle'] if correction_diag else float('nan'))
        d['dv'].append(correction_diag['dv'] if correction_diag else float('nan'))
        d['dx'].append(correction_diag['dx'] if correction_diag else float('nan'))
        d['dP_mean'].append(correction_diag['dP_mean'] if correction_diag else float('nan'))
        d['x'].append(self.state.x.copy())
        d['v_norm'].append(float(np.linalg.norm(self.state.v)))

        print(f"Map size: {len(self.state.landmarks)} | "
              f"Landmarks: {len(residuals)} | "
              f"mean residual: {residual_mean:.5f} m | "
              f"median: {residual_median:.5f} m | "
              f"max: {residual_max:.5f} m")

        return self.state

    def save_diagnostics(self, path: str):
        """
        Dump the diagnostics history to an .npz file for offline
        plotting (see observer_diagnostics.py: plot_diagnostics_from_file).
        Call this periodically on long-running sessions (e.g. every N
        frames, or at shutdown) rather than keeping everything in RAM
        forever.
        """
        arrays = {k: np.array(v) for k, v in self.diag.items()}
        np.savez(path, **arrays)

    def _initialize_new_landmarks(self, frame):
        R = self.state.R
        x = self.state.x

        for landmark in frame.landmarks:
            # Skip invalid measurements
            if (landmark.position_camera is None or landmark.depth <= 0.0 ):
                continue

            landmark_id = landmark.id

            # Already initialized
            if landmark_id in self.state.landmarks:
                continue

            p_camera = np.asarray(landmark.position_camera, dtype=np.float64)

            # Assuming camera frame == IMU/body frame
            p_world = (x + R @ p_camera)

            self.state.landmarks[landmark_id] = p_world.copy()      # persistent world-frame landmark positions
    
    def _compute_landmark_residuals(self, frame):

        residuals = []
        R = self.state.R
        x = self.state.x

        for landmark in frame.landmarks:
            # Must have a valid 3D measurement
            if (landmark.position_camera is None or landmark.depth <= 0.0 ):
                continue

            landmark_id = landmark.id

            # Only use landmarks already in the observer map
            if landmark_id not in self.state.landmarks:
                continue

            p_world = self.state.landmarks[landmark_id]

            p_camera_measured = np.asarray(landmark.position_camera, dtype=np.float64)

            # Predict where this world landmark should appear
            # in the current camera frame
            p_camera_predicted = (R.T @ (p_world - x))

            # residual = y_i - yhat_i  (Section 4.2 notation)
            residual = p_camera_measured - p_camera_predicted

            residuals.append(
                (
                    landmark_id,
                    p_camera_measured,
                    p_camera_predicted,
                    residual
                )
            )

        return residuals

    # Sub-stepping safety margin for the correction integration (see
    # docstring below). Keep dt_sub * |dominant_pole| well under the
    # explicit-Euler stability limit of 2.
    _CORRECTION_STABILITY_SAFETY = 0.2
    _CORRECTION_MAX_SUBSTEPS = 400

    def _apply_correction(self, residuals, dt):
        """
        Theorem 1 correction terms:

            Omega_delta = kR * (e3x @ sum_i R(y_i - yhat_i))
            R  <- exp(Omega_delta * dt) @ R           (world-frame, left-mult)
            v  <- v + dt*(-kv * S + Omega_delta_x @ (v - vZ))
            x  <- x + dt*(-kx * S + Omega_delta_x @ (x - xZ))
            p_i<- p_i + dt*(kp * r_i + Omega_delta_x @ p_i)

        where S = sum_i r_i, r_i = R(y_i - yhat_i), and
            vZ = (kp + n*kx)/(n*kv) * g_vec
            xZ = 1/(n*kv) * g_vec
        with g_vec the same gravity vector used by the propagator
        (already sign-consistent with the paper's "g*e3" convention).

        Numerical note
        --------------
        Theorem 1 proves this is globally exponentially stable in
        *continuous time*. The (v, x, p_i) subsystem's linearization has
        poles at -kp and -(kp+n*kx)/2 +- sqrt(...)/2, whose magnitude
        grows with n (the number of landmarks in the sum this frame).
        With hundreds of landmarks at a normal camera frame rate, a
        single explicit-Euler step over dt is far outside the stability
        region (dt * |pole| >> 2) and will visibly diverge even though
        the underlying dynamics are stable. We sub-step dt into pieces
        small enough to stay inside that region, re-evaluating the
        residuals from the updated state each sub-step (measurements
        y_i are fixed for the frame; only R, x, p_i change).

        residuals: output of _compute_landmark_residuals — list of
                   (landmark_id, p_measured, p_predicted, residual).
        dt: time elapsed since the previous frame.
        """
        n = len(residuals)
        if n == 0:
            return None

        e3 = np.array([0.0, 0.0, 1.0])
        g_vec = self.propagator.gravity   # == g*e3 in the paper's sign convention

        lm_ids = [lm_id for lm_id, _, _, _ in residuals]
        Y = np.array([p_meas for _, p_meas, _, _ in residuals])   # (n,3), fixed y_i

        vZ = (self.kp + n * self.kx) / (n * self.kv) * g_vec
        xZ = (1.0 / (n * self.kv)) * g_vec

        dominant_pole = max(self.kp, self.kp + n * self.kx)
        num_substeps = int(np.ceil(dt * dominant_pole / self._CORRECTION_STABILITY_SAFETY))
        num_substeps = max(1, min(num_substeps, self._CORRECTION_MAX_SUBSTEPS))
        # num_substeps = max(1, num_substeps)  
        dt_sub = dt / num_substeps

        P = np.array([self.state.landmarks[i] for i in lm_ids])   # (n,3)

        # --- snapshot "before" state for diagnostics ---
        R_before = self.state.R.copy()
        v_before = self.state.v.copy()
        x_before = self.state.x.copy()
        P_before = P.copy()

        last_omega_delta_norm = 0.0

        for _ in range(num_substeps):
            R = self.state.R
            x = self.state.x

            # r_i = R(y_i - yhat_i) = R y_i - (p_i - x), stacked (n,3)
            r_all = Y @ R.T - (P - x)
            S = r_all.sum(axis=0)

            Omega_delta = self.kR * (skew(e3) @ S)
            Omega_delta_x = skew(Omega_delta)
            last_omega_delta_norm = float(np.linalg.norm(Omega_delta))

            self.state.R = so3_exp(Omega_delta * dt_sub) @ R
            self.state.R = project_to_so3(self.state.R)

            self.state.v = self.state.v + dt_sub * (
                -self.kv * S + Omega_delta_x @ (self.state.v - vZ)
            )
            self.state.x = self.state.x + dt_sub * (
                -self.kx * S + Omega_delta_x @ (self.state.x - xZ)
            )

            # p_i <- p_i + dt_sub * (kp * r_i + Omega_delta_x @ p_i), vectorized
            P = P + dt_sub * (self.kp * r_all + P @ Omega_delta_x.T)

        for idx, lm_id in enumerate(lm_ids):
            self.state.landmarks[lm_id] = P[idx]

        # --- diagnostics ---
        rotation_delta = R_before.T @ self.state.R
        cos_angle = np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0)
        dR_angle = float(np.arccos(cos_angle))
        dv = float(np.linalg.norm(self.state.v - v_before))
        dx = float(np.linalg.norm(self.state.x - x_before))
        dP_mean = float(np.mean(np.linalg.norm(P - P_before, axis=1)))

        return {
            'num_substeps': num_substeps,
            'omega_delta_norm': last_omega_delta_norm,
            'dR_angle': dR_angle,
            'dv': dv,
            'dx': dx,
            'dP_mean': dP_mean,
        }