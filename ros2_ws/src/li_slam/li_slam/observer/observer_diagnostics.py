#!/usr/bin/env python3
"""
observer_diagnostics.py
────────────────────────
Plotting for Observer.diag — lets you visually sanity-check the
Section 4.2 observer without ground truth (which the paper's
simulation had but a real run doesn't).

Two figures:

  1. Error/convergence trends (analog of paper Fig. 2, minus the parts
     that need ground truth):
       - landmark residual norms (mean / median / max) over time
       - correction step magnitudes (dR angle, dv, dx) over time
       - ||Omega_delta|| over time (proxy for how "surprised" the
         attitude correction is each frame — should trend to ~0)
       - dt and num_substeps over time (useful for spotting the
         dropped/glitched-frame pattern directly)

  2. 3D trajectory + final landmark map (analog of paper Fig. 1,
     without a "true" overlay to align against).

What "working as intended" looks like
--------------------------------------
- residual_mean/median should drop after the observer initializes and
  then hover near your sensor noise floor (mm-level for a good D455
  RGB-D setup), NOT trend upward over the session.
- dR_angle / dv / dx should also drop and stay small after an initial
  settling period. Sudden isolated spikes in these (with a matching
  spike in dt or a "WARNING" print from Observer) point to a bad frame
  upstream (dropped frame, timestamp glitch, bad IMU window) rather
  than an observer bug — see the correlated dt subplot.
- num_substeps should be roughly stable (tracks map size + dt); wildly
  jumping values suggest dt itself is unstable (irregular frame timing).

Usage
-----
    # at the end of a run / periodically during a long one:
    observer.save_diagnostics("/tmp/li_slam_diag.npz")

    # offline:
    from li_slam.observer_diagnostics import plot_diagnostics_from_file
    plot_diagnostics_from_file("/tmp/li_slam_diag.npz", save_dir="/tmp/plots")

    # or, live in the same process, straight from Observer.diag:
    from li_slam.observer_diagnostics import plot_diagnostics
    plot_diagnostics(observer.diag, observer.state, save_dir="/tmp/plots")
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # safe default for headless ROS2 nodes; ignored if a GUI backend is already set
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)


def _load_npz_as_dict(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def plot_diagnostics_from_file(path: str, save_dir: str = None, show: bool = True,
                                landmarks: dict = None):
    """Load a .npz saved via Observer.save_diagnostics() and plot it."""
    diag = _load_npz_as_dict(path)
    plot_diagnostics(diag, landmarks=landmarks, save_dir=save_dir, show=show)


def plot_diagnostics(diag: dict, state=None, landmarks: dict = None,
                      save_dir: str = None, show: bool = True):
    """
    diag: Observer.diag (or the dict loaded from an .npz saved by
          Observer.save_diagnostics()).
    state: optional ObserverState — if given, its .landmarks are plotted
           as the final map in the trajectory figure. Alternatively pass
           `landmarks` directly (dict id -> (3,) array), e.g. if you only
           saved diagnostics offline and don't have the live state.
    save_dir: if given, saves PNGs there instead of/as well as showing.
    show: whether to call plt.show() (set False for headless/batch use).
    """
    t = np.asarray(diag['t'], dtype=np.float64)
    if len(t) == 0:
        print("No diagnostics recorded yet.")
        return
    t0 = t[0]
    t = t - t0   # relative time, easier to read

    fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
    fig.suptitle("LI-SLAM Observer Diagnostics (no ground truth — see docstring for how to read this)")

    # --- 1. Landmark residual norms ---
    ax = axes[0]
    ax.plot(t, diag['residual_mean'], label='mean', lw=1.2)
    ax.plot(t, diag['residual_median'], label='median', lw=1.2)
    ax.plot(t, diag['residual_max'], label='max', lw=0.8, alpha=0.6)
    ax.set_ylabel('residual |y_i - yhat_i| (m)')
    ax.set_title('Landmark residuals — should settle low after init, not drift up')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    # --- 2. Correction step magnitudes ---
    ax = axes[1]
    ax.plot(t, np.degrees(diag['dR_angle']), label='dR (deg)', lw=1.2)
    ax.plot(t, diag['dv'], label='dv (m/s)', lw=1.2)
    ax.plot(t, diag['dx'], label='dx (m)', lw=1.2)
    ax.plot(t, diag['dP_mean'], label='dP mean (m)', lw=1.0, alpha=0.7)
    ax.set_ylabel('correction step size')
    ax.set_title('Correction magnitude per frame — spikes = bad frame upstream, not necessarily observer bug')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    # --- 3. Omega_delta norm (attitude correction "effort") ---
    ax = axes[2]
    ax.plot(t, diag['omega_delta_norm'], color='tab:red', lw=1.0)
    ax.set_ylabel('||Omega_delta|| (rad/s)')
    ax.set_title('Attitude correction effort — should trend toward 0 as filter converges')
    ax.grid(alpha=0.3)

    # --- 4. dt and num_substeps (diagnose frame-timing issues) ---
    ax = axes[3]
    ax.plot(t, diag['dt'], color='tab:purple', lw=1.0, label='dt (s)')
    ax.set_ylabel('frame dt (s)')
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(t, diag['num_substeps'], color='tab:orange', lw=0.8, alpha=0.6, label='num_substeps')
    ax2.set_ylabel('correction sub-steps')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax.set_title('Frame timing — spikes in dt line up with residual/correction spikes if upstream framing is the cause')

    # --- 5. map size / landmarks per frame ---
    ax = axes[4]
    ax.plot(t, diag['map_size'], label='map size (total)', lw=1.2)
    ax.plot(t, diag['num_landmarks'], label='landmarks used this frame', lw=1.0, alpha=0.7)
    ax.set_ylabel('landmark count')
    ax.set_xlabel('time (s)')
    ax.set_title('Map growth')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, 'observer_error_metrics.png'), dpi=150)

    # --- Trajectory + landmark map (paper Fig. 1 analog, no ground truth) ---
    xs = np.asarray(diag['x'], dtype=np.float64)   # (T, 3)
    fig2 = plt.figure(figsize=(8, 8))
    ax3d = fig2.add_subplot(111, projection='3d')
    ax3d.plot(xs[:, 0], xs[:, 1], xs[:, 2], color='tab:blue', lw=1.5, label='Est. robot trajectory')
    ax3d.scatter(*xs[0], color='green', s=40, label='start')
    ax3d.scatter(*xs[-1], color='red', s=40, marker='*', label='end')

    lm_dict = landmarks
    if lm_dict is None and state is not None:
        lm_dict = state.landmarks
    if lm_dict:
        pts = np.array(list(lm_dict.values()))
        ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color='gray', s=8, alpha=0.6, label='Est. landmarks')

    ax3d.set_xlabel('x (m)')
    ax3d.set_ylabel('y (m)')
    ax3d.set_zlabel('z (m)')
    ax3d.set_title('Estimated trajectory + landmark map (no ground truth overlay)')
    ax3d.legend()

    if save_dir:
        import os
        fig2.savefig(os.path.join(save_dir, 'observer_trajectory.png'), dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig2)
