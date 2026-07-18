import numpy as np
from scipy.linalg import expm, logm


def skew(v: np.ndarray) -> np.ndarray:
    """R³ → so(3)."""

    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0]
    ], dtype=np.float64)
    

def vee(S: np.ndarray) -> np.ndarray:
    """so(3) → R³."""
    return np.array([
        S[2,1],
        S[0,2],
        S[1,0]
    ], dtype=np.float64)
    
def so3_exp(w: np.ndarray) -> np.ndarray:
    '''R³ → SO(3)'''
    return expm(skew(w))

def so3_log(R: np.ndarray) -> np.ndarray:
    '''SO(3) → R³'''
    return vee(logm(R))


def project_to_so3(R: np.ndarray) -> np.ndarray:
    '''Project a 3x3 matrix to the nearest valid rotation matrix in SO(3).'''

    U, _, Vt = np.linalg.svd(R)

    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return R


def rotation_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return the rotation matrix R such that:

        R @ a_hat ≈ b_hat

    where a_hat and b_hat are normalized versions of a and b.
    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm < 1e-12 or b_norm < 1e-12:
        raise ValueError(
            "Cannot compute rotation from a zero-length vector"
        )

    a = a / a_norm
    b = b / b_norm

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    # Already aligned
    if s < 1e-12 and c > 0.0:
        return np.eye(3, dtype=np.float64)

    # Opposite directions: 180° rotation.
    # Need to choose any axis perpendicular to a.
    if s < 1e-12 and c < 0.0:

        # Pick the coordinate axis least parallel to a
        if abs(a[0]) < abs(a[1]) and abs(a[0]) < abs(a[2]):
            axis = np.array([1.0, 0.0, 0.0])
        elif abs(a[1]) < abs(a[2]):
            axis = np.array([0.0, 1.0, 0.0])
        else:
            axis = np.array([0.0, 0.0, 1.0])

        axis = axis - np.dot(axis, a) * a
        axis /= np.linalg.norm(axis)

        # Rotation by π:
        # R = -I + 2 uuᵀ
        return -np.eye(3) + 2.0 * np.outer(axis, axis)

    # Rodrigues formula
    vx = skew(v)

    R = (
        np.eye(3)
        + vx
        + vx @ vx * ((1.0 - c) / (s ** 2))
    )

    return project_to_so3(R)
