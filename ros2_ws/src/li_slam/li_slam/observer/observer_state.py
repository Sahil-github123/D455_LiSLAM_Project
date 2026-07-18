from dataclasses import dataclass, field
import numpy as np

@dataclass
class ObserverState:

    R: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64))

    v: np.ndarray = field(
        default_factory=lambda: np.zeros(3))

    x: np.ndarray = field(
        default_factory=lambda: np.zeros(3))

    landmarks: dict = field(default_factory=dict)
