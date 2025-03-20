import math

import numpy as np

from hardspheres_2d._core import HardSpheres, update_spheres_from_bin


def update_spheres(s: HardSpheres) -> list[HardSpheres]:
    return update_spheres_from_bin(s)


def generate_velocities(
    m: np.ndarray, dim: int, T: float, kb: float = 1.0
) -> np.ndarray:
    n = len(m)
    if np.unique(m).shape[0] == 1:
        m = m[0]
    else:
        raise NotImplementedError(
            "Not yet implemented a sampling of velocities for spheres with different masses."
        )
    std_dev = math.sqrt(kb * T / m)
    velocities = np.random.normal(0, std_dev, size=(n, dim))

    return velocities
