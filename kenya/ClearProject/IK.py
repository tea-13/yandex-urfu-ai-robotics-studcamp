from typing import Tuple

import numpy as np

class IK:
    def __init__(self, _l1, _l2, _dx, _dy):
        self.l1 = _l1
        self.l2 = _l2
        self.dx = _dx
        self.dy = _dy

    def calculate(self, x: float, y: float) -> Tuple[int, int]:
        x += self.dx
        y += self.dy

        b = x ** 2 + y ** 2
        q1 = np.atan2(y, x)
        q2 = np.acos((self.l1 ** 2 - self.l2 ** 2 + b) / (2 * self.l1 * np.sqrt(b)))

        phi1 = np.rad2deg(q1 + q2)

        phi2 = np.acos((self.l1 ** 2 + self.l2 ** 2 - b) / (2 * self.l1 * self.l2))
        phi2 = np.rad2deg(phi2)

        return round(phi1), round(phi2)