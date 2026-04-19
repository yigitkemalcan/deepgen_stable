from .base import InjectionController


class PDController(InjectionController):
    """Proportional-Derivative controller for injection strength.

    Computes:
        error = drift - target_drift
        derivative = error - prev_error
        alpha = clip(base_alpha + kp * error + kd * derivative, alpha_min, alpha_max)
    """

    def __init__(
        self,
        kp: float = 0.5,
        kd: float = 0.1,
        target_drift: float = 0.1,
        base_alpha: float = 0.8,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
    ):
        self.kp = kp
        self.kd = kd
        self.target_drift = target_drift
        self.base_alpha = base_alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self._prev_error = 0.0

    def reset(self):
        self._prev_error = 0.0

    def step(self, drift: float, step_index: int, total_steps: int) -> float:
        error = drift - self.target_drift
        derivative = error - self._prev_error
        self._prev_error = error

        alpha = self.base_alpha + self.kp * error + self.kd * derivative
        return max(self.alpha_min, min(self.alpha_max, alpha))
