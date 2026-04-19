from .base import InjectionController


class PIDController(InjectionController):
    """Proportional-Integral-Derivative controller for injection strength.

    Extends PDController with an integral term and anti-windup clamping.
    """

    def __init__(
        self,
        kp: float = 0.5,
        ki: float = 0.05,
        kd: float = 0.1,
        target_drift: float = 0.1,
        base_alpha: float = 0.8,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        integral_clamp: float = 2.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_drift = target_drift
        self.base_alpha = base_alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.integral_clamp = integral_clamp
        self._prev_error = 0.0
        self._integral = 0.0

    def reset(self):
        self._prev_error = 0.0
        self._integral = 0.0

    def step(self, drift: float, step_index: int, total_steps: int) -> float:
        error = drift - self.target_drift
        self._integral += error
        self._integral = max(-self.integral_clamp, min(self.integral_clamp, self._integral))
        derivative = error - self._prev_error
        self._prev_error = error

        alpha = (
            self.base_alpha
            + self.kp * error
            + self.ki * self._integral
            + self.kd * derivative
        )
        return max(self.alpha_min, min(self.alpha_max, alpha))
