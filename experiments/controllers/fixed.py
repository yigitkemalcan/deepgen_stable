import math
from typing import Callable, Optional

from .base import InjectionController


class FixedAlphaController(InjectionController):
    """Returns a constant alpha at every step."""

    def __init__(self, alpha: float = 0.8):
        self.alpha = alpha

    def reset(self):
        pass

    def step(self, drift: float, step_index: int, total_steps: int) -> float:
        return self.alpha


class ScheduledAlphaController(InjectionController):
    """Returns alpha from a schedule function f(step_index, total_steps) -> alpha."""

    def __init__(
        self,
        schedule_fn: Callable[[int, int], float],
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
    ):
        self.schedule_fn = schedule_fn
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def reset(self):
        pass

    def step(self, drift: float, step_index: int, total_steps: int) -> float:
        alpha = self.schedule_fn(step_index, total_steps)
        return max(self.alpha_min, min(self.alpha_max, alpha))


def linear_decay(start: float = 1.0, end: float = 0.0) -> Callable[[int, int], float]:
    """Linear interpolation from start to end over all steps."""
    def fn(step_index: int, total_steps: int) -> float:
        if total_steps <= 1:
            return start
        t = step_index / (total_steps - 1)
        return start + (end - start) * t
    return fn


def cosine_decay(start: float = 1.0, end: float = 0.0) -> Callable[[int, int], float]:
    """Cosine annealing from start to end."""
    def fn(step_index: int, total_steps: int) -> float:
        if total_steps <= 1:
            return start
        t = step_index / (total_steps - 1)
        return end + (start - end) * 0.5 * (1 + math.cos(math.pi * t))
    return fn
