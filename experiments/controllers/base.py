from abc import ABC, abstractmethod


class InjectionController(ABC):
    """Base class for injection-strength controllers."""

    @abstractmethod
    def reset(self):
        """Reset internal state for a new run."""
        ...

    @abstractmethod
    def step(self, drift: float, step_index: int, total_steps: int) -> float:
        """Compute injection alpha for the current step.

        Args:
            drift: Current drift measurement.
            step_index: Current denoising step index.
            total_steps: Total number of denoising steps.

        Returns:
            alpha in [0, 1] controlling injection strength.
        """
        ...
