"""Adaptive callback for the FluxPipeline denoising loop."""

from typing import Dict, Optional

from .adapter import TransformerAdapter
from .controllers.base import InjectionController
from .drift.base import DriftMetric
from .logging_utils import StepLogger

import torch


class AdaptiveCallback:
    """Callback for callback_on_step_end that implements the adaptive control loop.

    At each denoising step (after scheduler.step):
    1. Compute drift between source (batch[0]) and edited (batch[1:]) latents
    2. Feed drift to the controller to get new alpha
    3. Update the adapter with new alpha for the next step
    4. Log step data

    The one-step lag (drift at step i -> alpha at step i+1) is standard for
    feedback controllers and does not cause issues in practice.
    """

    def __init__(
        self,
        controller: InjectionController,
        drift_metric: DriftMetric,
        adapter: TransformerAdapter,
        mask: Optional[torch.Tensor] = None,
        logger: Optional[StepLogger] = None,
    ):
        self.controller = controller
        self.drift_metric = drift_metric
        self.adapter = adapter
        self.mask = mask
        self.logger = logger
        self.total_steps: int = 0

    def __call__(
        self,
        pipe,
        step_index: int,
        timestep: int,
        callback_kwargs: Dict,
    ) -> Dict:
        latents = callback_kwargs["latents"]

        drift = self.drift_metric.compute(
            latents[0:1],
            latents[1:],
            self.mask,
        )

        alpha = self.controller.step(drift, step_index, self.total_steps)
        self.adapter.set_alpha(alpha)

        if self.logger is not None:
            self.logger.log_step(
                step_index=step_index,
                timestep=float(timestep),
                drift=drift,
                alpha=alpha,
            )

        return callback_kwargs
