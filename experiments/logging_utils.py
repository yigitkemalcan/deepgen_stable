"""JSONL step logger for adaptive injection experiments."""

import json
import os
from typing import Optional


class StepLogger:
    """Logs per-step controller data to a JSONL file."""

    def __init__(self, log_dir: str, filename: str = "step_log.jsonl"):
        os.makedirs(log_dir, exist_ok=True)
        self._path = os.path.join(log_dir, filename)
        self._records = []
        # Truncate file at start
        with open(self._path, "w") as f:
            pass

    def log_step(
        self,
        step_index: int,
        timestep: float,
        drift: float,
        alpha: float,
        **extra,
    ):
        record = {
            "step": step_index,
            "timestep": timestep,
            "drift": drift,
            "alpha": alpha,
            **extra,
        }
        self._records.append(record)
        with open(self._path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def finalize(self, output_dir: Optional[str] = None):
        """Write a summary JSON with full trajectories."""
        summary_dir = output_dir or os.path.dirname(self._path)
        summary_path = os.path.join(summary_dir, "summary.json")
        summary = {
            "num_steps": len(self._records),
            "drift_trajectory": [r["drift"] for r in self._records],
            "alpha_trajectory": [r["alpha"] for r in self._records],
        }
        if self._records:
            summary["drift_mean"] = sum(summary["drift_trajectory"]) / len(self._records)
            summary["alpha_mean"] = sum(summary["alpha_trajectory"]) / len(self._records)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    @property
    def path(self):
        return self._path
