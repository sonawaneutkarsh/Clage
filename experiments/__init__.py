"""Reproducible experimental framework for Clage.

Runs one-factor-at-a-time environmental sweeps on the validated NEAT engine
against the artificial-life world, with config-driven parameters, multiple
independent seeds, machine-readable raw results, and a cross-trial analysis
layer.

CLI:
    python -m experiments.run --config experiments/configs/food_abundance.json
    python -m experiments.analyze --results results/food_abundance
"""

from .config import PARAMETERS, Condition, ExperimentConfig, ResolvedConfig, load_experiment
from .metrics import compute_metrics

__all__ = [
    "PARAMETERS",
    "Condition",
    "ExperimentConfig",
    "ResolvedConfig",
    "compute_metrics",
    "load_experiment",
]
