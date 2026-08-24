"""Cropmix: spatial crop varietal-mixture epidemiology."""

from .biology import (
    HostTransmission,
    PathogenParameters,
    PlantParameters,
    TransmissionDraw,
    Variety,
    VectorParameters,
    YieldParameters,
)
from .calibration import (
    KernelCalibrationResult,
    MeanFieldConsistencyResult,
    assess_mean_field_consistency,
    calibrate_kernel,
)
from .design import MixtureDesign
from .geometry import Field
from .kernels import ExponentialKernel, PreparedKernel
from .meanfield import MeanFieldResult, solve_mean_field
from .optimization import OptimizationConfig, OptimizationResult, optimize_mixture
from .results import SimulationResult
from .scenario import Inoculum, Scenario, VectorInoculum
from .simulation import simulate_mixture
from .system import CropMixSystem

__version__ = "0.2.0"

__all__ = [
    "CropMixSystem",
    "ExponentialKernel",
    "Field",
    "HostTransmission",
    "Inoculum",
    "KernelCalibrationResult",
    "MeanFieldConsistencyResult",
    "MeanFieldResult",
    "MixtureDesign",
    "OptimizationConfig",
    "OptimizationResult",
    "PathogenParameters",
    "PlantParameters",
    "PreparedKernel",
    "Scenario",
    "SimulationResult",
    "TransmissionDraw",
    "Variety",
    "VectorInoculum",
    "VectorParameters",
    "YieldParameters",
    "assess_mean_field_consistency",
    "calibrate_kernel",
    "optimize_mixture",
    "simulate_mixture",
    "solve_mean_field",
]
