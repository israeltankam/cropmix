"""Subprocess-based Python-to-R bridge for EpiPvr."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from importlib.resources import as_file, files
from pathlib import Path

import numpy as np
import pandas as pd

from ..errors import EpiPvrError
from .models import (
    AccessPeriodExperiment,
    EpiPvrFit,
    EpiPvrFitOptions,
    EpidemicProbabilityResult,
    LocalEpidemicParameters,
)


class EpiPvrBackend:
    """Call EpiPvr through `Rscript` while keeping the public API Python-only."""

    def __init__(self, rscript: str = "Rscript") -> None:
        self.rscript = rscript

    def check_installation(self) -> dict[str, object]:
        executable = shutil.which(self.rscript)
        if executable is None:
            return {"rscript": False, "epipvr": False, "message": "Rscript not found on PATH."}
        command = [
            executable,
            "-e",
            "cat(ifelse(requireNamespace('EpiPvr', quietly=TRUE), as.character(packageVersion('EpiPvr')), 'MISSING'))",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        version = completed.stdout.strip()
        installed = completed.returncode == 0 and version not in {"", "MISSING"}
        return {
            "rscript": True,
            "epipvr": installed,
            "version": version if installed else None,
            "message": completed.stderr.strip(),
        }

    def require_installation(self) -> None:
        status = self.check_installation()
        if not status["rscript"]:
            raise EpiPvrError("Rscript was not found. Install R and add Rscript to PATH.")
        if not status["epipvr"]:
            raise EpiPvrError(
                "EpiPvr is not installed in the R library visible to Rscript. "
                "Run in R once: install.packages('EpiPvr')."
            )

    def fit(
        self,
        experiment: AccessPeriodExperiment,
        *,
        options: EpiPvrFitOptions | None = None,
    ) -> EpiPvrFit:
        """Fit EpiPvr without exposing an R object to the Python caller."""
        self.require_installation()
        options = EpiPvrFitOptions() if options is None else options

        bridge = files("cropmix.epipvr").joinpath("resources", "fit_bridge.R")
        with tempfile.TemporaryDirectory(prefix="cropmix_epipvr_") as temporary:
            workdir = Path(temporary)
            input_dir = workdir / "input"
            output_dir = workdir / "output"
            experiment.write_bundle(input_dir)
            output_dir.mkdir()

            with as_file(bridge) as bridge_path:
                command = [
                    self.rscript,
                    str(bridge_path),
                    str(input_dir),
                    str(output_dir),
                    str(options.survival_upper_days),
                    str(options.d_num_pts_pd),
                    str(options.warmup),
                    str(options.iterations),
                    str(options.chains),
                    str(options.parallel),
                    str(options.seed),
                ]
                completed = subprocess.run(command, capture_output=True, text=True, check=False)

            if completed.returncode != 0:
                raise EpiPvrError(
                    "EpiPvr fitting failed.\n\nSTDOUT:\n"
                    + completed.stdout
                    + "\n\nSTDERR:\n"
                    + completed.stderr
                )

            posterior = pd.read_csv(output_dir / "posterior.csv")
            summary = pd.read_csv(output_dir / "summary.csv")
            diagnostics_frame = pd.read_csv(output_dir / "diagnostics.csv")
            diagnostics = dict(
                zip(diagnostics_frame["key"].astype(str), diagnostics_frame["value"].astype(str))
            )
            bayes_path = output_dir / "bayes_r2.csv"
            bayes_r2 = pd.read_csv(bayes_path) if bayes_path.exists() else pd.DataFrame()

        return EpiPvrFit(
            transmission_type=experiment.transmission_type,
            posterior_hourly=posterior,
            summary_table=summary,
            diagnostics=diagnostics,
            bayes_r2=bayes_r2,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def epidemic_probability(
        self,
        *,
        vectors_per_plant: int,
        virus_parameters_per_day: tuple[float, float, float],
        local_parameters: LocalEpidemicParameters,
        initial_interval: float = 0.1,
    ) -> EpidemicProbabilityResult:
        """Call EpiPvr's branching-process epidemic-probability calculator."""
        self.require_installation()
        if vectors_per_plant <= 0:
            raise EpiPvrError("vectors_per_plant must be positive.")

        bridge = files("cropmix.epipvr").joinpath("resources", "epidemic_bridge.R")
        with tempfile.TemporaryDirectory(prefix="cropmix_epipvr_bp_") as temporary:
            output = Path(temporary) / "probabilities.csv"
            with as_file(bridge) as bridge_path:
                command = [
                    self.rscript,
                    str(bridge_path),
                    str(output),
                    str(vectors_per_plant),
                    str(initial_interval),
                    *(str(float(x)) for x in virus_parameters_per_day),
                    str(local_parameters.dispersal_rate),
                    str(local_parameters.roguing_rate),
                    str(local_parameters.harvest_rate),
                    str(local_parameters.vector_mortality_rate),
                    str(local_parameters.plant_latent_progression_rate),
                ]
                completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                raise EpiPvrError(
                    "EpiPvr epidemic-probability calculation failed.\n"
                    + completed.stdout
                    + "\n"
                    + completed.stderr
                )
            probabilities = pd.read_csv(output)["probability"].to_numpy(float)

        return EpidemicProbabilityResult(
            probabilities=np.asarray(probabilities, dtype=float),
            vectors_per_plant=vectors_per_plant,
        )
