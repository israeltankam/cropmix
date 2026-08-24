"""Finite spatial stochastic simulation using an accelerated Gillespie CTMC.

The public spatial engine currently implements semi-persistent transmission (SPT).
A Numba-compiled event loop keeps the exact stochastic process practical for
moderate fields and Monte Carlo ensembles while retaining a clear Python API.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .biology import TransmissionDraw
from .design import MixtureDesign
from .errors import ValidationError
from .results import SimulationResult
from .scenario import Scenario
from .system import CropMixSystem

S_STATE = np.int8(0)
L_STATE = np.int8(1)
I_STATE = np.int8(2)


@njit(cache=True)
def _weighted_index(weights: np.ndarray, total: float, u: float) -> int:
    """Sample an index from non-negative weights with known positive total."""
    target = u * total
    cumulative = 0.0
    for index in range(weights.shape[0]):
        cumulative += weights[index]
        if cumulative >= target:
            return index
    return weights.shape[0] - 1


@njit(cache=True)
def _sample_cdf_row(cdf: np.ndarray, row: int, u: float) -> int:
    """Binary-search one row of a cumulative destination matrix."""
    low = 0
    high = cdf.shape[1] - 1
    while low < high:
        middle = (low + high) // 2
        if u <= cdf[row, middle]:
            high = middle
        else:
            low = middle + 1
    return low


@njit(cache=True)
def _record_state(
    obs_index: int,
    plant_state: np.ndarray,
    cultivar: np.ndarray,
    vir: np.ndarray,
    variety_counts: np.ndarray,
    vectors_per_plant: int,
    incidence: np.ndarray,
    vector_prevalence: np.ndarray,
    incidence_by_variety: np.ndarray,
) -> None:
    n_sites = plant_state.shape[0]
    n_varieties = variety_counts.shape[0]

    total_infectious = 0
    infectious_by_variety = np.zeros(n_varieties, dtype=np.int64)
    vir_total = 0

    for i in range(n_sites):
        if plant_state[i] == I_STATE:
            total_infectious += 1
            infectious_by_variety[cultivar[i]] += 1
        for origin in range(n_varieties):
            vir_total += vir[i, origin]

    incidence[obs_index] = total_infectious / n_sites
    vector_prevalence[obs_index] = vir_total / (n_sites * vectors_per_plant)

    for variety in range(n_varieties):
        if variety_counts[variety] > 0:
            incidence_by_variety[variety, obs_index] = (
                infectious_by_variety[variety] / variety_counts[variety]
            )
        else:
            incidence_by_variety[variety, obs_index] = np.nan


@njit(cache=True)
def _single_gillespie_numba(
    cultivar: np.ndarray,
    destination_cdf: np.ndarray,
    vectors_per_plant: int,
    initial_sites: np.ndarray,
    observation_times: np.ndarray,
    duration: float,
    event_seed: int,
    acquisition: np.ndarray,
    inoculation: np.ndarray,
    gamma: np.ndarray,
    rho: np.ndarray,
    sigma: float,
    omega: float,
    clearance: float,
    yield_healthy: np.ndarray,
    yield_infected: np.ndarray,
    initial_vector_fraction: float,
    initial_origin_weights: np.ndarray,
):
    """One exact SPT trajectory of the tracked finite-state process.

    Vector movement uses the exact thinning representation of the conservative
    unordered pair-swap generator described in the scientific documentation.
    """
    np.random.seed(event_seed)

    n_sites = cultivar.shape[0]
    n_varieties = acquisition.shape[0]
    n_obs = observation_times.shape[0]

    plant_state = np.zeros(n_sites, dtype=np.int8)
    for z in range(initial_sites.shape[0]):
        plant_state[initial_sites[z]] = I_STATE

    # vir[i, v] = number of virus-bearing vectors at site i whose acquisition
    # origin was variety v. Vector infection pressure may seed virus-bearing
    # vectors independently of plant inoculum.
    vir = np.zeros((n_sites, n_varieties), dtype=np.int32)
    if initial_vector_fraction > 0.0:
        cumulative_origin = np.cumsum(initial_origin_weights)
        for i in range(n_sites):
            for _ in range(vectors_per_plant):
                if np.random.random() < initial_vector_fraction:
                    u = np.random.random()
                    origin = 0
                    while origin < n_varieties - 1 and u > cumulative_origin[origin]:
                        origin += 1
                    vir[i, origin] += 1

    variety_counts = np.zeros(n_varieties, dtype=np.int64)
    for i in range(n_sites):
        variety_counts[cultivar[i]] += 1

    incidence = np.empty(n_obs, dtype=np.float64)
    vector_prevalence = np.empty(n_obs, dtype=np.float64)
    incidence_by_variety = np.empty((n_varieties, n_obs), dtype=np.float64)

    vir_site = np.zeros(n_sites, dtype=np.float64)
    inoc_rates = np.zeros(n_sites, dtype=np.float64)
    prog_rates = np.zeros(n_sites, dtype=np.float64)
    rogue_rates = np.zeros(n_sites, dtype=np.float64)
    acq_rates = np.zeros(n_sites, dtype=np.float64)

    obs_index = 0
    time_now = 0.0

    while obs_index < n_obs and observation_times[obs_index] <= 0.0:
        _record_state(
            obs_index,
            plant_state,
            cultivar,
            vir,
            variety_counts,
            vectors_per_plant,
            incidence,
            vector_prevalence,
            incidence_by_variety,
        )
        obs_index += 1

    while time_now < duration:
        inoc_total = 0.0
        prog_total = 0.0
        rogue_total = 0.0
        acq_total = 0.0
        total_vir = 0

        for i in range(n_sites):
            v_i = 0
            for origin in range(n_varieties):
                v_i += vir[i, origin]
            vir_site[i] = v_i
            total_vir += v_i

            cultivar_i = cultivar[i]
            state_i = plant_state[i]

            if state_i == S_STATE:
                inoc_rates[i] = inoculation[cultivar_i] * v_i
            else:
                inoc_rates[i] = 0.0
            inoc_total += inoc_rates[i]

            if state_i == L_STATE:
                prog_rates[i] = gamma[cultivar_i]
            else:
                prog_rates[i] = 0.0
            prog_total += prog_rates[i]

            if state_i == I_STATE:
                rogue_rates[i] = rho[cultivar_i]
                healthy_i = vectors_per_plant - v_i
                if healthy_i < 0:
                    # This should be impossible if swap/acquisition invariants hold.
                    return (
                        np.nan,
                        np.nan,
                        plant_state,
                        vir,
                        incidence,
                        vector_prevalence,
                        incidence_by_variety,
                        1,
                    )
                acq_rates[i] = acquisition[cultivar_i] * healthy_i
            else:
                rogue_rates[i] = 0.0
                acq_rates[i] = 0.0
            rogue_total += rogue_rates[i]
            acq_total += acq_rates[i]

        movement_proposal_total = sigma * total_vir
        mortality_total = omega * total_vir
        clearance_total = clearance * total_vir

        total_rate = (
            inoc_total
            + prog_total
            + rogue_total
            + acq_total
            + movement_proposal_total
            + mortality_total
            + clearance_total
        )

        if total_rate <= 0.0:
            break

        waiting = -np.log(max(np.random.random(), 1e-15)) / total_rate
        event_time = time_now + waiting

        while obs_index < n_obs and observation_times[obs_index] < event_time:
            _record_state(
                obs_index,
                plant_state,
                cultivar,
                vir,
                variety_counts,
                vectors_per_plant,
                incidence,
                vector_prevalence,
                incidence_by_variety,
            )
            obs_index += 1

        if event_time > duration:
            time_now = duration
            break

        time_now = event_time
        draw = np.random.random() * total_rate
        threshold = inoc_total

        # Plant inoculation S -> L.
        if draw < threshold:
            site = _weighted_index(inoc_rates, inoc_total, np.random.random())
            plant_state[site] = L_STATE
            continue

        # End of plant latency L -> I.
        threshold += prog_total
        if draw < threshold:
            site = _weighted_index(prog_rates, prog_total, np.random.random())
            plant_state[site] = I_STATE
            continue

        # Roguing I -> S.
        threshold += rogue_total
        if draw < threshold:
            site = _weighted_index(rogue_rates, rogue_total, np.random.random())
            plant_state[site] = S_STATE
            continue

        # Virus acquisition by a healthy vector on an infectious plant.
        threshold += acq_total
        if draw < threshold:
            site = _weighted_index(acq_rates, acq_total, np.random.random())
            origin = cultivar[site]
            vir[site, origin] += 1
            continue

        # Conservative vector-exchange proposal.
        threshold += movement_proposal_total
        if draw < threshold:
            if total_vir <= 0:
                continue

            source = _weighted_index(vir_site, float(total_vir), np.random.random())
            source_total = int(vir_site[source])
            if source_total <= 0:
                continue

            source_weights = vir[source].astype(np.float64)
            origin = _weighted_index(source_weights, float(source_total), np.random.random())
            destination = _sample_cdf_row(destination_cdf, source, np.random.random())

            destination_total = 0
            for v in range(n_varieties):
                destination_total += vir[destination, v]
            destination_healthy = vectors_per_plant - destination_total

            ticket = np.random.random() * vectors_per_plant
            if ticket < destination_healthy:
                # Virus-bearing <-> virus-free exchange.
                vir[source, origin] -= 1
                vir[destination, origin] += 1
            else:
                # Reciprocal vector is also virus-bearing. Determine its origin.
                target = ticket - destination_healthy
                cumulative = 0.0
                reciprocal_origin = n_varieties - 1
                for v in range(n_varieties):
                    cumulative += vir[destination, v]
                    if target < cumulative:
                        reciprocal_origin = v
                        break

                # Equal-origin exchange is state-null. Different-origin exchange
                # is proposed from both directions, so accept with probability 1/2.
                if reciprocal_origin != origin and np.random.random() < 0.5:
                    vir[source, origin] -= 1
                    vir[source, reciprocal_origin] += 1
                    vir[destination, reciprocal_origin] -= 1
                    vir[destination, origin] += 1
            continue

        # Virus-bearing-vector mortality with immediate virus-free replacement.
        threshold += mortality_total
        if draw < threshold:
            site = _weighted_index(vir_site, float(total_vir), np.random.random())
            site_total = int(vir_site[site])
            origin = _weighted_index(
                vir[site].astype(np.float64), float(site_total), np.random.random()
            )
            vir[site, origin] -= 1
            continue

        # Loss of vector infectivity; return to virus-free class is implicit.
        site = _weighted_index(vir_site, float(total_vir), np.random.random())
        site_total = int(vir_site[site])
        origin = _weighted_index(
            vir[site].astype(np.float64), float(site_total), np.random.random()
        )
        vir[site, origin] -= 1

    while obs_index < n_obs:
        _record_state(
            obs_index,
            plant_state,
            cultivar,
            vir,
            variety_counts,
            vectors_per_plant,
            incidence,
            vector_prevalence,
            incidence_by_variety,
        )
        obs_index += 1

    final_yield = 0.0
    final_infectious = 0
    for i in range(n_sites):
        variety = cultivar[i]
        if plant_state[i] == I_STATE:
            final_yield += yield_infected[variety]
            final_infectious += 1
        else:
            final_yield += yield_healthy[variety]

    final_yield /= n_sites
    final_incidence = final_infectious / n_sites

    return (
        final_yield,
        final_incidence,
        plant_state,
        vir,
        incidence,
        vector_prevalence,
        incidence_by_variety,
        0,
    )


@njit(cache=True, parallel=True)
def _batch_gillespie_numba(
    cultivar: np.ndarray,
    destination_cdf: np.ndarray,
    vectors_per_plant: int,
    initial_sites_table: np.ndarray,
    observation_times: np.ndarray,
    duration: float,
    event_seeds: np.ndarray,
    acquisition: np.ndarray,
    inoculation: np.ndarray,
    gamma: np.ndarray,
    rho: np.ndarray,
    sigma: float,
    omega: float,
    clearance: float,
    yield_healthy: np.ndarray,
    yield_infected: np.ndarray,
    initial_vector_fraction: float,
    initial_origin_weights: np.ndarray,
):
    n_runs = event_seeds.shape[0]
    n_sites = cultivar.shape[0]
    n_varieties = acquisition.shape[0]
    n_obs = observation_times.shape[0]

    yields = np.empty(n_runs, dtype=np.float64)
    final_incidence = np.empty(n_runs, dtype=np.float64)
    final_states = np.empty((n_runs, n_sites), dtype=np.int8)
    incidence = np.empty((n_runs, n_obs), dtype=np.float64)
    vector_prevalence = np.empty((n_runs, n_obs), dtype=np.float64)
    by_variety = np.empty((n_runs, n_varieties, n_obs), dtype=np.float64)
    errors = np.zeros(n_runs, dtype=np.int8)

    for run in prange(n_runs):
        result = _single_gillespie_numba(
            cultivar,
            destination_cdf,
            vectors_per_plant,
            initial_sites_table[run],
            observation_times,
            duration,
            int(event_seeds[run]),
            acquisition,
            inoculation,
            gamma,
            rho,
            sigma,
            omega,
            clearance,
            yield_healthy,
            yield_infected,
            initial_vector_fraction,
            initial_origin_weights,
        )
        yields[run] = result[0]
        final_incidence[run] = result[1]
        final_states[run] = result[2]
        incidence[run] = result[4]
        vector_prevalence[run] = result[5]
        by_variety[run] = result[6]
        errors[run] = result[7]

    return yields, final_incidence, final_states, incidence, vector_prevalence, by_variety, errors


def _resolve_transmission_draw(
    system: CropMixSystem,
    transmission_draw: TransmissionDraw | None,
) -> TransmissionDraw:
    draw = system.point_transmission_draw() if transmission_draw is None else transmission_draw
    expected = set(system.variety_names)
    if set(draw.acquisition_rates) != expected or set(draw.inoculation_rates) != expected:
        raise ValidationError(
            "TransmissionDraw must contain exactly the varieties present in CropMixSystem."
        )
    return draw


def _initial_sites_table(
    scenario: Scenario,
    n_sites: int,
    n_runs: int,
    seed: int,
) -> np.ndarray:
    inoculum = scenario.inoculum
    if inoculum.sites is not None:
        sites = np.asarray(inoculum.sites, dtype=np.int64)
        if np.any(sites < 0) or np.any(sites >= n_sites):
            raise ValidationError("An explicit inoculum site is outside the field.")
        return np.repeat(sites[None, :], n_runs, axis=0)

    if inoculum.count > n_sites:
        raise ValidationError("Inoculum count exceeds the number of field sites.")

    table = np.empty((n_runs, inoculum.count), dtype=np.int64)
    for run in range(n_runs):
        rng = np.random.default_rng(seed + 10_000_000 + run)
        table[run] = rng.choice(n_sites, size=inoculum.count, replace=False)
    return table


def simulate_mixture(
    design: MixtureDesign,
    system: CropMixSystem,
    scenario: Scenario,
    *,
    n_runs: int = 100,
    seed: int = 12345,
    observation_times: np.ndarray | None = None,
    transmission_draw: TransmissionDraw | None = None,
    store_final_states: bool = True,
) -> SimulationResult:
    """Simulate a supplied planting design under the spatial SPT model.

    Parameters
    ----------
    design:
        Arbitrary assignment of system varieties to field coordinates.
    system:
        Varieties, vector/pathogen parameters and movement kernel.
    scenario:
        Season duration, vector burden and initial inoculum.
    n_runs:
        Number of stochastic epidemic replicates.
    seed:
        Master seed. Reusing it across candidate designs supplies common random
        numbers for paired comparisons and optimization.
    observation_times:
        Times at which trajectories are recorded. Defaults to 101 equally
        spaced points including 0 and harvest.
    transmission_draw:
        Optional coherent parameter draw. If omitted, point values in `system`
        are used.
    store_final_states:
        If false, final state arrays are discarded after computing summaries.

    Notes
    -----
    The event loop is Numba compiled. The first call in a Python process pays a
    compilation cost; subsequent calls reuse cached compiled functions when
    possible and are substantially faster than the reference pure-Python loop.
    """
    if n_runs <= 0:
        raise ValidationError("n_runs must be positive.")

    system.ensure_spatial_supported()
    system.validate_design(design)

    if observation_times is None:
        observation_times = np.linspace(0.0, scenario.duration, 101)
    else:
        observation_times = np.asarray(observation_times, dtype=float)
        if observation_times.ndim != 1 or len(observation_times) == 0:
            raise ValidationError("observation_times must be a non-empty 1D array.")
        if np.any(np.diff(observation_times) < 0):
            raise ValidationError("observation_times must be sorted.")
        if observation_times[0] < 0 or observation_times[-1] > scenario.duration:
            raise ValidationError("observation_times must lie inside [0, scenario.duration].")

    resolved_draw = _resolve_transmission_draw(system, transmission_draw)
    prepared_kernel = system.kernel.prepare(design.field)

    names = system.variety_names
    variety_to_index = {name: i for i, name in enumerate(names)}
    cultivar = np.asarray([variety_to_index[name] for name in design.assignment], dtype=np.int16)

    acquisition = np.asarray(
        [resolved_draw.acquisition_rates[name] for name in names], dtype=np.float64
    )
    inoculation = np.asarray(
        [resolved_draw.inoculation_rates[name] for name in names], dtype=np.float64
    )
    gamma = np.asarray(
        [variety.plant.latent_progression_rate for variety in system.varieties],
        dtype=np.float64,
    )
    rho = np.asarray(
        [variety.plant.roguing_rate for variety in system.varieties], dtype=np.float64
    )
    yield_healthy = np.asarray(
        [variety.yield_model.healthy for variety in system.varieties], dtype=np.float64
    )
    yield_infected = np.asarray(
        [variety.yield_model.infected for variety in system.varieties], dtype=np.float64
    )

    vector_inoculum = scenario.vector_inoculum
    if vector_inoculum.origin_weights is None:
        counts = np.bincount(cultivar.astype(np.int64), minlength=len(names)).astype(float)
        initial_origin_weights = counts / counts.sum()
    else:
        if len(vector_inoculum.origin_weights) != len(names):
            raise ValidationError(
                "vector_inoculum.origin_weights must have one value per system variety."
            )
        initial_origin_weights = np.asarray(vector_inoculum.origin_weights, dtype=np.float64)
        initial_origin_weights = initial_origin_weights / initial_origin_weights.sum()

    destination_cdf = np.cumsum(prepared_kernel.probabilities, axis=1)
    destination_cdf[:, -1] = 1.0

    initial_sites = _initial_sites_table(scenario, design.n_sites, n_runs, seed)
    event_seeds = np.arange(seed, seed + n_runs, dtype=np.int64)

    (
        yields,
        final_incidence,
        final_states,
        incidence,
        vector_prevalence,
        by_variety,
        errors,
    ) = _batch_gillespie_numba(
        cultivar,
        destination_cdf,
        scenario.vectors_per_plant,
        initial_sites,
        np.asarray(observation_times, dtype=np.float64),
        float(scenario.duration),
        event_seeds,
        acquisition,
        inoculation,
        gamma,
        rho,
        float(system.vector.dispersal_rate),
        float(system.vector.mortality_rate),
        float(resolved_draw.vector_clearance_rate),
        yield_healthy,
        yield_infected,
        float(vector_inoculum.infectious_fraction),
        initial_origin_weights,
    )

    if np.any(errors != 0):
        raise RuntimeError("Vector-count invariant failed inside the compiled simulator.")

    final_states_output = final_states if store_final_states else None
    infection_probability = (
        np.mean(final_states == I_STATE, axis=0) if store_final_states else None
    )
    incidence_by_variety_runs = {
        name: by_variety[:, index, :] for index, name in enumerate(names)
    }

    return SimulationResult(
        design=design,
        scenario=scenario,
        time=np.asarray(observation_times, dtype=float).copy(),
        yield_runs=yields,
        final_incidence_runs=final_incidence,
        incidence_runs=incidence,
        vector_prevalence_runs=vector_prevalence,
        incidence_by_variety_runs=incidence_by_variety_runs,
        final_states=final_states_output,
        infection_probability=infection_probability,
        kernel_scale=prepared_kernel.scale,
        mean_step_distance=prepared_kernel.mean_step_distance,
        seed=seed,
        yield_unit=system.yield_unit,
    )
