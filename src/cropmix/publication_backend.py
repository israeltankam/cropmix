from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from numba import njit, prange
from scipy.signal import fftconvolve
from scipy.ndimage import convolve, label, gaussian_filter, distance_transform_edt


@dataclass
class StructuredKernel:
    N: int
    scale: float
    balance: np.ndarray
    displacement_dr: np.ndarray
    displacement_dc: np.ndarray
    displacement_cdf: np.ndarray
    mean_step_distance: float
    max_row_error: float
    iterations: int


def _offset_kernel_arrays(N: int, scale: float):
    offsets = np.arange(-(N - 1), N, dtype=float)
    dr, dc = np.meshgrid(offsets, offsets, indexing="ij")
    distance = np.sqrt(dr * dr + dc * dc)
    kernel = np.exp(-distance / scale)
    kernel[N - 1, N - 1] = 0.0
    return distance, kernel


def prepare_structured_kernel(
    N: int,
    scale: float,
    *,
    tolerance: float = 1e-8,
    max_iter: int = 5000,
    damping: float = 0.5,
) -> StructuredKernel:
    """Exact symmetric DWD balancing on an open N x N lattice by FFT convolution.

    Very small scales are numerically close to a nearest-neighbour graph and
    converge slowly under the direct symmetric fixed-point iteration. For that
    regime we use alternating Sinkhorn row/column updates and recover the
    symmetric scaling as sqrt(u*v); for larger scales the damped symmetric
    iteration is substantially faster.
    """
    distance, kernel = _offset_kernel_arrays(N, scale)
    error = np.inf

    if scale < 0.075:
        u = np.ones((N, N), dtype=float)
        v = np.ones((N, N), dtype=float)
        for iteration in range(1, max_iter + 1):
            u = 1.0 / fftconvolve(v, kernel, mode="same")
            v = 1.0 / fftconvolve(u, kernel, mode="same")
            if iteration % 20 == 0 or iteration == max_iter:
                d = np.sqrt(u * v)
                row_sums = d * fftconvolve(d, kernel, mode="same")
                error = float(np.max(np.abs(row_sums - 1.0)))
                if error < tolerance:
                    break
        d = np.sqrt(u * v)
    else:
        log_d = np.zeros((N, N), dtype=float)
        for iteration in range(1, max_iter + 1):
            d = np.exp(log_d)
            wd = fftconvolve(d, kernel, mode="same")
            target = -np.log(wd)
            log_d = (1.0 - damping) * log_d + damping * target
            if iteration % 5 == 0 or iteration == max_iter:
                d = np.exp(log_d)
                row_sums = d * fftconvolve(d, kernel, mode="same")
                error = float(np.max(np.abs(row_sums - 1.0)))
                if error < tolerance:
                    break
        d = np.exp(log_d)

    if not np.isfinite(error) or error >= tolerance:
        raise RuntimeError(
            f"Symmetric FFT balancing failed for a={scale:g}; max row error={error:.3e}"
        )

    cost_kernel = distance * kernel
    mean_step = float(np.sum(d * fftconvolve(d, cost_kernel, mode="same")) / (N * N))

    drs = []
    dcs = []
    weights = []
    for dr in range(-(N - 1), N):
        for dc in range(-(N - 1), N):
            if dr == 0 and dc == 0:
                continue
            drs.append(dr)
            dcs.append(dc)
            weights.append(math.exp(-math.hypot(dr, dc) / scale))

    weights = np.asarray(weights, dtype=float)
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    cdf[-1] = 1.0

    return StructuredKernel(
        N=N,
        scale=float(scale),
        balance=np.asarray(d.ravel(), dtype=np.float64),
        displacement_dr=np.asarray(drs, dtype=np.int16),
        displacement_dc=np.asarray(dcs, dtype=np.int16),
        displacement_cdf=np.asarray(cdf, dtype=np.float64),
        mean_step_distance=mean_step,
        max_row_error=error,
        iterations=iteration,
    )


@njit(cache=True)
def _fw_add(tree, idx, delta):
    n = tree.shape[0]
    i = idx + 1
    while i <= n:
        tree[i - 1] += delta
        i += i & -i


@njit(cache=True)
def _fw_total(tree):
    s = 0.0
    i = tree.shape[0]
    while i > 0:
        s += tree[i - 1]
        i -= i & -i
    return s


@njit(cache=True)
def _fw_find(tree, target):
    n = tree.shape[0]
    idx = 0
    bit = 1
    while (bit << 1) <= n:
        bit <<= 1
    cumulative = 0.0
    while bit:
        nxt = idx + bit
        if nxt <= n and cumulative + tree[nxt - 1] <= target:
            cumulative += tree[nxt - 1]
            idx = nxt
        bit >>= 1
    if idx >= n:
        idx = n - 1
    return idx


@njit(cache=True)
def _build_fenwick(values):
    tree = np.zeros(values.shape[0], dtype=np.float64)
    for i in range(values.shape[0]):
        _fw_add(tree, i, values[i])
    return tree


@njit(cache=True)
def _cdf_index(cdf, u):
    low = 0
    high = cdf.shape[0] - 1
    while low < high:
        middle = (low + high) // 2
        if u <= cdf[middle]:
            high = middle
        else:
            low = middle + 1
    return low


@njit(cache=True)
def _sample_balanced_destination(source, N, drs, dcs, displacement_cdf, balance, max_balance):
    """Exact rejection sampler for one row of P = D W D."""
    row = source // N
    col = source - row * N
    while True:
        k = _cdf_index(displacement_cdf, np.random.random())
        rr = row + drs[k]
        cc = col + dcs[k]
        if rr < 0 or rr >= N or cc < 0 or cc >= N:
            continue
        destination = rr * N + cc
        if np.random.random() <= balance[destination] / max_balance:
            return destination


@njit(cache=True)
def _update_site_rates(
    i,
    plant_state,
    cultivar,
    vir_total,
    vectors_per_plant,
    acquisition,
    inoculation,
    gamma,
    rho,
    inoc_rates,
    prog_rates,
    rogue_rates,
    acq_rates,
    inoc_tree,
    prog_tree,
    rogue_tree,
    acq_tree,
):
    if inoc_rates[i] != 0.0:
        _fw_add(inoc_tree, i, -inoc_rates[i])
    if prog_rates[i] != 0.0:
        _fw_add(prog_tree, i, -prog_rates[i])
    if rogue_rates[i] != 0.0:
        _fw_add(rogue_tree, i, -rogue_rates[i])
    if acq_rates[i] != 0.0:
        _fw_add(acq_tree, i, -acq_rates[i])

    inoc_rates[i] = 0.0
    prog_rates[i] = 0.0
    rogue_rates[i] = 0.0
    acq_rates[i] = 0.0

    variety = cultivar[i]
    state = plant_state[i]

    if state == 0:
        inoc_rates[i] = inoculation[variety] * vir_total[i]
        if inoc_rates[i] != 0.0:
            _fw_add(inoc_tree, i, inoc_rates[i])
    elif state == 1:
        prog_rates[i] = gamma[variety]
        if prog_rates[i] != 0.0:
            _fw_add(prog_tree, i, prog_rates[i])
    else:
        rogue_rates[i] = rho[variety]
        acq_rates[i] = acquisition[variety] * (vectors_per_plant - vir_total[i])
        if rogue_rates[i] != 0.0:
            _fw_add(rogue_tree, i, rogue_rates[i])
        if acq_rates[i] != 0.0:
            _fw_add(acq_tree, i, acq_rates[i])


@njit(cache=True)
def _record_state(
    obs_index,
    plant_state,
    cultivar,
    vir_total,
    vectors_per_plant,
    incidence,
    vector_prevalence,
    incidence_susc,
    incidence_res,
    n_susc,
    n_res,
):
    n_sites = plant_state.shape[0]
    infectious = 0
    infectious_susc = 0
    infectious_res = 0
    viruliferous = 0.0

    for i in range(n_sites):
        if plant_state[i] == 2:
            infectious += 1
            if cultivar[i] == 0:
                infectious_susc += 1
            else:
                infectious_res += 1
        viruliferous += vir_total[i]

    incidence[obs_index] = infectious / n_sites
    vector_prevalence[obs_index] = viruliferous / (n_sites * vectors_per_plant)
    incidence_susc[obs_index] = infectious_susc / n_susc if n_susc > 0 else np.nan
    incidence_res[obs_index] = infectious_res / n_res if n_res > 0 else np.nan


@njit(cache=True)
def _single_publication_ssa(
    N,
    cultivar,
    balance,
    drs,
    dcs,
    displacement_cdf,
    vectors_per_plant,
    initial_sites,
    observation_times,
    duration,
    event_seed,
    acquisition,
    inoculation,
    gamma,
    rho,
    sigma,
    omega,
    clearance,
    yield_healthy,
    yield_infected,
):
    np.random.seed(event_seed)

    n_sites = N * N
    plant_state = np.zeros(n_sites, dtype=np.int8)
    for k in range(initial_sites.shape[0]):
        plant_state[initial_sites[k]] = 2

    vir = np.zeros((n_sites, 2), dtype=np.int16)
    vir_total = np.zeros(n_sites, dtype=np.float64)

    inoc_rates = np.zeros(n_sites, dtype=np.float64)
    prog_rates = np.zeros(n_sites, dtype=np.float64)
    rogue_rates = np.zeros(n_sites, dtype=np.float64)
    acq_rates = np.zeros(n_sites, dtype=np.float64)

    n_susc = 0
    n_res = 0
    for i in range(n_sites):
        variety = cultivar[i]
        if variety == 0:
            n_susc += 1
        else:
            n_res += 1
        if plant_state[i] == 1:
            prog_rates[i] = gamma[variety]
        elif plant_state[i] == 2:
            rogue_rates[i] = rho[variety]
            acq_rates[i] = acquisition[variety] * vectors_per_plant

    inoc_tree = _build_fenwick(inoc_rates)
    prog_tree = _build_fenwick(prog_rates)
    rogue_tree = _build_fenwick(rogue_rates)
    acq_tree = _build_fenwick(acq_rates)
    vir_tree = _build_fenwick(vir_total)

    n_obs = observation_times.shape[0]
    incidence = np.empty(n_obs, dtype=np.float64)
    vector_prevalence = np.empty(n_obs, dtype=np.float64)
    incidence_susc = np.empty(n_obs, dtype=np.float64)
    incidence_res = np.empty(n_obs, dtype=np.float64)

    obs_index = 0
    time_now = 0.0
    event_count = 0
    max_balance = np.max(balance)

    while obs_index < n_obs and observation_times[obs_index] <= 0.0:
        _record_state(
            obs_index, plant_state, cultivar, vir_total, vectors_per_plant,
            incidence, vector_prevalence, incidence_susc, incidence_res,
            n_susc, n_res,
        )
        obs_index += 1

    while time_now < duration:
        inoc_total = _fw_total(inoc_tree)
        prog_total = _fw_total(prog_tree)
        rogue_total = _fw_total(rogue_tree)
        acq_total = _fw_total(acq_tree)
        total_vir = _fw_total(vir_tree)

        movement_total = sigma * total_vir
        mortality_total = omega * total_vir
        clearance_total = clearance * total_vir
        total_rate = (
            inoc_total + prog_total + rogue_total + acq_total
            + movement_total + mortality_total + clearance_total
        )
        if total_rate <= 0.0:
            break

        waiting = -math.log(max(np.random.random(), 1e-15)) / total_rate
        event_time = time_now + waiting

        while obs_index < n_obs and observation_times[obs_index] < event_time:
            _record_state(
                obs_index, plant_state, cultivar, vir_total, vectors_per_plant,
                incidence, vector_prevalence, incidence_susc, incidence_res,
                n_susc, n_res,
            )
            obs_index += 1

        if event_time > duration:
            time_now = duration
            break

        time_now = event_time
        draw = np.random.random() * total_rate

        if draw < inoc_total:
            site = _fw_find(inoc_tree, np.random.random() * inoc_total)
            plant_state[site] = 1
            _update_site_rates(
                site, plant_state, cultivar, vir_total, vectors_per_plant,
                acquisition, inoculation, gamma, rho,
                inoc_rates, prog_rates, rogue_rates, acq_rates,
                inoc_tree, prog_tree, rogue_tree, acq_tree,
            )
            event_count += 1
            continue

        threshold = inoc_total + prog_total
        if draw < threshold:
            site = _fw_find(prog_tree, np.random.random() * prog_total)
            plant_state[site] = 2
            _update_site_rates(
                site, plant_state, cultivar, vir_total, vectors_per_plant,
                acquisition, inoculation, gamma, rho,
                inoc_rates, prog_rates, rogue_rates, acq_rates,
                inoc_tree, prog_tree, rogue_tree, acq_tree,
            )
            event_count += 1
            continue

        threshold += rogue_total
        if draw < threshold:
            site = _fw_find(rogue_tree, np.random.random() * rogue_total)
            plant_state[site] = 0
            _update_site_rates(
                site, plant_state, cultivar, vir_total, vectors_per_plant,
                acquisition, inoculation, gamma, rho,
                inoc_rates, prog_rates, rogue_rates, acq_rates,
                inoc_tree, prog_tree, rogue_tree, acq_tree,
            )
            event_count += 1
            continue

        threshold += acq_total
        if draw < threshold:
            site = _fw_find(acq_tree, np.random.random() * acq_total)
            origin = cultivar[site]
            vir[site, origin] += 1
            vir_total[site] += 1.0
            _fw_add(vir_tree, site, 1.0)
            _update_site_rates(
                site, plant_state, cultivar, vir_total, vectors_per_plant,
                acquisition, inoculation, gamma, rho,
                inoc_rates, prog_rates, rogue_rates, acq_rates,
                inoc_tree, prog_tree, rogue_tree, acq_tree,
            )
            event_count += 1
            continue

        threshold += movement_total
        if draw < threshold:
            source = _fw_find(vir_tree, np.random.random() * total_vir)
            source_total = int(vir_total[source])
            ticket = np.random.randint(source_total)
            origin = 0 if ticket < vir[source, 0] else 1

            destination = _sample_balanced_destination(
                source, N, drs, dcs, displacement_cdf, balance, max_balance
            )
            destination_total = int(vir_total[destination])
            destination_healthy = vectors_per_plant - destination_total
            reciprocal_ticket = np.random.random() * vectors_per_plant

            if reciprocal_ticket < destination_healthy:
                vir[source, origin] -= 1
                vir[destination, origin] += 1
                vir_total[source] -= 1.0
                vir_total[destination] += 1.0
                _fw_add(vir_tree, source, -1.0)
                _fw_add(vir_tree, destination, 1.0)
                _update_site_rates(
                    source, plant_state, cultivar, vir_total, vectors_per_plant,
                    acquisition, inoculation, gamma, rho,
                    inoc_rates, prog_rates, rogue_rates, acq_rates,
                    inoc_tree, prog_tree, rogue_tree, acq_tree,
                )
                _update_site_rates(
                    destination, plant_state, cultivar, vir_total, vectors_per_plant,
                    acquisition, inoculation, gamma, rho,
                    inoc_rates, prog_rates, rogue_rates, acq_rates,
                    inoc_tree, prog_tree, rogue_tree, acq_tree,
                )
            else:
                target = reciprocal_ticket - destination_healthy
                reciprocal_origin = 0 if target < vir[destination, 0] else 1
                if reciprocal_origin != origin and np.random.random() < 0.5:
                    vir[source, origin] -= 1
                    vir[source, reciprocal_origin] += 1
                    vir[destination, reciprocal_origin] -= 1
                    vir[destination, origin] += 1
            event_count += 1
            continue

        threshold += mortality_total
        if draw < threshold:
            source = _fw_find(vir_tree, np.random.random() * total_vir)
            source_total = int(vir_total[source])
            ticket = np.random.randint(source_total)
            origin = 0 if ticket < vir[source, 0] else 1
            vir[source, origin] -= 1
            vir_total[source] -= 1.0
            _fw_add(vir_tree, source, -1.0)
            _update_site_rates(
                source, plant_state, cultivar, vir_total, vectors_per_plant,
                acquisition, inoculation, gamma, rho,
                inoc_rates, prog_rates, rogue_rates, acq_rates,
                inoc_tree, prog_tree, rogue_tree, acq_tree,
            )
            event_count += 1
            continue

        source = _fw_find(vir_tree, np.random.random() * total_vir)
        source_total = int(vir_total[source])
        ticket = np.random.randint(source_total)
        origin = 0 if ticket < vir[source, 0] else 1
        vir[source, origin] -= 1
        vir_total[source] -= 1.0
        _fw_add(vir_tree, source, -1.0)
        _update_site_rates(
            source, plant_state, cultivar, vir_total, vectors_per_plant,
            acquisition, inoculation, gamma, rho,
            inoc_rates, prog_rates, rogue_rates, acq_rates,
            inoc_tree, prog_tree, rogue_tree, acq_tree,
        )
        event_count += 1

    while obs_index < n_obs:
        _record_state(
            obs_index, plant_state, cultivar, vir_total, vectors_per_plant,
            incidence, vector_prevalence, incidence_susc, incidence_res,
            n_susc, n_res,
        )
        obs_index += 1

    final_yield = 0.0
    final_infectious = 0
    for i in range(n_sites):
        variety = cultivar[i]
        if plant_state[i] == 2:
            final_yield += yield_infected[variety]
            final_infectious += 1
        else:
            final_yield += yield_healthy[variety]

    final_yield /= n_sites
    final_incidence = final_infectious / n_sites

    return (
        final_yield,
        final_incidence,
        incidence,
        vector_prevalence,
        incidence_susc,
        incidence_res,
        plant_state,
        event_count,
    )


@njit(cache=True, parallel=True)
def batch_publication_ssa(
    N,
    cultivar,
    balance,
    drs,
    dcs,
    displacement_cdf,
    vectors_per_plant,
    initial_sites_table,
    observation_times,
    duration,
    event_seeds,
    acquisition,
    inoculation,
    gamma,
    rho,
    sigma,
    omega,
    clearance,
    yield_healthy,
    yield_infected,
):
    n_runs = event_seeds.shape[0]
    n_obs = observation_times.shape[0]
    n_sites = N * N

    yields = np.empty(n_runs, dtype=np.float64)
    final_incidence = np.empty(n_runs, dtype=np.float64)
    incidence = np.empty((n_runs, n_obs), dtype=np.float64)
    vector_prevalence = np.empty((n_runs, n_obs), dtype=np.float64)
    incidence_susc = np.empty((n_runs, n_obs), dtype=np.float64)
    incidence_res = np.empty((n_runs, n_obs), dtype=np.float64)
    final_states = np.empty((n_runs, n_sites), dtype=np.int8)
    event_counts = np.empty(n_runs, dtype=np.int64)

    for r in prange(n_runs):
        (
            y,
            fi,
            inc,
            vec,
            inc_s,
            inc_r,
            state,
            events,
        ) = _single_publication_ssa(
            N,
            cultivar,
            balance,
            drs,
            dcs,
            displacement_cdf,
            vectors_per_plant,
            initial_sites_table[r],
            observation_times,
            duration,
            int(event_seeds[r]),
            acquisition,
            inoculation,
            gamma,
            rho,
            sigma,
            omega,
            clearance,
            yield_healthy,
            yield_infected,
        )
        yields[r] = y
        final_incidence[r] = fi
        incidence[r] = inc
        vector_prevalence[r] = vec
        incidence_susc[r] = inc_s
        incidence_res[r] = inc_r
        final_states[r] = state
        event_counts[r] = events

    return (
        yields,
        final_incidence,
        incidence,
        vector_prevalence,
        incidence_susc,
        incidence_res,
        final_states,
        event_counts,
    )


def event_seeds(seed: int, n: int) -> np.ndarray:
    base = np.arange(n, dtype=np.uint64)
    values = (
        np.uint64(seed)
        + np.uint64(1_000_003) * base
        + np.uint64(97_531)
    ) % np.uint64(2_147_483_647)
    return values.astype(np.int64)


def random_initial_sites(n_runs: int, n_sites: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 31_337)
    out = np.empty((n_runs, count), dtype=np.int64)
    for r in range(n_runs):
        out[r] = rng.choice(n_sites, size=count, replace=False)
    return out


def fixed_initial_sites(n_runs: int, sites) -> np.ndarray:
    values = np.asarray(sites, dtype=np.int64)
    return np.tile(values[None, :], (n_runs, 1))


def mcse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.inf
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def run_ensemble(
    *,
    N: int,
    cultivar: np.ndarray,
    kernel: StructuredKernel,
    vectors_per_plant: int,
    initial_infectious: int,
    observation_times: np.ndarray,
    duration: float,
    seed: int,
    acquisition: np.ndarray,
    inoculation: np.ndarray,
    gamma: np.ndarray,
    rho: np.ndarray,
    sigma: float,
    omega: float,
    clearance: float,
    yield_healthy: np.ndarray,
    yield_infected: np.ndarray,
    min_runs: int,
    max_runs: int,
    batch_size: int,
    final_incidence_mcse_tol: float,
    relative_yield_mcse_tol: float,
    fixed_sites=None,
    require_precision: bool = True,
    store_infection_probability: bool = False,
):
    n_sites = N * N
    if fixed_sites is None:
        initial_table = random_initial_sites(max_runs, n_sites, initial_infectious, seed)
    else:
        initial_table = fixed_initial_sites(max_runs, fixed_sites)
    seeds = event_seeds(seed, max_runs)

    yield_parts = []
    final_parts = []
    incidence_parts = []
    vector_parts = []
    susc_parts = []
    res_parts = []
    event_parts = []
    infected_count = np.zeros(n_sites, dtype=np.int64) if store_infection_probability else None

    n_done = 0
    start = time.time()
    while n_done < max_runs:
        n_batch = min(batch_size, max_runs - n_done)
        sl = slice(n_done, n_done + n_batch)
        out = batch_publication_ssa(
            N,
            cultivar,
            kernel.balance,
            kernel.displacement_dr,
            kernel.displacement_dc,
            kernel.displacement_cdf,
            vectors_per_plant,
            initial_table[sl],
            np.asarray(observation_times, dtype=np.float64),
            float(duration),
            seeds[sl],
            acquisition,
            inoculation,
            gamma,
            rho,
            sigma,
            omega,
            clearance,
            yield_healthy,
            yield_infected,
        )
        yields, finals, inc, vec, inc_s, inc_r, states, events = out
        yield_parts.append(yields)
        final_parts.append(finals)
        incidence_parts.append(inc)
        vector_parts.append(vec)
        susc_parts.append(inc_s)
        res_parts.append(inc_r)
        event_parts.append(events)
        if infected_count is not None:
            infected_count += np.sum(states == 2, axis=0)
        n_done += n_batch

        all_y = np.concatenate(yield_parts)
        all_i = np.concatenate(final_parts)
        if n_done >= min_runs:
            y_mcse = mcse(all_y)
            i_mcse = mcse(all_i)
            rel_y = y_mcse / max(abs(float(np.mean(all_y))), 1e-12)
            if (not require_precision) or (
                i_mcse <= final_incidence_mcse_tol and rel_y <= relative_yield_mcse_tol
            ):
                break

    yields = np.concatenate(yield_parts)
    finals = np.concatenate(final_parts)
    incidence = np.concatenate(incidence_parts, axis=0)
    vectors = np.concatenate(vector_parts, axis=0)
    incidence_susc = np.concatenate(susc_parts, axis=0)
    incidence_res = np.concatenate(res_parts, axis=0)
    event_counts = np.concatenate(event_parts)

    result = {
        "n_runs": len(yields),
        "yield_runs": yields,
        "final_incidence_runs": finals,
        "incidence_runs": incidence,
        "vector_prevalence_runs": vectors,
        "incidence_susc_runs": incidence_susc,
        "incidence_res_runs": incidence_res,
        "time": np.asarray(observation_times, dtype=float),
        "mean_yield": float(np.mean(yields)),
        "yield_sd": float(np.std(yields, ddof=1)) if len(yields) > 1 else 0.0,
        "yield_mcse": mcse(yields),
        "mean_final_incidence": float(np.mean(finals)),
        "final_incidence_sd": float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0,
        "final_incidence_mcse": mcse(finals),
        "mean_incidence": incidence.mean(axis=0),
        "mean_vector_prevalence": vectors.mean(axis=0),
        "mean_incidence_susc": (np.nanmean(incidence_susc, axis=0) if not np.isnan(incidence_susc).all() else np.full(incidence_susc.shape[1], np.nan)),
        "mean_incidence_res": (np.nanmean(incidence_res, axis=0) if not np.isnan(incidence_res).all() else np.full(incidence_res.shape[1], np.nan)),
        "mean_event_count": float(np.mean(event_counts)),
        "runtime_seconds": time.time() - start,
        "kernel_scale": kernel.scale,
        "mean_step_distance": kernel.mean_step_distance,
    }
    if infected_count is not None:
        result["infection_probability"] = infected_count / len(yields)
    return result


# ------------------------- Spatial metrics -------------------------
MOORE_KERNEL = np.ones((3, 3), dtype=float)
MOORE_KERNEL[1, 1] = 0.0
PATCH_STRUCTURE = np.ones((3, 3), dtype=np.int8)


def hni(grid):
    same0 = convolve((grid == 0).astype(float), MOORE_KERNEL, mode="constant", cval=0.0)
    same1 = convolve((grid == 1).astype(float), MOORE_KERNEL, mode="constant", cval=0.0)
    degree = convolve(np.ones_like(grid, dtype=float), MOORE_KERNEL, mode="constant", cval=0.0)
    hetero = np.where(grid == 0, same1, same0)
    return float(np.mean(hetero / degree))


def amd(grid):
    d_to_1 = distance_transform_edt(grid == 0)
    d_to_0 = distance_transform_edt(grid == 1)
    values = np.where(grid == 0, d_to_1, d_to_0)
    return float(np.mean(values))


def local_diversity(grid, window=3):
    kernel = np.ones((window, window), dtype=float)
    n = convolve(np.ones_like(grid, dtype=float), kernel, mode="constant", cval=0.0)
    count1 = convolve((grid == 1).astype(float), kernel, mode="constant", cval=0.0)
    p1 = count1 / n
    p0 = 1.0 - p1
    with np.errstate(divide="ignore", invalid="ignore"):
        sh = -(np.where(p0 > 0, p0 * np.log(p0), 0.0) + np.where(p1 > 0, p1 * np.log(p1), 0.0))
    si = 1.0 - (p0 * p0 + p1 * p1)
    return float(np.mean(sh)), float(np.mean(si))


def ji_icm(grid):
    N = grid.shape[0]
    directed_counts = np.zeros((2, 2), dtype=float)
    total_pairs = 0
    heterotypic_pairs = 0
    shifts = ((0, 1), (1, -1), (1, 0), (1, 1))
    for dr, dc in shifts:
        r0a = max(0, -dr)
        r1a = min(N, N - dr)
        c0a = max(0, -dc)
        c1a = min(N, N - dc)
        a = grid[r0a:r1a, c0a:c1a]
        b = grid[r0a + dr:r1a + dr, c0a + dc:c1a + dc]
        total_pairs += a.size
        heterotypic_pairs += int(np.sum(a != b))
        for u in (0, 1):
            for v in (0, 1):
                directed_counts[u, v] += np.sum((a == u) & (b == v))
                directed_counts[v, u] += np.sum((b == u) & (a == v))
    ji = heterotypic_pairs / total_pairs
    p = directed_counts.ravel()
    p = p / p.sum()
    entropy_term = np.sum(np.where(p > 0, p * np.log(p), 0.0))
    contagion = 1.0 + entropy_term / np.log(4.0)
    return float(ji), float(contagion)


def b2a(grid):
    M = grid.size
    vertical = np.sum(grid[1:, :] != grid[:-1, :])
    horizontal = np.sum(grid[:, 1:] != grid[:, :-1])
    return float((vertical + horizontal) / M)


def patch_metrics(grid):
    M = grid.size
    n_patches = 0
    sizes = []
    for value in (0, 1):
        labelled, n = label(grid == value, structure=PATCH_STRUCTURE)
        n_patches += int(n)
        if n:
            counts = np.bincount(labelled.ravel())[1:]
            sizes.extend(counts.tolist())
    return float(n_patches / M), float(np.mean(sizes))


def morisita_horn(grid, block=5):
    N = grid.shape[0]
    if N % block != 0:
        raise ValueError("MH block must divide N")
    q = N // block
    reshaped = grid.reshape(q, block, q, block)
    count1 = reshaped.sum(axis=(1, 3)).astype(float).ravel()
    count0 = (block * block - count1).astype(float)
    A = count0.sum()
    B = count1.sum()
    DA = np.sum((count0 / A) ** 2)
    DB = np.sum((count1 / B) ** 2)
    return float(2.0 * np.sum(count0 * count1) / ((DA + DB) * A * B))


def coarse_transport_emd(grid, block=5, epsilon_blocks=1.0):
    """Entropically regularised transport distance on a fixed block aggregation."""
    N = grid.shape[0]
    if N % block != 0:
        raise ValueError("EMD block must divide N")
    q = N // block
    reshaped = grid.reshape(q, block, q, block)
    mass1 = reshaped.sum(axis=(1, 3)).astype(float)
    mass0 = block * block - mass1
    a = mass0 / mass0.sum()
    b = mass1 / mass1.sum()

    offsets = np.arange(-(q - 1), q, dtype=float)
    dr, dc = np.meshgrid(offsets, offsets, indexing="ij")
    distance_blocks = np.sqrt(dr * dr + dc * dc)
    G = np.exp(-distance_blocks / epsilon_blocks)
    G_cost = (distance_blocks * block) * G

    u = np.ones_like(a)
    v = np.ones_like(b)
    tiny = 1e-300
    for _ in range(1000):
        Kv = fftconvolve(v, G, mode="same")
        u_new = np.where(a > 0, a / np.maximum(Kv, tiny), 0.0)
        Ku = fftconvolve(u_new, G, mode="same")
        v_new = np.where(b > 0, b / np.maximum(Ku, tiny), 0.0)
        if np.max(np.abs(u_new - u)) < 1e-11 and np.max(np.abs(v_new - v)) < 1e-11:
            u = u_new
            v = v_new
            break
        u = u_new
        v = v_new
    return float(np.sum(u * fftconvolve(v, G_cost, mode="same")))


def ripley_cross_deviation(grid, rmax):
    N = grid.shape[0]
    M = N * N
    A = (grid == 0).astype(float)
    B = (grid == 1).astype(float)
    corr = fftconvolve(A, B[::-1, ::-1], mode="full")
    offsets = np.arange(-(N - 1), N)
    dr, dc = np.meshgrid(offsets, offsets, indexing="ij")
    distance = np.sqrt(dr * dr + dc * dc)
    overlap = (N - np.abs(dr)) * (N - np.abs(dc))
    centre = N - 1
    corr[centre, centre] = 0.0
    weighted = corr * (M / overlap)
    nA = float(np.sum(A))
    nB = float(np.sum(B))
    values = []
    for radius in np.arange(1, int(rmax) + 1, dtype=float):
        mask = (distance <= radius) & (distance > 0)
        Kobs = (M / (nA * nB)) * np.sum(weighted[mask])
        Kref = (M / (M - 1.0)) * int(np.sum(mask))
        values.append(abs(Kobs - Kref))
    return float(np.mean(values)) if values else 0.0


def compute_arrangement_metrics(grid, *, rmax, local_window=3, mh_block=5, emd_block=5, emd_epsilon_blocks=1.0):
    sh, si = local_diversity(grid, window=local_window)
    ji, icm = ji_icm(grid)
    frag, mpa = patch_metrics(grid)
    return {
        "EMD": coarse_transport_emd(grid, block=emd_block, epsilon_blocks=emd_epsilon_blocks),
        "HNI": hni(grid),
        "AMD": amd(grid),
        "MH": morisita_horn(grid, block=mh_block),
        "SH": sh,
        "SI": si,
        "JI": ji,
        "ICM": icm,
        "B2A": b2a(grid),
        "RK": ripley_cross_deviation(grid, rmax=rmax),
        "FRAG": frag,
        "MPA": mpa,
    }


def correlated_exact_grid(N: int, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(N, N))
    score = gaussian_filter(noise, sigma=float(sigma), mode="reflect") if sigma > 0 else noise
    order = np.argsort(score.ravel())
    labels = np.zeros(N * N, dtype=np.int8)
    labels[order[(N * N) // 2:]] = 1
    return labels.reshape(N, N)

@njit(cache=True)
def _local_hni_contribution(labels, N, idx):
    r = idx // N
    c = idx - r * N
    degree = 0
    hetero = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if rr < 0 or rr >= N or cc < 0 or cc >= N:
                continue
            degree += 1
            j = rr * N + cc
            if labels[j] != labels[idx]:
                hetero += 1
    return hetero / degree


@njit(cache=True)
def _hni_flat(labels, N):
    total = 0.0
    for i in range(labels.shape[0]):
        total += _local_hni_contribution(labels, N, i)
    return total / labels.shape[0]


@njit(cache=True)
def _target_hni_swap_anneal(initial, N, target, seed, max_iter=250_000, tolerance=5e-5):
    np.random.seed(seed)
    labels = initial.copy()
    n = labels.shape[0]
    current = _hni_flat(labels, N)
    best = current
    best_labels = labels.copy()
    temperature = 0.010
    cooling = 0.99996
    affected = np.empty(18, dtype=np.int64)
    for step in range(max_iter):
        if abs(best - target) <= tolerance:
            break
        i = np.random.randint(n)
        j = np.random.randint(n)
        while labels[j] == labels[i]:
            j = np.random.randint(n)
        count = 0
        for center in (i, j):
            cr = center // N
            cc = center - cr * N
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = cr + dr
                    c2 = cc + dc
                    if rr < 0 or rr >= N or c2 < 0 or c2 >= N:
                        continue
                    q = rr * N + c2
                    duplicate = False
                    for z in range(count):
                        if affected[z] == q:
                            duplicate = True
                            break
                    if not duplicate:
                        affected[count] = q
                        count += 1
        before = 0.0
        for z in range(count):
            before += _local_hni_contribution(labels, N, affected[z])
        tmp = labels[i]
        labels[i] = labels[j]
        labels[j] = tmp
        after = 0.0
        for z in range(count):
            after += _local_hni_contribution(labels, N, affected[z])
        proposed = current + (after - before) / n
        delta = abs(proposed - target) - abs(current - target)
        accept = (delta <= 0.0) or (np.random.random() < math.exp(-delta / max(temperature, 1e-12)))
        if accept:
            current = proposed
            if abs(current - target) < abs(best - target):
                best = current
                best_labels = labels.copy()
        else:
            tmp = labels[i]
            labels[i] = labels[j]
            labels[j] = tmp
        temperature *= cooling
    if abs(current - target) < abs(best - target):
        best = current
        best_labels = labels.copy()
    return best_labels, best


def generate_hni_target_ensemble(*, N, n_targets=25, replicates=8, seed=20260824, tolerance=5e-4):
    """Generate exact 50:50 binary arrangements spanning the HNI mixing axis.

    The generator uses only spatial labels, never epidemic outcomes. N must be even.
    Returns ``(grids, manifest_dataframe)``.
    """
    import pandas as pd
    if N % 2:
        raise ValueError("N must be even for an exact 50:50 ensemble.")
    M = N * N
    rr, cc = np.indices((N, N))
    checker = ((rr + cc) % 2).astype(np.int8)
    half = (cc >= N // 2).astype(np.int8)
    low = float(hni(half))
    high = float(hni(checker))
    targets = np.linspace(low, high, n_targets + 2)[1:-1]
    grids = {}
    rows = []
    for ti, target in enumerate(targets):
        for rep in range(replicates):
            local_seed = int(seed + ti * 100 + rep)
            rng = np.random.default_rng(local_seed)
            midpoint = 0.5 * (low + high)
            if target <= midpoint:
                flat = half.copy().ravel()
                frac = (target - low) / max(midpoint - low, 1e-12)
            else:
                flat = checker.copy().ravel()
                frac = (high - target) / max(high - midpoint, 1e-12)
            n_perturb = max(0, int(4 * N * frac))
            zeros = np.where(flat == 0)[0]
            ones = np.where(flat == 1)[0]
            z = rng.choice(zeros, size=n_perturb, replace=True)
            o = rng.choice(ones, size=n_perturb, replace=True)
            for a, b in zip(z, o):
                flat[a], flat[b] = flat[b], flat[a]
            labels, achieved = _target_hni_swap_anneal(flat.astype(np.int8), N, float(target), local_seed + 17)
            if abs(float(achieved) - float(target)) > tolerance:
                raise RuntimeError(
                    f"Could not reach HNI target {target:.6f}; achieved {achieved:.6f}."
                )
            grid = labels.reshape(N, N)
            if np.sum(grid == 0) != M // 2 or np.sum(grid == 1) != M // 2:
                raise RuntimeError("Exact 50:50 count was not preserved.")
            did = f"HNI_t{ti:02d}_r{rep:02d}"
            grids[did] = grid
            rows.append(
                dict(
                    design_id=did,
                    target_hni=float(target),
                    achieved_hni=float(achieved),
                    seed=local_seed,
                )
            )
    return grids, pd.DataFrame(rows)
