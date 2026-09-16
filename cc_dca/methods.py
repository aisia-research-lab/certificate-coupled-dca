from __future__ import annotations
from dataclasses import dataclass, field
import math, time
import numpy as np
from .problem import TRSInstance, Array
from .krylov import PersistentKrylov
from .correction import structural_correction, best_circle_correction

@dataclass
class RunResult:
    method: str
    x: Array
    status: str
    q: float
    S: float
    N_DCA: int
    N_E: int
    N_Av: int
    stages_completed: int
    first_escape_dca: int | None
    trace: list[dict] = field(default_factory=list)
    corrections: list[dict] = field(default_factory=list)
    stage_outputs: list[dict] = field(default_factory=list)
    runtime_sec: float = 0.0


def _J_delta(n: int, M: float, delta: float, p: float) -> int:
    val = 1 + math.ceil(0.5 * math.log(11.0 * n / (p * p)) * math.sqrt(2.0 * M / delta))
    return min(n, val)


def _Lbar(inst: TRSInstance) -> float:
    return 2.0 * inst.M + np.linalg.norm(inst.b) / inst.r


def _alpha_delta(inst: TRSInstance, delta: float) -> float:
    return min(1.0 / 16.0, 3.0 * delta / (128.0 * _Lbar(inst)))


def _tau(inst: TRSInstance, eps: float, delta: float) -> float:
    a = _alpha_delta(inst, delta)
    return min(eps, delta * a * inst.r / 9.0)


def _rng_vec(n: int, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=n)
    return q / np.linalg.norm(q)


def run_plain_dca(inst: TRSInstance, eps: float, rho: float | None = None,
                  max_dca: int = 200000, log_every: int = 1) -> RunResult:
    t0 = time.perf_counter()
    rho = float(max(1e-12, inst.M if rho is None else rho))
    x = inst.x0.copy()
    trace = []
    k = 0
    while inst.S(x) > eps and k < max_dca:
        if k % log_every == 0:
            trace.append({"event":"dca", "k":k, "q":inst.q(x), "S":inst.S(x)})
        x = inst.dca_step(x, rho)
        k += 1
    status = "FIRST_ORDER" if inst.S(x) <= eps else "MAX_DCA"
    return RunResult("plain_dca", x, status, inst.q(x), inst.S(x), k, 0, 0, 0, None,
                     trace=trace, runtime_sec=time.perf_counter()-t0)


def run_post_dca_globalization(inst: TRSInstance, eps: float, delta: float, p: float,
                               rho: float | None = None, spectral_seed: int = 1234,
                               max_dca: int = 200000, max_escapes: int = 1000) -> RunResult:
    """Post-convergence structural check + correction/restart baseline.

    Spectral state is deliberately cold-started after each correction.
    """
    t0 = time.perf_counter()
    rho = float(inst.M if rho is None else rho)
    x = inst.x0.copy()
    N_DCA = N_E = N_Av = 0
    first_escape = None
    trace, corrections = [], []
    restart = 0
    J = _J_delta(inst.n, inst.M, delta, p)
    while N_DCA < max_dca and N_E <= max_escapes:
        while inst.S(x) > eps and N_DCA < max_dca:
            trace.append({"event":"dca", "k":N_DCA, "q":inst.q(x), "S":inst.S(x)})
            x = inst.dca_step(x, rho)
            N_DCA += 1
        if inst.S(x) > eps:
            break
        K = PersistentKrylov(inst.A, _rng_vec(inst.n, spectral_seed + 10007 * restart))
        K.extend_to(J)
        N_Av += K.n_av
        low, _ = K.extreme_ritz()
        shifted = low.theta + inst.lambda_x(x)
        trace.append({"event":"post_spectral", "k":N_DCA, "q":inst.q(x), "S":inst.S(x),
                      "j":K.dim, "theta_minus":low.theta, "shifted":shifted})
        if shifted <= -3.0 * delta / 4.0:
            kappa = -(shifted)
            # Strong post-convergence correction: minimize q over the sphere circle
            # spanned by the current point and the detected negative-curvature Ritz vector.
            # This makes the baseline deliberately competitive rather than a strawman.
            x_old = x.copy()
            x = best_circle_correction(inst, x, low.v)
            if first_escape is None:
                first_escape = N_DCA
            N_E += 1
            corrections.append({
                "case":"post_circle", "decrease":inst.q(x_old)-inst.q(x),
                "alignment":abs(float(x_old @ low.v))/inst.r,
                "curvature":float(low.v @ inst.B(x_old) @ low.v),
                "step_norm":float(np.linalg.norm(x-x_old)),
            })
            restart += 1
            continue
        status = "GREEN_POST" if K.dim >= J else "KRYLOV_BREAKDOWN"
        return RunResult("post_dca", x, status, inst.q(x), inst.S(x), N_DCA, N_E, N_Av,
                         1, first_escape, trace, corrections, runtime_sec=time.perf_counter()-t0)
    status = "MAX_WORK"
    return RunResult("post_dca", x, status, inst.q(x), inst.S(x), N_DCA, N_E, N_Av,
                     0, first_escape, trace, corrections, runtime_sec=time.perf_counter()-t0)


def run_cc_dca(inst: TRSInstance, delta0: float, beta: float, n_stages: int, p_total: float,
               rho_min: float = 1e-3, spectral_seed: int = 1234,
               persistent: bool = True, max_dca: int = 200000,
               max_escapes: int = 10000) -> RunResult:
    t0 = time.perf_counter()
    x = inst.x0.copy()
    U = inst.M
    rho_bar = max(rho_min, inst.M)
    N_DCA = N_E = N_Av = 0
    first_escape = None
    trace, corrections, stage_outputs = [], [], []
    K = None
    counted_av = 0
    if persistent:
        K = PersistentKrylov(inst.A, _rng_vec(inst.n, spectral_seed))
        counted_av = K.n_av
        N_Av = counted_av
    status = "RUNNING"
    stages_completed = 0

    for m in range(n_stages):
        delta = delta0 * (beta ** m)
        eps = inst.r * delta
        pm = p_total / (2.0 ** (m + 1))
        J = _J_delta(inst.n, inst.M, delta, pm)
        tau = _tau(inst, eps, delta)
        rho = max(rho_min, U)
        # Safety follows from the certified/stored upper bound U; avoid a dense
        # eigendecomposition inside the timed algorithm. Offline verification
        # checks this invariant independently.
        if rho + 1e-12 < -inst.M:
            raise RuntimeError("Invalid rho")
        lock_index = 0
        stage_done = False
        while not stage_done:
            while inst.S(x) > tau:
                if N_DCA >= max_dca:
                    status = "MAX_DCA"
                    return RunResult("persistent_cc_dca" if persistent else "cold_cc_dca", x, status,
                                     inst.q(x), inst.S(x), N_DCA, N_E, N_Av, stages_completed,
                                     first_escape, trace, corrections, stage_outputs,
                                     time.perf_counter()-t0)
                trace.append({"event":"dca", "stage":m, "k":N_DCA, "q":inst.q(x), "S":inst.S(x),
                              "rho":rho, "tau":tau})
                x = inst.dca_step(x, rho)
                N_DCA += 1
            # Certificate lock.
            if not persistent:
                K = PersistentKrylov(inst.A, _rng_vec(inst.n, spectral_seed + 1000003*m + 10007*lock_index))
                N_Av += K.n_av
                lock_index += 1
            assert K is not None
            while True:
                low, high = K.extreme_ritz()
                shifted = low.theta + inst.lambda_x(x)
                trace.append({"event":"lock", "stage":m, "k":N_DCA, "q":inst.q(x), "S":inst.S(x),
                              "j":K.dim, "theta_minus":low.theta, "theta_plus":high.theta,
                              "lambda_x":inst.lambda_x(x), "shifted":shifted, "J":J})
                if shifted <= -3.0 * delta / 4.0:
                    kappa = -shifted
                    corr = structural_correction(inst, x, low.v, kappa)
                    # The theorem threshold should hold because x is locked at tau.
                    x = corr.x_plus
                    N_E += 1
                    if first_escape is None:
                        first_escape = N_DCA
                    corrections.append({"stage":m, "k":N_DCA, **vars(corr)})
                    if N_E >= max_escapes:
                        status = "MAX_ESCAPES"
                        return RunResult("persistent_cc_dca" if persistent else "cold_cc_dca", x, status,
                                         inst.q(x), inst.S(x), N_DCA, N_E, N_Av, stages_completed,
                                         first_escape, trace, corrections, stage_outputs,
                                         time.perf_counter()-t0)
                    break  # release lock -> primal dynamics
                if K.dim >= J or K.breakdown:
                    if K.dim < J and K.breakdown:
                        # Exact invariant subspace reached; Ritz extremes of reachable Krylov space may
                        # miss eigenvalues if q1 has zero components. Random starts make this measure-zero;
                        # flag rather than silently certify.
                        status = "KRYLOV_BREAKDOWN"
                        return RunResult("persistent_cc_dca" if persistent else "cold_cc_dca", x, status,
                                         inst.q(x), inst.S(x), N_DCA, N_E, N_Av, stages_completed,
                                         first_escape, trace, corrections, stage_outputs,
                                         time.perf_counter()-t0)
                    # GREEN at full prescribed depth.
                    stage_done = True
                    stages_completed += 1
                    stage_outputs.append({"stage":m, "delta":delta, "eps":eps, "p":pm, "J":J,
                                          "q":inst.q(x), "S":inst.S(x), "lambda_x":inst.lambda_x(x),
                                          "theta_minus":low.theta, "theta_plus":high.theta,
                                          "N_DCA":N_DCA, "N_E":N_E, "N_Av":N_Av})
                    # Completed upper-edge certificate calibrates next stage.
                    U = min(U, high.theta + delta / 4.0)
                    break
                before = K.n_av
                K.extend_one()
                N_Av += K.n_av - before
            # continue stage if correction happened
        status = "GREEN"

    method = "persistent_cc_dca" if persistent else "cold_cc_dca"
    return RunResult(method, x, status, inst.q(x), inst.S(x), N_DCA, N_E, N_Av,
                     stages_completed, first_escape, trace, corrections, stage_outputs,
                     time.perf_counter()-t0)
