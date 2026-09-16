from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq
from .problem import TRSInstance, Array

@dataclass
class TRSReferenceSolution:
    x: Array
    lam: float
    qstar: float
    feasibility: float
    stationarity: float
    complementarity: float
    min_shifted_eig: float
    hard_case: bool


def solve_trs_global(inst: TRSInstance, tol: float = 1e-12) -> TRSReferenceSolution:
    """Independent dense reference solver via eigendecomposition/secular equation.

    Handles the classical hard case by a pseudoinverse solution plus a component
    in the minimum-eigenspace.
    """
    A, b, r = inst.A, inst.b, inst.r
    d, Q = np.linalg.eigh(A)
    c = Q.T @ b
    dmin = float(d[0])
    lam_lb = max(0.0, -dmin)
    denom0 = d + lam_lb
    singular = np.abs(denom0) <= max(tol, 1e-13 * max(1.0, inst.M))
    regular = ~singular

    def x_of(lam: float) -> Array:
        den = d + lam
        return Q @ (-c / den)

    hard_case = False
    # Interior solution only possible for PSD A with compatible nullspace.
    if lam_lb == 0.0:
        if np.any(singular & (np.abs(c) > 1e-10)):
            # secular norm diverges at 0; boundary root is strictly positive
            pass
        else:
            y = np.zeros_like(c)
            y[regular] = -c[regular] / denom0[regular]
            if np.linalg.norm(y) <= r + 1e-11:
                # If nullspace exists, any null component preserves objective;
                # choose minimum-norm solution, which is feasible.
                x = Q @ y
                lam = 0.0
                return _certify(inst, x, lam, hard_case=False)

    # At positive lower bound, inspect the hard case.
    if lam_lb > 0.0 and not np.any(singular & (np.abs(c) > 1e-10)):
        y0 = np.zeros_like(c)
        y0[regular] = -c[regular] / denom0[regular]
        n0 = float(np.linalg.norm(y0))
        if n0 <= r + 1e-11:
            hard_case = True
            # Add along one minimum eigendirection to hit the boundary.
            idx = int(np.flatnonzero(singular)[0])
            y = y0.copy()
            y[idx] = np.sqrt(max(0.0, r * r - n0 * n0))
            x = Q @ y
            return _certify(inst, x, lam_lb, hard_case=True)

    # Otherwise solve the secular equation strictly above the lower bound.
    eps = max(1e-12, 1e-12 * max(1.0, abs(lam_lb), inst.M))
    lo = lam_lb + eps

    def phi(lam: float) -> float:
        return float(np.linalg.norm(x_of(lam)) - r)

    # phi(lo) should be positive; if numerical cancellation makes it nonpositive,
    # approach the pole more closely before declaring failure.
    for scale in [1e-14, 1e-15, 1e-16]:
        if phi(lo) > 0:
            break
        lo = lam_lb + max(np.finfo(float).eps * max(1.0, abs(lam_lb)), scale)
    if phi(lo) <= 0:
        # This can only be a numerically unresolved hard case.
        y0 = np.zeros_like(c)
        good = np.abs(denom0) > 1e-10
        y0[good] = -c[good] / denom0[good]
        n0 = float(np.linalg.norm(y0))
        if lam_lb > 0 and n0 <= r + 1e-8:
            idx = int(np.argmin(np.abs(denom0)))
            y0[idx] = np.sqrt(max(0.0, r * r - n0 * n0))
            return _certify(inst, Q @ y0, lam_lb, hard_case=True)
        raise RuntimeError("Could not bracket TRS secular root at lower endpoint")

    hi = max(1.0, lam_lb + 1.0, np.linalg.norm(b) / max(r, 1e-15) + inst.M + 1.0)
    while phi(hi) > 0:
        hi *= 2.0
        if hi > 1e16:
            raise RuntimeError("Could not bracket TRS secular root")
    lam = float(brentq(phi, lo, hi, xtol=tol, rtol=1e-13, maxiter=500))
    x = x_of(lam)
    return _certify(inst, x, lam, hard_case=hard_case)


def _certify(inst: TRSInstance, x: Array, lam: float, hard_case: bool) -> TRSReferenceSolution:
    feasibility = max(0.0, float(np.linalg.norm(x) - inst.r))
    stationarity = float(np.linalg.norm(inst.A @ x + inst.b + lam * x))
    complementarity = abs(float(lam * (np.linalg.norm(x) - inst.r)))
    min_shifted = float(np.linalg.eigvalsh(inst.A + lam * np.eye(inst.n))[0])
    return TRSReferenceSolution(
        x=x,
        lam=float(lam),
        qstar=inst.q(x),
        feasibility=feasibility,
        stationarity=stationarity,
        complementarity=complementarity,
        min_shifted_eig=min_shifted,
        hard_case=hard_case,
    )
