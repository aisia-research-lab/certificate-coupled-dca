from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .problem import TRSInstance, Array

@dataclass
class CorrectionResult:
    x_plus: Array
    decrease: float
    predicted_delta: float
    ratio: float
    case: str
    alpha: float
    alignment: float
    curvature: float
    step_norm: float


def alpha_kappa(inst: TRSInstance, kappa: float) -> float:
    Lbar = 2.0 * inst.M + np.linalg.norm(inst.b) / inst.r
    return min(1.0 / 16.0, kappa / (32.0 * Lbar))


def farther_sphere_intersection(inst: TRSInstance, x: Array, w: Array) -> Array:
    ww = float(w @ w)
    xw = float(x @ w)
    h2 = max(0.0, inst.r * inst.r - float(x @ x))
    disc = max(0.0, xw * xw + ww * h2)
    root = np.sqrt(disc)
    t1 = (-xw + root) / ww
    t2 = (-xw - root) / ww
    t = t1 if abs(t1) >= abs(t2) else t2
    return t * w


def structural_correction(inst: TRSInstance, x: Array, v: Array, kappa: float) -> CorrectionResult:
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v)
    alpha = alpha_kappa(inst, kappa)
    h = np.sqrt(max(0.0, inst.r * inst.r - float(x @ x)))
    c = float(x @ v) / inst.r
    if h >= alpha * inst.r:
        w = v
        case = "I-slack"
    elif abs(c) >= alpha:
        w = v
        case = "II-aligned"
    else:
        sgn = 1.0 if c >= 0 else -1.0
        tau = 2.0 * alpha * sgn
        w = v + tau * x / inst.r
        case = "III-hard"
    d = farther_sphere_intersection(inst, x, w)
    xp = x + d
    # normalize tiny floating error exactly to sphere if needed
    nxp = np.linalg.norm(xp)
    if nxp > inst.r * (1 + 1e-12):
        xp = inst.r * xp / nxp
    dec = inst.q(x) - inst.q(xp)
    pred = (4.0 / 27.0) * kappa * alpha * alpha * inst.r * inst.r
    curv = float(v @ inst.B(x) @ v)
    return CorrectionResult(
        x_plus=xp,
        decrease=float(dec),
        predicted_delta=float(pred),
        ratio=float(dec / pred) if pred > 0 else np.inf,
        case=case,
        alpha=float(alpha),
        alignment=abs(c),
        curvature=curv,
        step_norm=float(np.linalg.norm(d)),
    )


def best_circle_correction(inst: TRSInstance, x: Array, v: Array, grid: int = 2048) -> Array:
    """Strong post-convergence correction in span{x,v} on the trust-region sphere.

    Used only for the post-DCA globalization baseline to avoid making that baseline
    artificially weak.  It globally minimizes the objective over the unit circle in
    span{x,v} by a dense periodic angular search followed by local refinement.
    """
    from scipy.optimize import minimize_scalar
    nx = float(np.linalg.norm(x))
    if nx < 1e-14:
        return structural_correction(inst, x, v, max(1e-12, -float(v @ inst.B(x) @ v))).x_plus
    u1 = x / nx
    vt = v - u1 * float(u1 @ v)
    nvt = float(np.linalg.norm(vt))
    if nvt < 1e-12:
        # Degenerate span; fall back to the theorem correction.
        kappa = max(1e-12, -float(v @ inst.B(x) @ v))
        return structural_correction(inst, x, v, kappa).x_plus
    u2 = vt / nvt
    U = np.column_stack([u1, u2])
    C = U.T @ inst.A @ U
    db = U.T @ inst.b
    def f(th):
        z = np.array([np.cos(th), np.sin(th)])
        return 0.5 * inst.r * inst.r * float(z @ C @ z) + inst.r * float(db @ z)
    ths = np.linspace(-np.pi, np.pi, grid, endpoint=False)
    vals = np.array([f(t) for t in ths])
    i = int(np.argmin(vals))
    step = 2*np.pi/grid
    center = ths[i]
    res = minimize_scalar(f, bounds=(center-step, center+step), method='bounded',
                          options={'xatol':1e-14})
    th = float(res.x)
    z = np.array([np.cos(th), np.sin(th)])
    return inst.r * (U @ z)
