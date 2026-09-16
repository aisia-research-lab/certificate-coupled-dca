from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np

Array = np.ndarray

@dataclass
class TRSInstance:
    A: Array
    b: Array
    r: float
    M: float
    x0: Array
    meta: dict
    Q: Array | None = None
    eigvals: Array | None = None

    @property
    def n(self) -> int:
        return int(self.b.size)

    def q(self, x: Array) -> float:
        return float(0.5 * x @ self.A @ x + self.b @ x)

    def g(self, x: Array) -> Array:
        return self.A @ x + self.b

    def lambda_x(self, x: Array, boundary_tol: float = 1e-10) -> float:
        nx = float(np.linalg.norm(x))
        if nx < self.r * (1.0 - boundary_tol):
            return 0.0
        # Algorithmic iterates should be feasible; clip tiny numerical drift.
        return max(0.0, -float(x @ self.g(x)) / (self.r * self.r))

    def e(self, x: Array) -> Array:
        lam = self.lambda_x(x)
        return self.g(x) + lam * x

    def S(self, x: Array) -> float:
        return float(np.linalg.norm(self.e(x)))

    def B(self, x: Array) -> Array:
        return self.A + self.lambda_x(x) * np.eye(self.n)

    def project(self, y: Array) -> Array:
        ny = float(np.linalg.norm(y))
        if ny <= self.r:
            return y.copy()
        return (self.r / ny) * y

    def dca_step(self, x: Array, rho: float) -> Array:
        return self.project(x - self.g(x) / rho)


def _random_orthogonal(n: int, rng: np.random.Generator) -> Array:
    H = rng.normal(size=(n, n))
    Q, R = np.linalg.qr(H)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    return Q * signs


def make_controlled_instance(
    n: int = 80,
    seed: int = 0,
    r: float = 1.0,
    lambda1: float = -5.0,
    lambda2: float = -3.0,
    positive_min: float = 0.5,
    positive_max: float = 4.0,
    initial_mix: float = 0.35,
    random_rotate: bool = True,
    multiplicity: int = 1,
    extra_negative: int = 0,
) -> TRSInstance:
    """Controlled family with a nonglobal DCA attractor.

    With b=0, x=e_2 is stationary on the sphere with multiplier -lambda2.
    If lambda1 < lambda2, it is nonglobal and has shifted negative curvature
    along e_1. Starting in span{e_2,e_3} leaves the ground-state direction
    absent, so DCA approaches the nonglobal e_2 state while a structural
    certificate can detect e_1.
    """
    if n < max(4, multiplicity + 3 + extra_negative):
        raise ValueError("n too small for requested spectrum")
    rng = np.random.default_rng(seed)
    eig = np.linspace(positive_min, positive_max, n)
    eig[:multiplicity] = lambda1
    idx2 = multiplicity
    eig[idx2] = lambda2
    # Optional additional negative eigenvalues above lambda2 in magnitude.
    for j in range(extra_negative):
        eig[idx2 + 1 + j] = lambda2 + (j + 1) * (0.7 * abs(lambda2) / (extra_negative + 1))
    eig = np.sort(eig)
    # Sorting may shift semantic positions; identify one lambda2 index and first positive.
    idx2 = int(np.argmin(np.abs(eig - lambda2)))
    pos_idx = int(np.flatnonzero(eig > 0)[0])
    Q = _random_orthogonal(n, rng) if random_rotate else np.eye(n)
    A = (Q * eig) @ Q.T
    A = 0.5 * (A + A.T)
    b = np.zeros(n)
    v2 = Q[:, idx2]
    vp = Q[:, pos_idx]
    mix = float(initial_mix)
    x0 = r * (np.sqrt(max(0.0, 1.0 - mix * mix)) * v2 + mix * vp)
    x0 /= np.linalg.norm(x0) / r
    M = float(np.max(np.abs(eig)))
    meta = {
        "family": "controlled_nonglobal",
        "seed": seed,
        "lambda1": float(eig[0]),
        "lambda2_target": lambda2,
        "lambda2_index": idx2,
        "positive_index": pos_idx,
        "multiplicity": multiplicity,
        "extra_negative": extra_negative,
        "initial_mix": mix,
        "random_rotate": random_rotate,
    }
    return TRSInstance(A=A, b=b, r=r, M=M, x0=x0, meta=meta, Q=Q, eigvals=eig)


def make_random_spectral_instance(
    n: int,
    seed: int,
    r: float = 1.0,
    lambda_min: float = -5.0,
    lambda_max: float = 4.0,
    negative_fraction: float = 0.2,
    b_scale: float = 0.3,
) -> TRSInstance:
    rng = np.random.default_rng(seed)
    Q = _random_orthogonal(n, rng)
    kneg = max(1, int(round(n * negative_fraction)))
    neg = np.linspace(lambda_min, -0.25, kneg)
    pos = np.linspace(0.25, lambda_max, n - kneg)
    eig = np.concatenate([neg, pos])
    A = (Q * eig) @ Q.T
    A = 0.5 * (A + A.T)
    b = b_scale * rng.normal(size=n) / np.sqrt(n)
    x0 = rng.normal(size=n)
    x0 = r * x0 / max(np.linalg.norm(x0), 1e-15)
    M = float(np.max(np.abs(eig)))
    meta = {
        "family": "random_spectral",
        "seed": seed,
        "negative_fraction": negative_fraction,
        "b_scale": b_scale,
    }
    return TRSInstance(A=A, b=b, r=r, M=M, x0=x0, meta=meta, Q=Q, eigvals=eig)
