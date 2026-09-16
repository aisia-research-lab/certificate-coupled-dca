from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .problem import Array

@dataclass
class RitzPair:
    theta: float
    v: Array

class PersistentKrylov:
    """Fully reorthogonalized symmetric Krylov process with explicit Av count.

    This is an experiment engine, not a production Lanczos implementation.
    Full reorthogonalization improves reproducibility and makes Ritz extraction
    robust for the dimensions used in the paper experiments.
    """
    def __init__(self, A: Array, q1: Array, breakdown_tol: float = 1e-13):
        self.A = A
        self.n = A.shape[0]
        q = np.asarray(q1, dtype=float).copy()
        nq = np.linalg.norm(q)
        if nq == 0:
            raise ValueError("q1 must be nonzero")
        q /= nq
        self.Q = [q]
        self.AQ = [self._matvec(q)]
        self.n_av = 1
        self.breakdown_tol = breakdown_tol
        self.breakdown = False

    def _matvec(self, q: Array) -> Array:
        return self.A @ q

    @property
    def dim(self) -> int:
        return len(self.Q)

    def extend_one(self) -> bool:
        if self.breakdown or self.dim >= self.n:
            self.breakdown = True
            return False
        # Krylov successor generated from A q_j, already cached.
        w = self.AQ[-1].copy()
        # Two-pass modified Gram-Schmidt for reproducibility.
        for _ in range(2):
            for q in self.Q:
                w -= q * float(q @ w)
        nw = float(np.linalg.norm(w))
        if nw <= self.breakdown_tol:
            self.breakdown = True
            return False
        qnew = w / nw
        self.Q.append(qnew)
        self.AQ.append(self._matvec(qnew))
        self.n_av += 1
        return True

    def extend_to(self, target_dim: int) -> int:
        target_dim = min(int(target_dim), self.n)
        while self.dim < target_dim and self.extend_one():
            pass
        return self.dim

    def projected_matrix(self) -> Array:
        Q = np.column_stack(self.Q)
        AQ = np.column_stack(self.AQ)
        T = Q.T @ AQ
        return 0.5 * (T + T.T)

    def extreme_ritz(self) -> tuple[RitzPair, RitzPair]:
        Q = np.column_stack(self.Q)
        T = self.projected_matrix()
        vals, vecs = np.linalg.eigh(T)
        vlo = Q @ vecs[:, 0]
        vhi = Q @ vecs[:, -1]
        return RitzPair(float(vals[0]), vlo), RitzPair(float(vals[-1]), vhi)
