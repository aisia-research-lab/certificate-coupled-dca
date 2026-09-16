from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .problem import make_controlled_instance
from .reference import solve_trs_global
from .methods import run_plain_dca, run_post_dca_globalization, run_cc_dca
from .logging_utils import save_run
from .correction import structural_correction

@dataclass
class PilotConfig:
    dimensions: tuple[int,...] = (40, 80, 120)
    seeds: tuple[int,...] = (0, 1, 2)
    delta0: float = 1.0
    beta: float = 0.5
    n_stages: int = 3
    p_total: float = 0.10
    rho_min: float = 1e-3
    initial_mix: float = 0.35
    max_dca: int = 50000


def run_smoke(outdir: str|Path) -> dict:
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    inst = make_controlled_instance(n=30, seed=7, initial_mix=0.25)
    ref = solve_trs_global(inst)
    checks = {
        "ref_feasibility": ref.feasibility,
        "ref_stationarity": ref.stationarity,
        "ref_complementarity": ref.complementarity,
        "ref_min_shifted_eig": ref.min_shifted_eig,
    }
    # Exact nonglobal stationary vector (second negative eigenvector) gives hard geometry.
    eigvals, Q = np.linalg.eigh(inst.A)
    v = Q[:, 0]
    x = Q[:, 1] * inst.r
    lamx = inst.lambda_x(x)
    curvature = float(v @ (inst.A + lamx*np.eye(inst.n)) @ v)
    kappa = -curvature
    corr = structural_correction(inst, x, v, kappa)
    checks.update({
        "hard_case_alignment": corr.alignment,
        "hard_case_decrease_ratio": corr.ratio,
        "hard_case_decrease": corr.decrease,
        "hard_case_predicted": corr.predicted_delta,
    })
    res = run_cc_dca(inst, delta0=1.0, beta=0.5, n_stages=2, p_total=0.1,
                     spectral_seed=19, persistent=True, max_dca=20000)
    checks.update({"cc_status":res.status, "cc_N_DCA":res.N_DCA, "cc_N_E":res.N_E,
                   "cc_N_Av":res.N_Av, "cc_stages":res.stages_completed})
    (out/"smoke_results.json").write_text(json.dumps(checks, indent=2))
    return checks


def run_pilot(outdir: str|Path, cfg: PilotConfig = PilotConfig()) -> pd.DataFrame:
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    final_delta = cfg.delta0 * cfg.beta**(cfg.n_stages-1)
    final_eps = final_delta
    for n in cfg.dimensions:
        for seed in cfg.seeds:
            inst = make_controlled_instance(n=n, seed=seed, initial_mix=cfg.initial_mix)
            ref = solve_trs_global(inst)
            spectral_seed = 100000 + seed
            methods = [
                run_plain_dca(inst, eps=final_eps, rho=inst.M, max_dca=cfg.max_dca),
                run_post_dca_globalization(inst, eps=final_eps, delta=final_delta,
                                           p=cfg.p_total, rho=inst.M,
                                           spectral_seed=spectral_seed, max_dca=cfg.max_dca),
                run_cc_dca(inst, cfg.delta0, cfg.beta, cfg.n_stages, cfg.p_total,
                           cfg.rho_min, spectral_seed, persistent=False,
                           max_dca=cfg.max_dca),
                run_cc_dca(inst, cfg.delta0, cfg.beta, cfg.n_stages, cfg.p_total,
                           cfg.rho_min, spectral_seed, persistent=True,
                           max_dca=cfg.max_dca),
            ]
            for res in methods:
                tag = f"n{n}_seed{seed}_{res.method}"
                row = save_run(out, tag, inst, ref, res)
                row.update({"delta0":cfg.delta0,"beta":cfg.beta,"n_stages":cfg.n_stages,
                            "final_delta":final_delta,"p_total":cfg.p_total})
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out/"pilot_summary.csv", index=False)
    (out/"pilot_config.json").write_text(json.dumps(cfg.__dict__, indent=2))
    return df


def summarize_pilot(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["method","n","N_DCA","N_E","N_Av","objective_gap","runtime_sec"]
    agg = (df[cols].groupby(["method","n"])
           .agg(["median","min","max"]).reset_index())
    return agg
