from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .methods import RunResult
from .problem import TRSInstance
from .reference import TRSReferenceSolution

def _jsonable(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)): return obj.item()
    raise TypeError(type(obj).__name__)

def result_summary(inst: TRSInstance, ref: TRSReferenceSolution, res: RunResult) -> dict:
    return {
        "family": inst.meta.get("family"), "instance_seed": inst.meta.get("seed"), "n":inst.n,
        "method":res.method, "status":res.status, "q":res.q, "qstar":ref.qstar,
        "objective_gap":res.q-ref.qstar, "S":res.S, "N_DCA":res.N_DCA,
        "N_E":res.N_E, "N_Av":res.N_Av, "stages_completed":res.stages_completed,
        "first_escape_dca":res.first_escape_dca, "runtime_sec":res.runtime_sec,
    }

def save_run(outdir: str|Path, tag: str, inst: TRSInstance, ref: TRSReferenceSolution, res: RunResult):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    summary = result_summary(inst, ref, res)
    (out/f"{tag}_summary.json").write_text(json.dumps(summary, indent=2, default=_jsonable))
    pd.DataFrame(res.trace).to_csv(out/f"{tag}_trace.csv", index=False)
    pd.DataFrame(res.corrections).to_csv(out/f"{tag}_corrections.csv", index=False)
    pd.DataFrame(res.stage_outputs).to_csv(out/f"{tag}_stages.csv", index=False)
    return summary
