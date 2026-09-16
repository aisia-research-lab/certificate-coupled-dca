# Certificate-Coupled DCA for the Trust-Region Subproblem

Reproducibility code and archived experimental results for the manuscript (under a submission):

> **Certificate-Coupled DCA for the Trust-Region Subproblem: Early Structural Escape and Progressive Global Certification**  
> Author: Nguyen Thanh Binh
> (University of Science, Vietnam National University Ho Chi Minh City, Vietnam). Email: ngtbinh@hcmus.edu.vn

This repository contains the implementation, experiment configuration, and archived numerical results for Experiments E1-E5 in the manuscript. The scientific source files under `cc_dca/` are the exact sources associated with the archived results.

## Repository contents

- `cc_dca/` — scientific implementation.
- `run_experiments.py` — runner for Experiments E1--E5.
- `experiment_config.json` — experimental configuration used by the runner.
- `reference_results/` — archived CSV results, checkpoints, summaries, figures, and provenance manifest.
- `RESULTS_SUMMARY.md` — numerical results corresponding to the manuscript.
- `REPRODUCIBILITY.md` — environment and rerun instructions.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_experiments.py --exp E2
```

E2 is a convenient short verification run. To reproduce all five experiment families:

```bash
python run_experiments.py --exp all
```

Fresh outputs are written to `results/reproduction/`; the archived results in `reference_results/` are never overwritten.

## Experiment sizes

| Experiment | Rows | Purpose |
|---|---:|---|
| E1 | 320 | Timing of the first negative-curvature correction |
| E2 | 120 | Near-orthogonal quantitative decrease test |
| E3 | 80 | Effect of Krylov persistence |
| E4 | 160 | Progressive versus direct final-tolerance certification |
| E5 | 620 | Spectral sensitivity and generic random instances |
| **Total** | **1300** | |

The archived `reference_results/run_manifest.json` records the original environment metadata and SHA-256 hashes of the scientific source files. Runtime measurements are machine-dependent; deterministic numerical quantities should reproduce up to ordinary floating-point tolerance under the pinned environment.

See `REPRODUCIBILITY.md` for detailed instructions and `RESULTS_SUMMARY.md` for the numerical checks reported in the manuscript.

## License

The code is released under the MIT License. See `LICENSE`.

## Citation

Citation metadata are provided in `CITATION.cff`. If you use this code or the archived experimental results, please cite the accompanying manuscript.
