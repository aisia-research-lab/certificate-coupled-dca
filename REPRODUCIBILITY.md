# Reproducibility Guide

This repository accompanies **“Certificate-Coupled DCA for the Trust-Region Subproblem: Early Structural Escape and Progressive Global Certification.”**

## 1. Environment

The archived run used Python 3.13.5 with NumPy 2.3.5 and Pandas 2.2.3. SciPy 1.17.0 is pinned because the reference TRS solver uses `scipy.optimize.brentq`.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Configuration and source provenance

The experiment configuration is stored in `experiment_config.json`. The archived provenance file `reference_results/run_manifest.json` records the original environment and SHA-256 hashes of the scientific source files under `cc_dca/`. Those source files are unchanged in this release.

The manifest retains a few internal development labels in its historical metadata. They are provenance fields only and do not affect the public directory structure or the numerical results.

## 3. Running the experiments

Run all experiment families with:

```bash
python run_experiments.py --exp all
```

or one family, for example:

```bash
python run_experiments.py --exp E2
```

Valid identifiers are `E1`, `E2`, `E3`, `E4`, and `E5`. A complete run contains 320, 120, 80, 160, and 620 rows, respectively, for a total of 1,300 records.

## 4. Outputs and restart behavior

Fresh runs write to `results/reproduction/`. Completed experimental units are appended to `results/reproduction/checkpoints/<EXPERIMENT>.jsonl`; on restart, completed keys are skipped. Materialized CSV files and a fresh provenance manifest are written to the same generated directory.

The archived release snapshot is stored separately in `reference_results/` and is not modified by the runner.

## 5. Seeds and work counters

Instance seeds are 0--19. Fixed spectral seed offsets are specified by `run_experiments.py` and the experiment configuration. The principal work counters are `N_DCA` (DCA updates), `N_E` (accepted negative-curvature corrections), and `N_Av` (matrix-vector products with `A`). Offline reference eigendecompositions are not charged to `N_Av`.

Runtime fields depend on hardware and system load and are not expected to match exactly. Deterministic numerical fields should agree up to ordinary floating-point tolerance under the pinned environment.

## 6. Verification targets

Representative checks against the archived results are:

- E1: 320 rows; median first correction 17.5 DCA updates for both certificate-coupled variants and 40.0 for the post-convergence method.
- E2: 120 rows; `R_dec` minimum 53.983, median 54.221, maximum 57.868.
- E3: 80 matched runs; median cold/persistent `R_Av` 2.612, with identical `N_DCA` and `N_E` on all matched runs.
- E4: median saving of 12 DCA updates and median increase of 35 spectral matrix-vector products for progressive certification relative to the direct one-stage final-tolerance run.
- E5: 620 rows; the generic-random subset contains 240 runs and has maximum objective gap approximately 0.0749836.

See `RESULTS_SUMMARY.md` for the complete numerical summary used in the manuscript.

## 7. Clean rerun

To start a fresh reproduction:

```bash
rm -rf results/reproduction
python run_experiments.py --exp all
```

Do not delete `reference_results/` when comparing a new run with the archived release snapshot.
