# Experimental Results Summary

The archived study contains 1,300 records: E1=320, E2=120, E3=80, E4=160, and E5=620.

## E1 — Acting before post-convergence certification

Across 80 matched dimension--seed pairs, the median number of DCA updates before the first negative-curvature correction is 40.0 for the post-convergence method and 17.5 for both certificate-coupled variants. The certificate-coupled correction occurs 22--23 updates earlier on every matched instance, with median difference 22.5. Plain DCA terminates with median objective gap 1.000.

The cold and persistent certificate-coupled variants have identical first-correction, `N_DCA`, and `N_E` counts on all 80 matched instances. Their median spectral counts are 35.0 and 31.5, respectively, and both attain median final objective gap approximately 8.37e-10.

## E2 — Near-orthogonal negative-curvature correction

All 120 instances satisfy the hypotheses recorded for the quantitative decrease test. For `R_dec=(q(x)-q(x_plus))/Delta`, the minimum, median, and maximum are 53.983, 54.221, and 57.868. Thus `R_dec >= 1` throughout the tested configurations, including exact orthogonality.

## E3 — Krylov persistence

Across 80 matched runs with dimensions 100, 200, 400, and 800, the cold/persistent spectral-work ratio `R_Av` has median 2.612, interquartile range [2.594, 2.692], and range [2.593, 2.920]. The two variants have identical `N_DCA` and `N_E` on all matched runs.

## E4 — Progressive certification

Relative to the direct one-stage run at the final tolerance, progressive certification saves a median of 12 DCA updates and uses a median of 35 additional spectral matrix-vector products. Every final output satisfies the prescribed certificate; the largest observed objective-gap/certificate-radius ratio is approximately 1.04e-12.

## E5 — Spectral sensitivity and generic random instances

For controlled lower spectral gaps 0.1, 0.25, 0.5, 1, and 2, median `N_DCA` values are 1655, 669, 340, 160.5, and 83. Bottom-eigenvalue multiplicity and the number of additional negative directions have modest effects on median DCA work over the tested grid. As dimension increases from 100 to 800, median `N_Av` increases from 100 to 132 while median `N_DCA` changes only mildly.

The generic-random subset contains 240 runs. All terminate with the prescribed full-depth spectral certificate. The largest objective gap is 0.0749836, equal to approximately 0.300 of the final theoretical certificate radius. The largest recorded reference-solution stationarity residual is approximately 1.10e-14, with no shifted positive-semidefiniteness violation beyond the prescribed tolerance.

## Scope

These numerical results correspond to the controlled comparisons and validation experiments reported in the manuscript. They support the observed timing of negative-curvature corrections, the quantitative decrease estimate on the tested near-orthogonal family, the reduction in repeated spectral matrix-vector products from Krylov persistence in E3, and the reported sensitivity to spectral structure.
