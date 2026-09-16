"""Experiment engine for certificate-coupled DCA on trust-region subproblems."""
from .problem import TRSInstance, make_controlled_instance, make_random_spectral_instance
from .reference import solve_trs_global
from .methods import run_plain_dca, run_post_dca_globalization, run_cc_dca
