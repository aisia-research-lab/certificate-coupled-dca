from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .problem import make_controlled_instance
from .reference import solve_trs_global
from .methods import run_plain_dca, run_post_dca_globalization, run_cc_dca
from .correction import structural_correction, alpha_kappa


def e1_early_escape(outdir, dimensions=(40,80,120), seeds=(0,1,2),
                    stationarity_tol=1e-10, delta=0.5, p=0.1):
    rows=[]
    for n in dimensions:
        for seed in seeds:
            inst=make_controlled_instance(n=n,seed=seed,initial_mix=.35)
            ref=solve_trs_global(inst)
            plain=run_plain_dca(inst,stationarity_tol,rho=inst.M,max_dca=50000)
            post=run_post_dca_globalization(inst,stationarity_tol,delta,p,rho=inst.M,
                                            spectral_seed=100000+seed,max_dca=50000,max_escapes=20)
            cc=run_cc_dca(inst,delta,0.5,1,p,spectral_seed=100000+seed,
                          persistent=True,max_dca=50000)
            for r in [plain,post,cc]:
                rows.append({"experiment":"E1","n":n,"seed":seed,"method":r.method,
                    "status":r.status,"first_escape_dca":r.first_escape_dca,"N_DCA":r.N_DCA,
                    "N_E":r.N_E,"N_Av":r.N_Av,"objective_gap":r.q-ref.qstar,"S":r.S})
    df=pd.DataFrame(rows); Path(outdir).mkdir(parents=True,exist_ok=True)
    df.to_csv(Path(outdir)/"E1_early_escape.csv",index=False); return df


def e2_hard_geometry(outdir, n=80, seeds=(0,1,2),
                     etas=(0.0,1e-6,1e-5,1e-4,5e-4,9e-4)):
    rows=[]
    for seed in seeds:
        inst=make_controlled_instance(n=n,seed=seed)
        d,Q=np.linalg.eigh(inst.A); v1=Q[:,0]; v2=Q[:,1]
        for eta in etas:
            x=inst.r*(eta*v1+np.sqrt(max(0,1-eta*eta))*v2)
            curvature=float(v1@inst.B(x)@v1); kappa=-curvature
            a=alpha_kappa(inst,kappa); threshold=4/27*kappa*a*inst.r
            corr=structural_correction(inst,x,v1,kappa)
            rows.append({"experiment":"E2","seed":seed,"eta":eta,"S":inst.S(x),
                "threshold":threshold,"theorem_hypothesis":inst.S(x)<=threshold*(1+1e-12),
                "case":corr.case,"alignment":corr.alignment,"decrease":corr.decrease,
                "predicted_delta":corr.predicted_delta,"R_dec":corr.ratio})
    df=pd.DataFrame(rows); Path(outdir).mkdir(parents=True,exist_ok=True)
    df.to_csv(Path(outdir)/"E2_hard_geometry.csv",index=False); return df


def e3_persistence(outdir, dimensions=(80,120,200), seeds=(0,1,2),
                   delta0=1.0,beta=.5,n_stages=5,p=.1):
    rows=[]
    for n in dimensions:
        for seed in seeds:
            inst=make_controlled_instance(n=n,seed=seed)
            cold=run_cc_dca(inst,delta0,beta,n_stages,p,spectral_seed=100000+seed,persistent=False)
            per=run_cc_dca(inst,delta0,beta,n_stages,p,spectral_seed=100000+seed,persistent=True)
            rows.append({"experiment":"E3","n":n,"seed":seed,
                "cold_N_Av":cold.N_Av,"persistent_N_Av":per.N_Av,
                "R_Av":cold.N_Av/per.N_Av,"cold_N_DCA":cold.N_DCA,
                "persistent_N_DCA":per.N_DCA,"cold_status":cold.status,"persistent_status":per.status})
    df=pd.DataFrame(rows); Path(outdir).mkdir(parents=True,exist_ok=True)
    df.to_csv(Path(outdir)/"E3_persistence.csv",index=False); return df


def e4_progressive(outdir, dimensions=(80,120,200), seeds=(0,1,2),
                   delta0=1.0,beta=.5,n_stages=5,p=.1):
    rows=[]
    final_delta=delta0*beta**(n_stages-1)
    for n in dimensions:
        for seed in seeds:
            inst=make_controlled_instance(n=n,seed=seed)
            ref=solve_trs_global(inst)
            prog=run_cc_dca(inst,delta0,beta,n_stages,p,spectral_seed=200000+seed,persistent=True)
            cold=run_cc_dca(inst,final_delta,beta,1,p,spectral_seed=200000+seed,persistent=True)
            for label,r in [("progressive",prog),("cold_final",cold)]:
                rows.append({"experiment":"E4","n":n,"seed":seed,"mode":label,"status":r.status,
                    "N_DCA":r.N_DCA,"N_E":r.N_E,"N_Av":r.N_Av,"objective_gap":r.q-ref.qstar,
                    "final_delta":final_delta})
    df=pd.DataFrame(rows); Path(outdir).mkdir(parents=True,exist_ok=True)
    df.to_csv(Path(outdir)/"E4_progressive.csv",index=False); return df


def e5_sensitivity(outdir, n=80, seeds=(0,1,2), delta0=1.0,beta=.5,n_stages=5,p=.1):
    rows=[]
    for seed in seeds:
        for gap in (0.1,0.5,1.0,2.0):
            inst=make_controlled_instance(n=n,seed=seed,lambda1=-5,lambda2=-5+gap)
            r=run_cc_dca(inst,delta0,beta,n_stages,p,spectral_seed=300000+seed,persistent=True)
            rows.append({"experiment":"E5","axis":"gap","value":gap,"seed":seed,"status":r.status,
                         "N_DCA":r.N_DCA,"N_E":r.N_E,"N_Av":r.N_Av})
        for mult in (1,2,4,8):
            inst=make_controlled_instance(n=n,seed=seed,multiplicity=mult)
            r=run_cc_dca(inst,delta0,beta,n_stages,p,spectral_seed=310000+seed,persistent=True)
            rows.append({"experiment":"E5","axis":"multiplicity","value":mult,"seed":seed,"status":r.status,
                         "N_DCA":r.N_DCA,"N_E":r.N_E,"N_Av":r.N_Av})
        for neg in (0,2,5,10):
            inst=make_controlled_instance(n=n,seed=seed,extra_negative=neg)
            r=run_cc_dca(inst,delta0,beta,n_stages,p,spectral_seed=320000+seed,persistent=True)
            rows.append({"experiment":"E5","axis":"extra_negative","value":neg,"seed":seed,"status":r.status,
                         "N_DCA":r.N_DCA,"N_E":r.N_E,"N_Av":r.N_Av})
    df=pd.DataFrame(rows); Path(outdir).mkdir(parents=True,exist_ok=True)
    df.to_csv(Path(outdir)/"E5_sensitivity.csv",index=False); return df
