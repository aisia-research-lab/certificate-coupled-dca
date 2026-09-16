#!/usr/bin/env python
from pathlib import Path
import json, time, platform, sys, hashlib
import numpy as np, pandas as pd
from cc_dca.problem import make_controlled_instance, make_random_spectral_instance
from cc_dca.reference import solve_trs_global
from cc_dca.methods import run_plain_dca, run_post_dca_globalization, run_cc_dca
from cc_dca.correction import structural_correction, alpha_kappa

HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/"experiment_config.json").read_text())
OUT=HERE/"results"/"reproduction"; OUT.mkdir(parents=True,exist_ok=True)
CK=OUT/"checkpoints"; CK.mkdir(exist_ok=True)

def save_row(exp,row):
    p=CK/f"{exp}.jsonl"
    with p.open("a") as f: f.write(json.dumps(row,default=float)+"\n")

def keys_done(exp, fields):
    p=CK/f"{exp}.jsonl"; s=set()
    if p.exists():
        for line in p.read_text().splitlines():
            r=json.loads(line); s.add(tuple(r.get(x) for x in fields))
    return s

def ref_metrics(inst,ref):
    x=ref.x; lam=ref.lam
    return dict(ref_feas=max(0.,np.linalg.norm(x)-inst.r),
                ref_stationarity=np.linalg.norm(inst.A@x+inst.b+lam*x),
                ref_complementarity=abs(lam*(np.linalg.norm(x)-inst.r)),
                ref_shifted_psd=float(np.min(inst.eigvals)+lam))

def rr(r,ref=None):
    d=dict(method=r.method,status=r.status,N_DCA=r.N_DCA,N_E=r.N_E,N_Av=r.N_Av,
           first_escape_dca=r.first_escape_dca,q=r.q,S=r.S,runtime_sec=r.runtime_sec,
           stages_completed=r.stages_completed)
    if ref is not None: d["objective_gap"]=r.q-ref.qstar
    return d

def E1():
    done=keys_done("E1",["n","seed","method"])
    for n in CFG["E1"]["dimensions"]:
      for seed in CFG["global"]["seeds"]:
        inst=make_controlled_instance(n=n,seed=seed,initial_mix=CFG["E1"]["initial_mix"])
        ref=solve_trs_global(inst)
        methods=[
          run_plain_dca(inst,CFG["E1"]["stationarity_tolerance_post_and_plain"],rho=inst.M,max_dca=50000),
          run_post_dca_globalization(inst,CFG["E1"]["stationarity_tolerance_post_and_plain"],CFG["E1"]["delta"],CFG["global"]["p_total"],rho=inst.M,spectral_seed=100000+seed,max_dca=50000,max_escapes=1000),
          run_cc_dca(inst,CFG["E1"]["delta"],.5,1,CFG["global"]["p_total"],spectral_seed=100000+seed,persistent=False,max_dca=50000),
          run_cc_dca(inst,CFG["E1"]["delta"],.5,1,CFG["global"]["p_total"],spectral_seed=100000+seed,persistent=True,max_dca=50000)]
        for r in methods:
          key=(n,seed,r.method)
          if key not in done:
            save_row("E1",{"experiment":"E1","n":n,"seed":seed,**rr(r,ref),**ref_metrics(inst,ref)})

def E2():
    done=keys_done("E2",["seed","eta"])
    n=CFG["E2"]["dimension"]
    for seed in CFG["global"]["seeds"]:
      inst=make_controlled_instance(n=n,seed=seed)
      d,Q=np.linalg.eigh(inst.A); v1=Q[:,0]; v2=Q[:,1]
      for eta in CFG["E2"]["alignment_eta"]:
        if (seed,eta) in done: continue
        x=inst.r*(eta*v1+np.sqrt(max(0,1-eta*eta))*v2)
        kappa=-float(v1@inst.B(x)@v1); a=alpha_kappa(inst,kappa)
        threshold=4/27*kappa*a*inst.r; corr=structural_correction(inst,x,v1,kappa)
        save_row("E2",{"experiment":"E2","n":n,"seed":seed,"eta":eta,"S":inst.S(x),
          "threshold":threshold,"theorem_hypothesis":bool(inst.S(x)<=threshold*(1+1e-12)),
          "case":corr.case,"alignment":corr.alignment,"decrease":corr.decrease,
          "predicted_delta":corr.predicted_delta,"R_dec":corr.ratio})

def E3():
    done=keys_done("E3",["n","seed"])
    for n in CFG["E3"]["dimensions"]:
      for seed in CFG["global"]["seeds"]:
        if (n,seed) in done: continue
        inst=make_controlled_instance(n=n,seed=seed)
        kw=dict(delta0=CFG["E3"]["delta0"],beta=CFG["E3"]["beta"],n_stages=CFG["E3"]["n_stages"],
                p_total=CFG["global"]["p_total"],spectral_seed=100000+seed,max_dca=50000)
        cold=run_cc_dca(inst,persistent=False,**kw); per=run_cc_dca(inst,persistent=True,**kw)
        save_row("E3",{"experiment":"E3","n":n,"seed":seed,
          "cold_N_Av":cold.N_Av,"persistent_N_Av":per.N_Av,"R_Av":cold.N_Av/per.N_Av,
          "cold_N_DCA":cold.N_DCA,"persistent_N_DCA":per.N_DCA,
          "cold_N_E":cold.N_E,"persistent_N_E":per.N_E,
          "cold_status":cold.status,"persistent_status":per.status,
          "cold_runtime":cold.runtime_sec,"persistent_runtime":per.runtime_sec})

def E4():
    done=keys_done("E4",["n","seed","mode"])
    fd=CFG["E4"]["final_delta"]
    for n in CFG["E4"]["dimensions"]:
      for seed in CFG["global"]["seeds"]:
        inst=make_controlled_instance(n=n,seed=seed); ref=solve_trs_global(inst)
        prog=run_cc_dca(inst,CFG["E4"]["delta0"],CFG["E4"]["beta"],CFG["E4"]["n_stages"],CFG["global"]["p_total"],spectral_seed=200000+seed,persistent=True,max_dca=50000)
        cold=run_cc_dca(inst,fd,CFG["E4"]["beta"],1,CFG["global"]["p_total"],spectral_seed=200000+seed,persistent=True,max_dca=50000)
        for mode,r in [("progressive",prog),("cold_final",cold)]:
          if (n,seed,mode) not in done:
            save_row("E4",{"experiment":"E4","n":n,"seed":seed,"mode":mode,"final_delta":fd,**rr(r,ref),**ref_metrics(inst,ref)})

def E5():
    done=keys_done("E5",["axis","value","n","seed","negfrac","bscale"])
    base=CFG["E5"]["base_dimension"]
    for seed in CFG["global"]["seeds"]:
      specs=[]
      for gap in CFG["E5"]["spectral_gap_grid"]: specs.append(("gap",gap,base,dict(lambda1=-5,lambda2=-5+gap)))
      for mult in CFG["E5"]["bottom_multiplicity_grid"]: specs.append(("multiplicity",mult,base,dict(multiplicity=mult)))
      for neg in CFG["E5"]["extra_negative_grid"]: specs.append(("extra_negative",neg,base,dict(extra_negative=neg)))
      for n in CFG["E5"]["dimension_grid"]: specs.append(("dimension",n,n,dict()))
      for axis,val,n,kw in specs:
        key=(axis,val,n,seed,None,None)
        if key in done: continue
        inst=make_controlled_instance(n=n,seed=seed,**kw)
        r=run_cc_dca(inst,1,.5,5,CFG["global"]["p_total"],spectral_seed=300000+seed,persistent=True,max_dca=50000)
        save_row("E5",{"experiment":"E5","axis":axis,"value":val,"n":n,"seed":seed,"negfrac":None,"bscale":None,**rr(r)})
      gr=CFG["E5"]["generic_random_robustness"]
      for n in gr["dimensions"]:
       for nf in gr["negative_fraction"]:
        for bs in gr["b_scale"]:
          key=("generic",0,n,seed,nf,bs)
          if key in done: continue
          inst=make_random_spectral_instance(n,seed,negative_fraction=nf,b_scale=bs)
          ref=solve_trs_global(inst)
          r=run_cc_dca(inst,1,.5,5,CFG["global"]["p_total"],spectral_seed=400000+seed,persistent=True,max_dca=50000)
          save_row("E5",{"experiment":"E5","axis":"generic","value":0,"n":n,"seed":seed,"negfrac":nf,"bscale":bs,**rr(r,ref),**ref_metrics(inst,ref)})

def materialize():
    for exp in ["E1","E2","E3","E4","E5"]:
      p=CK/f"{exp}.jsonl"
      if p.exists():
        df=pd.read_json(p,lines=True); df.to_csv(OUT/f"{exp}.csv",index=False)
    manifest={"freeze":CFG,"python":sys.version,"numpy":np.__version__,"pandas":pd.__version__,
      "platform":platform.platform(),"timestamp_utc":pd.Timestamp.utcnow().isoformat(),
      "code_sha256":{}}
    for p in sorted((HERE/"cc_dca").glob("*.py")):
      manifest["code_sha256"][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/"run_manifest.json").write_text(json.dumps(manifest,indent=2))

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--exp",default="all")
    a=ap.parse_args()
    fs={"E1":E1,"E2":E2,"E3":E3,"E4":E4,"E5":E5}
    for k,f in fs.items():
      if a.exp in ("all",k):
        print("RUN",k,flush=True); f(); materialize()
    materialize()
