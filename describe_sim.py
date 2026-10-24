"""Site-level characteristics of the simulated cohort (seed 0) for Table II."""
import json, numpy as np
from sim_exp import make_cohort
from lungsim import SITES
d, du, sp, D = make_cohort(0)
out = {}
for s in SITES:
    m = D['site'] == s
    pre = [sum(1 for e in ev if e[0] < 0) for ev, mm in zip(D['events'], m) if mm]
    out[s] = dict(n=int(m.sum()), asian=float(D['asian'][m].mean()), never=float(np.mean(D['static'][m, 3])),
                  drv=float((D['egfr'] | D['alk'])[m].mean()), pdl1hi=float((D['pdl1'] >= 50)[m].mean()),
                  ecog2=float((D['ps'] >= 2)[m].mean()), gen=float(D['a_gen'][m].mean()), pdl1=float(D['a_pdl1'][m].mean()),
                  wsi=float(D['a_wsi'][m].mean()), ctdna=float(D['a_ctdna'][m].mean()), pet=float(D['a_pet'][m].mean()),
                  rna=float(D['a_rna'][m].mean()), arms=(np.bincount(D['arm'][m], minlength=4) / m.sum()).tolist(),
                  os_ev=float(D['os_e'][m].mean()), events=float(np.mean(pre)), age=float(np.median(D['age'][m])),
                  best=(np.bincount(D['rmst_true'][m].argmax(1), minlength=4) / m.sum()).tolist(),
                  overlap_tki=float((D['prop'][m, 3] >= .05).mean()), mu=D['rmst_true'][m].mean(0).tolist())
out['unlabeled'] = int(len(du['arm']))
json.dump(out, open('runs/sim_describe.json', 'w'), indent=1)
for s in SITES: print(s, {k: (np.round(v, 3).tolist() if isinstance(v, list) else round(v, 3)) for k, v in out[s].items()})
