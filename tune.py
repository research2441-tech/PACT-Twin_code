"""Hyperparameter selection on an independent development cohort (seed 99)."""
import sys, time, copy, json, numpy as np, jax
from sim_exp import *
import twin
d, du, sp, D = make_cohort(99)
dtr, dva, dcal, dte = (subset(d, sp[k]) for k in ('train', 'val', 'cal', 'test'))
grid = [dict(d=48, lr=7e-4, dropout=.2, wd=1e-3, pat=6), dict(d=64, lr=5e-4, dropout=.3, wd=1e-3, pat=6),
        dict(d=32, lr=1e-3, dropout=.1, wd=1e-3, pat=6), dict(d=48, lr=3e-4, dropout=.2, wd=1e-3, pat=10),
        dict(d=64, lr=3e-4, dropout=.3, wd=1e-2, pat=10)]
res = []
for g in grid:
    t = time.time()
    cfg = twin.make_cfg(d, d=g['d'])
    o = copy.deepcopy(twin.DEFAULT_OPTS); o['dropout'] = g['dropout']
    p = twin.init(jax.random.PRNGKey(0), cfg)
    p, v = twin.fit(p, dtr, dva, cfg, o, lr=g['lr'], wd=g['wd'], patience=g['pat'])
    pc, pt = twin.predict(p, dcal, cfg, o), twin.predict(p, dte, cfg, o)
    for x in (pc, pt): x['g12'] = x['Sc'][:, 5]
    m, _ = eval_method(pc, pt, dcal, dte, 'rmst', 'p12')
    c = cindex(dte['os_t'], dte['os_e'], -pt['rmst'][np.arange(len(dte['arm'])), dte['arm']])
    r = dict(g, val=round(float(v), 4), C=round(c, 4), pehe_ov=round(m['pehe_ov'], 3), pehe=round(m['pehe_all'], 3),
             v_ov=round(m['v_overlap'], 3), v_obs=round(m['v_obs'], 3), sec=round(time.time() - t))
    print(r, flush=True); res.append(r)
json.dump(res, open('runs/tune_dev99.json', 'w'), indent=1)
