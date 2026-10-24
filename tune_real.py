"""Real-data hyperparameter selection by validation C-index on a development split (rep 99)."""
import pickle, copy, json, time, numpy as np, jax
from data_prep import from_chord, subset
from run_real import splits
from sim_exp import cindex
import twin
R0 = pickle.load(open('data/chord_cohort.pkl', 'rb'))
d = from_chord(R0)
sp = splits(d, 'r', 99)
dtr, dva = subset(d, sp['train']), subset(d, sp['val'])
res = []
for g in [dict(d=32, lr=7e-4, dropout=.2, wd=1e-3, pat=6, bs=128), dict(d=16, lr=1e-3, dropout=.2, wd=1e-2, pat=10, bs=64),
          dict(d=32, lr=3e-4, dropout=.4, wd=1e-2, pat=15, bs=64), dict(d=24, lr=5e-4, dropout=.3, wd=3e-2, pat=15, bs=64),
          dict(d=32, lr=3e-4, dropout=.4, wd=1e-2, pat=15, bs=64, mod_drop=.4)]:
    t = time.time()
    cfg = twin.make_cfg(d, d=g['d'])
    o = copy.deepcopy(twin.DEFAULT_OPTS); o['dropout'] = g['dropout']; o['mod_drop'] = g.get('mod_drop', .25)
    p = twin.init(jax.random.PRNGKey(0), cfg)
    p, v = twin.fit(p, dtr, dva, cfg, o, lr=g['lr'], wd=g['wd'], patience=g['pat'], bs=g['bs'], epochs=150)
    pv = twin.predict(p, dva, cfg, o)
    n = len(dva['arm'])
    c = cindex(dva['os_t'], dva['os_e'], -pv['rmst'][np.arange(n), dva['arm']])
    r = dict(g, val=round(float(v), 4), C_val=round(float(c), 4), sec=round(time.time() - t)); print(r, flush=True); res.append(r)
json.dump(res, open('runs/tune_real99.json', 'w'), indent=1)
