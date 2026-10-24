"""Sensitivity experiments on LungTwinSim.
  gamma : strength of the unmeasured confounder (0, 0.6, 1.0; 0.3 is the main setting)
  label : fraction of labelled training patients (0.1, 0.25, 0.5) with and without SSL
usage: python run_sens.py gamma|label SEED
"""
import sys, pickle, time
import numpy as np
from sim_exp import make_cohort, subset
import twin, pipeline
from baselines import TabularT, NeuralCATE

KEEP = ('rmst', 'p12', 'prop', 'g12')


def slim(pr):
    return {k: np.asarray(v, np.float32) for k, v in pr.items() if k in KEEP}


def main(kind, seed):
    out = {}
    if kind == 'gamma':
        for g in (0.0, 0.6, 1.0):
            d, du, sp, D = make_cohort(seed, gamma_u=g)
            sets = {k: subset(d, sp[k]) for k in ('cal', 'test')}
            dtr, dva = subset(d, sp['train']), subset(d, sp['val'])
            cfg = twin.make_cfg(d, d=48)
            o = pipeline.run(dtr, dva, sets, cfg, 'full', seed=seed, unlabeled=du)
            res = {'Twin': {k: slim(v) for k, v in o.items()}}
            res['CoxT'] = {k: slim(TabularT('cox', seed).fit(dtr).predict(x)) for k, x in sets.items()}
            tn = NeuralCATE('tarnet', 64, seed).fit_all(dtr, dva)
            res['TARNet'] = {k: slim(tn.predict(x)) for k, x in sets.items()}
            out[g] = dict(sp=sp, preds=res)
            print(kind, seed, g, flush=True)
    else:
        d, du, sp, D = make_cohort(seed)
        sets = {k: subset(d, sp[k]) for k in ('cal', 'test')}
        dva = subset(d, sp['val'])
        cfg = twin.make_cfg(d, d=48)
        rng = np.random.default_rng(seed + 7)
        for f in (0.1, 0.25, 0.5):
            idx = rng.permutation(sp['train'])[: int(f * len(sp['train']))]
            dtr = subset(d, idx)
            res = {}
            for v in ('full', 'no_ssl'):
                o = pipeline.run(dtr, dva, sets, cfg, v, seed=seed, unlabeled=du)
                res['Twin:' + v] = {k: slim(x) for k, x in o.items()}
            res['CoxT'] = {k: slim(TabularT('cox', seed).fit(dtr).predict(x)) for k, x in sets.items()}
            out[f] = dict(sp=sp, preds=res)
            print(kind, seed, f, flush=True)
    pickle.dump(out, open(f'runs/sens_{kind}_{seed}.pkl', 'wb'))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
