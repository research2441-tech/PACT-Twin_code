"""Dynamic twin updating: residual overall-survival discrimination at landmark
times after first-line start, using events accrued up to the landmark.

Compared at each landmark (patients alive and under follow-up at the landmark):
  Twin-updated   : twin state rebuilt from all events up to the landmark
  Twin-baseline  : twin restricted to decision-time information (no post-decision events)
  CoxT-LOCF      : Cox on flattened features with last-observation event summaries at the landmark
usage: python run_landmark.py sim|real SEED
"""
import sys, pickle, copy
import numpy as np
from data_prep import from_sim, from_chord, subset, _targets
from sim_exp import make_cohort, cindex
from run_real import splits
from baselines import TabularT
import twin, pipeline
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sklearn.preprocessing import StandardScaler
from data_prep import flat


def at_landmark(d, ell):
    keep = np.where(d['os_t'] > ell)[0]
    s = subset(d, keep)
    s['os_t'] = s['os_t'] - ell
    s['pfs_t'], s['pfs_e'] = s['os_t'].copy(), s['os_e'].copy()
    s.update(_targets(s))
    return s, keep


def cox_flat(dtr, dte):
    Xtr, Xte = flat(dtr), flat(dte)
    sc = StandardScaler().fit(Xtr)
    m = CoxPHSurvivalAnalysis(alpha=10.0).fit(sc.transform(Xtr), Surv.from_arrays(dtr['os_e'].astype(bool), dtr['os_t']))
    return m.predict(sc.transform(Xte))


def main(kind, seed):
    res = {}
    if kind == 'sim':
        d0, du, sp, D = make_cohort(seed)
        lms = [0.0, 2.0, 4.0, 6.0]
        cfg_kw, fk = dict(d=48), {}
        build = lambda ell: from_sim(D, cutoff=ell - 1e-3 if ell == 0 else ell, L=24)
    else:
        R = pickle.load(open('data/chord_cohort.pkl', 'rb'))
        d0 = from_chord(R)
        sp = splits(d0, 'r', seed)
        lms = [0.0, 3.0, 6.0]
        cfg_kw, fk = dict(d=16), dict(lr=1e-3, wd=1e-2, patience=10, bs=64, epochs=150)
        build = lambda ell: from_chord(R, cutoff_days=7.0 if ell == 0 else ell * 30.4375, L=32)
    base_d = build(0.0)
    for ell in lms:
        dl = build(ell)
        dl_s, keep = at_landmark(dl, ell)
        db_s, _ = at_landmark(base_d, ell)
        pos = {k: np.where(np.isin(keep, sp[k]))[0] for k in ('train', 'val', 'test')}
        r = {}
        cfg = twin.make_cfg(dl_s, **cfg_kw)
        tr, va, te = (subset(dl_s, pos[k]) for k in ('train', 'val', 'test'))
        o = pipeline.run(tr, va, {'test': te}, cfg, 'no_ssl', seed=seed, fit_kw=fk)['test']
        n = len(te['arm'])
        r['Twin-updated'] = cindex(te['os_t'], te['os_e'], -o['rmst'][np.arange(n), te['arm']])
        trb, vab, teb = (subset(db_s, pos[k]) for k in ('train', 'val', 'test'))
        ob = pipeline.run(trb, vab, {'test': teb}, cfg, 'no_ssl', seed=seed, fit_kw=fk)['test']
        r['Twin-baseline'] = cindex(teb['os_t'], teb['os_e'], -ob['rmst'][np.arange(n), teb['arm']])
        r['CoxT-LOCF'] = cindex(te['os_t'], te['os_e'], cox_flat(tr, te))
        r['CoxT-baseline'] = cindex(teb['os_t'], teb['os_e'], cox_flat(trb, teb))
        r['n_test'] = int(n)
        res[ell] = r
        print(kind, seed, ell, {k: round(v, 4) for k, v in r.items()}, flush=True)
    pickle.dump(res, open(f'runs/landmark_{kind}_{seed}.pkl', 'wb'))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
