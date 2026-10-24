"""Summaries for the sensitivity, landmark, real-data and attribution experiments."""
import glob, pickle, json
import numpy as np
from sim_exp import make_cohort, subset, eval_method
from analysis_sim import oracle_w


def sens():
    out = {'gamma': {}, 'label': {}}
    for f in sorted(glob.glob('runs/sens_gamma_*.pkl')):
        seed = int(f.split('_')[-1].split('.')[0])
        R = pickle.load(open(f, 'rb'))
        for g, blk in R.items():
            d, du, sp, D = make_cohort(seed, gamma_u=g)
            cal, te = subset(d, sp['cal']), subset(d, sp['test'])
            for m, P in blk['preds'].items():
                r, _ = eval_method(P['cal'], P['test'], cal, te)
                out['gamma'].setdefault(m, {}).setdefault(str(g), []).append(r)
                if m == 'Twin':
                    r, _ = eval_method(oracle_w(P['cal'], cal), oracle_w(P['test'], te), cal, te)
                    out['gamma'].setdefault('Twin_oraclew', {}).setdefault(str(g), []).append(r)
    for f in sorted(glob.glob('runs/sim_seed[0-9].pkl')):
        R = pickle.load(open(f, 'rb'))
        if R['seed'] > 2:
            continue
        d, du, sp, D = make_cohort(R['seed'])
        cal, te = subset(d, sp['cal']), subset(d, sp['test'])
        for m, key in [('Twin', 'Twin:full'), ('CoxT', 'CoxT'), ('TARNet', 'TARNet')]:
            r, _ = eval_method(R['preds'][key]['cal'], R['preds'][key]['test'], cal, te)
            out['gamma'].setdefault(m, {}).setdefault('0.3', []).append(r)
        P = R['preds']['Twin:full']
        r, _ = eval_method(oracle_w(P['cal'], cal), oracle_w(P['test'], te), cal, te)
        out['gamma'].setdefault('Twin_oraclew', {}).setdefault('0.3', []).append(r)
        for m in ['Twin:full', 'Twin:no_ssl', 'CoxT']:
            r, _ = eval_method(R['preds'][m]['cal'], R['preds'][m]['test'], cal, te)
            out['label'].setdefault(m, {}).setdefault('1.0', []).append(r)
    for f in sorted(glob.glob('runs/sens_label_*.pkl')):
        seed = int(f.split('_')[-1].split('.')[0])
        R = pickle.load(open(f, 'rb'))
        d, du, sp, D = make_cohort(seed)
        cal, te = subset(d, sp['cal']), subset(d, sp['test'])
        for frac, blk in R.items():
            for m, P in blk['preds'].items():
                r, _ = eval_method(P['cal'], P['test'], cal, te)
                out['label'].setdefault(m, {}).setdefault(str(frac), []).append(r)
    keys = ['pehe_ov', 'v_overlap', 'v_conf_fallback', 'fer_max', 'C_os', 'defer', 'fer_0', 'fer_1', 'fer_2', 'fer_3', 'power']
    S = {kind: {m: {g: {k: (float(np.mean([x[k] for x in L])), float(np.std([x[k] for x in L])), len(L)) for k in keys}
                    for g, L in v.items()} for m, v in out[kind].items()} for kind in out}
    json.dump(S, open('runs/sens_summary.json', 'w'), indent=1)
    return S


def landmark():
    out = {}
    for kind in ('sim', 'real'):
        rs = [pickle.load(open(f, 'rb')) for f in sorted(glob.glob(f'runs/landmark_{kind}_*.pkl'))]
        if not rs:
            continue
        out[kind] = {str(l): {m: (float(np.mean([r[l][m] for r in rs])), float(np.std([r[l][m] for r in rs])))
                              for m in rs[0][l]} for l in rs[0]}
    json.dump(out, open('runs/landmark_summary.json', 'w'), indent=1)
    return out


if __name__ == '__main__':
    import sys
    for f in sys.argv[1:]:
        print(json.dumps(globals()[f](), indent=0)[:3000])
