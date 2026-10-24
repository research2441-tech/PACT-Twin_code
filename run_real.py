"""MSK-CHORD real-world experiments (first-line Chemo / Chemo-IO / IO-mono).

Splits: five stratified random splits (r0..r4) and one temporal split (t0, by
sequencing order). For every split: twin (full + two ablations) and the tabular
and neural comparators; predictions on cal/test (+ stress sets) are stored.
Policy value is estimated afterwards with a held-out AIPW x IPCW evaluator.
"""
import sys, pickle, time, copy
import numpy as np
from data_prep import from_chord, subset
import twin, pipeline
from baselines import TabularT, DRGB, NeuralCATE

EV_OF_MOD = {'genomic': [1], 'pdl1': [2], 'radiology': [5], 'cea': [4], 'ecog': [3]}
STRESS = {'drop_pdl1': ['pdl1'], 'drop_genomic': ['genomic'], 'drop_radiology': ['radiology'],
          'drop_pdl1_genomic': ['pdl1', 'genomic']}
KEEP = ('rmst', 'p12', 'prop', 'g12', 'pheno', 'gate', 'Sc')


def splits(d, kind, rep):
    n = len(d['arm'])
    if kind == 't':
        it = np.argsort(d['order'])
    else:
        rng = np.random.default_rng(100 + rep)
        it = rng.permutation(n)
    a, b, c = int(.55 * n), int(.65 * n), int(.85 * n)
    return dict(train=it[:a], val=it[a:b], cal=it[b:c], test=it[c:])


def drop_mods(d, mods):
    d = copy.deepcopy(d)
    for m in mods:
        j = d['mod_names'].index(m)
        d['av'][:, j] = 0
        d['mods'][m][:] = 0
        for ty in EV_OF_MOD[m]:
            d['ev_m'][d['ev_ty'] == ty] = 0
    return d


def slim(pr):
    return {k: np.asarray(v, np.float32) for k, v in pr.items() if k in KEEP}


def main(tag):
    R0 = pickle.load(open('data/chord_cohort.pkl', 'rb'))
    d = from_chord(R0)
    kind, rep = tag[0], int(tag[1:])
    sp = splits(d, kind, rep)
    sets = {k: subset(d, sp[k]) for k in ('cal', 'test')}
    stress = {f'test__{k}': drop_mods(sets['test'], v) for k, v in STRESS.items()}
    dtr, dva = subset(d, sp['train']), subset(d, sp['val'])
    cfg = twin.make_cfg(d, d=16)
    fk = dict(lr=1e-3, wd=1e-2, patience=10, bs=64, epochs=150)      # selected on development splits 97-99
    base = None
    out = dict(tag=tag, sp=sp, preds={}, timing={})
    for v in ['full', 'attmoe', 'no_moe', 'no_avail', 'no_events', 'untyped', 'no_ssl']:
        t = time.time()
        ev = dict(sets)
        if v == 'full':
            ev.update(stress)
        o = pipeline.run(dtr, dva, ev, cfg, v, seed=rep, fit_kw=fk, base_enc=base)
        out['preds']['Twin:' + v] = {k: slim(x) for k, x in o.items()}
        out['timing']['Twin:' + v] = time.time() - t
        print(tag, v, round(time.time() - t), flush=True)
    for name, mk in [('CoxT', lambda: TabularT('cox', rep)), ('RSFT', lambda: TabularT('rsf', rep)),
                     ('DRGB', lambda: DRGB(rep)), ('SMLP', lambda: NeuralCATE('smlp', 32, rep)),
                     ('TARNet', lambda: NeuralCATE('tarnet', 32, rep)), ('CFR', lambda: NeuralCATE('cfr', 32, rep)),
                     ('CRN', lambda: NeuralCATE('crn', 32, rep))]:
        t = time.time()
        m = mk()
        m.fit_all(dtr, dva) if isinstance(m, NeuralCATE) else m.fit(dtr)
        ev = dict(sets)
        if name in ('CoxT', 'TARNet'):
            ev.update(stress)
        out['preds'][name] = {k: slim(m.predict(x)) for k, x in ev.items()}
        out['timing'][name] = time.time() - t
        print(tag, name, round(time.time() - t), flush=True)
    # independent evaluation nuisance for off-policy value (fitted on train+val only)
    ev_n = TabularT('cox', 1000 + rep).fit(subset(d, np.r_[sp['train'], sp['val']]))
    out['eval_nuisance'] = {k: ev_n.predict(sets[k]) for k in ('cal', 'test')}
    pickle.dump(out, open(f'runs/real_{tag}.pkl', 'wb'))
    print('done', tag)


if __name__ == '__main__':
    main(sys.argv[1])
