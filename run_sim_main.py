"""Main LungTwinSim experiment for one seed: twin, ablations, baselines,
missing-modality stress tests and genomic-reveal (value of information) sets.

usage: python run_sim_main.py SEED
"""
import sys, time, pickle, copy
import numpy as np
from sim_exp import make_cohort, subset
from lungsim import MODS
import twin, pipeline
from baselines import TabularT, DRGB, NeuralCATE

EV_OF_MOD = {'gen': [7], 'pdl1': [6], 'wsi': [5], 'ctdna': [4], 'pet': [2], 'rna': [8], 'ct': []}
STRESS = {'drop_gen': ['gen'], 'drop_pdl1': ['pdl1'], 'drop_wsi': ['wsi'], 'drop_ctdna': ['ctdna'],
          'drop_pet_rna': ['pet', 'rna'], 'ct_ehr_only': ['gen', 'pdl1', 'wsi', 'ctdna', 'pet', 'rna']}
KEEP = ('rmst', 'p12', 'prop', 'g12', 'pheno', 'gate', 'resp')


def drop_mods(d, mods):
    d = copy.deepcopy(d)
    names = d['mod_names']
    for m in mods:
        j = names.index(m)
        d['av'][:, j] = 0
        d['mods'][m][:] = 0
        for ty in EV_OF_MOD[m]:
            d['ev_m'][d['ev_ty'] == ty] = 0
    return d


def reveal_gen(d, D, idx):
    """Make next-generation sequencing available before the decision for patients in idx."""
    d = copy.deepcopy(d)
    j = d['mod_names'].index('gen')
    miss = np.where(d['av'][:, j] == 0)[0]
    full = D['gen_full'][idx]
    for i in miss:
        d['av'][i, j] = 1
        d['mods']['gen'][i] = full[i]
        slot = int(d['ev_m'][i].sum())
        if slot < d['ev_m'].shape[1]:
            d['ev_x'][i, slot] = 0
            d['ev_x'][i, slot, :6] = full[i]
            d['ev_t'][i, slot] = -0.05
            d['ev_ty'][i, slot] = 7
            d['ev_m'][i, slot] = 1
    d['revealed'] = np.isin(np.arange(len(d['arm'])), miss)
    return d


def slim(pr):
    return {k: np.asarray(v, np.float32) for k, v in pr.items() if k in KEEP + ('rmst_dr', 'p12_dr')}


def main(seed, quick=False):
    t0 = time.time()
    d, du, sp, D = make_cohort(seed)
    sets = {k: subset(d, sp[k]) for k in ('cal', 'test', 'ext_E', 'ext_F')}
    stress = {f'test__{k}': drop_mods(sets['test'], v) for k, v in STRESS.items()}
    stress['test__reveal'] = reveal_gen(sets['test'], D, sp['test'])
    stress['ext_F__reveal'] = reveal_gen(sets['ext_F'], D, sp['ext_F'])
    dtr, dva = subset(d, sp['train']), subset(d, sp['val'])
    cfg = twin.make_cfg(d, d=48)
    R = dict(seed=seed, sp=sp, timing={}, preds={})
    variants = ['full', 'no_ssl', 'no_events', 'untyped', 'no_moe', 'no_avail', 'attmoe', 'no_moddrop', 'dr_mlp', 'dr_shift']
    if quick:
        variants = ['full']
    for v in variants:
        t = time.time()
        ev = dict(sets)
        if v == 'full':
            ev.update(stress)
        out = pipeline.run(dtr, dva, ev, cfg, v, seed=seed, unlabeled=du)
        out.pop('_dr', None)
        R['preds']['Twin:' + v] = {k: slim(o) for k, o in out.items()}
        R['timing']['Twin:' + v] = time.time() - t
        print(seed, v, round(time.time() - t), flush=True)
    base = [('CoxT', lambda: TabularT('cox', seed)), ('RSFT', lambda: TabularT('rsf', seed)), ('DRGB', lambda: DRGB(seed)),
            ('SMLP', lambda: NeuralCATE('smlp', 64, seed)), ('TARNet', lambda: NeuralCATE('tarnet', 64, seed)),
            ('CFR', lambda: NeuralCATE('cfr', 64, seed)), ('CRN', lambda: NeuralCATE('crn', 64, seed))]
    if quick:
        base = base[:1]
    for name, mk in base:
        t = time.time()
        m = mk()
        if isinstance(m, NeuralCATE):
            m.fit_all(dtr, dva)
        else:
            m.fit(dtr)
        R['timing'][name] = time.time() - t
        ev = dict(sets)
        if name in ('CoxT', 'TARNet', 'CRN'):
            ev.update(stress)
        R['preds'][name] = {k: slim(m.predict(x)) for k, x in ev.items()}
        print(seed, name, round(time.time() - t), flush=True)
    R['total'] = time.time() - t0
    pickle.dump(R, open(f'runs/sim_seed{seed}.pkl', 'wb'))
    print('done', seed, round(R['total']))


if __name__ == '__main__':
    main(int(sys.argv[1]), quick=len(sys.argv) > 2)
