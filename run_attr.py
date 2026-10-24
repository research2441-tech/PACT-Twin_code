"""Attribution, expert specialisation and efficiency for the full twin (simulation seed 0).

Occlusion attribution: for modality group m, A[m, a] = mean |tau_a(x) - tau_a(x without m)|
over test patients with m available, with tau_a = mu_a - mu_ChemoIO.
Ground-truth relevance: |Spearman| between true tau_a and latent effect modifiers.
"""
import time, pickle, copy
import numpy as np
import jax
from scipy.stats import spearmanr
from sim_exp import make_cohort, subset
from run_sim_main import drop_mods, EV_OF_MOD
import twin, pipeline
from baselines import TabularT, NeuralCATE


def drop_events(d, types):
    d = copy.deepcopy(d)
    for ty in types:
        d['ev_m'][d['ev_ty'] == ty] = 0
    return d


def nparams(p):
    return int(sum(np.size(x) for x in jax.tree_util.tree_leaves(p)))


def main(seed=0):
    d, du, sp, D = make_cohort(seed)
    dtr, dva, dte = subset(d, sp['train']), subset(d, sp['val']), subset(d, sp['test'])
    cfg = twin.make_cfg(d, d=48)
    t = time.time()
    out, p, opts = pipeline.run(dtr, dva, {'test': dte}, cfg, 'full', seed=seed, unlabeled=du, return_params=True)
    train_time = time.time() - t
    base = out['test']['rmst']
    tau = lambda mu: mu - mu[:, [1]]
    groups = {'Genomics (NGS)': ('mod', ['gen']), 'PD-L1 IHC': ('mod', ['pdl1']), 'WSI': ('mod', ['wsi']),
              'ctDNA': ('mod', ['ctdna']), 'PET': ('mod', ['pet']), 'Transcriptome': ('mod', ['rna']),
              'CT history': ('ev', [1]), 'Laboratory history': ('ev', [3])}
    names = dte['mod_names']
    A = {}
    for g, (kind, what) in groups.items():
        dd = drop_mods(dte, what) if kind == 'mod' else drop_events(dte, what)
        mu = twin.predict(p, dd, cfg, opts)['rmst']
        if kind == 'mod':
            m = dte['av'][:, names.index(what[0])] > 0
        else:
            m = np.isin(dte['ev_ty'], what).any(1) & (dte['ev_m'].sum(1) > 0)
        A[g] = [float(np.abs(tau(base)[m, a] - tau(mu)[m, a]).mean()) for a in (0, 2, 3)]
        print(g, np.round(A[g], 3), flush=True)
    tt = dte['mu_true'] - dte['mu_true'][:, [1]]
    lat = {'EGFR/ALK driver': (dte['g_egfr'] | dte['g_alk']).astype(float), 'Immune score': dte['g_zimm'],
           'PD-L1 TPS': dte['g_pdl1'], 'Frailty': dte['g_frail'], 'Growth rate': np.log(D['g'][sp['test']]),
           'Age': dte['g_age']}
    G = {k: [abs(spearmanr(v, tt[:, a])[0]) for a in (0, 2, 3)] for k, v in lat.items()}
    # expert usage by availability pattern
    gate = out['test']['gate']
    pat = {}
    for gi, gn in [(names.index('gen'), 'gen'), (names.index('pdl1'), 'pdl1'), (names.index('wsi'), 'wsi')]:
        for v in (0, 1):
            m = dte['av'][:, gi] == v
            pat[f'{gn}={v}'] = gate[m].mean(0).tolist()
    # efficiency
    B1 = subset(dte, np.arange(1))
    twin.predict(p, B1, cfg, opts)
    t = time.time()
    for _ in range(50):
        twin.predict(p, B1, cfg, opts)
    lat1 = (time.time() - t) / 50
    t = time.time()
    twin.predict(p, dte, cfg, opts)
    latN = (time.time() - t) / len(dte['arm'])
    eff = {'Twin': dict(params=nparams(p), train_s=train_time, lat1_ms=lat1 * 1e3, latN_ms=latN * 1e3)}
    for nm, mk in [('TARNet', lambda: NeuralCATE('tarnet', 64, seed)), ('CRN', lambda: NeuralCATE('crn', 64, seed)),
                   ('CoxT', lambda: TabularT('cox', seed)), ('RSFT', lambda: TabularT('rsf', seed))]:
        t = time.time()
        m = mk()
        m.fit_all(dtr, dva) if isinstance(m, NeuralCATE) else m.fit(dtr)
        tr_s = time.time() - t
        m.predict(B1)
        t = time.time()
        for _ in range(20):
            m.predict(B1)
        l1 = (time.time() - t) / 20
        t = time.time()
        m.predict(dte)
        lN = (time.time() - t) / len(dte['arm'])
        npar = nparams(m.p) if isinstance(m, NeuralCATE) else None
        eff[nm] = dict(params=npar, train_s=tr_s, lat1_ms=l1 * 1e3, latN_ms=lN * 1e3)
        print(nm, eff[nm], flush=True)
    print(eff['Twin'])
    pickle.dump(dict(A=A, G=G, pat=pat, eff=eff, contrasts=['Chemo', 'IO-mono', 'TKI']), open('runs/attr_seed0.pkl', 'wb'))


if __name__ == '__main__':
    main(0)
