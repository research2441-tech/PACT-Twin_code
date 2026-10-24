"""Full twin pipeline: SSL pretraining -> supervised twin -> cross-fitted nuisance ->
doubly robust potential-outcome heads -> predictions for any number of evaluation sets."""
import copy
import numpy as np
import jax
import twin
from data_prep import subset, J


def variant_opts(name, base_enc=None):
    o = copy.deepcopy(twin.DEFAULT_OPTS)
    if base_enc:
        o['enc'].update(base_enc)
    ssl, dr = True, False
    if name == 'full':
        pass
    elif name == 'no_ssl':
        ssl = False
    elif name == 'no_events':
        o['enc']['use_events'] = False
    elif name == 'untyped':
        o['enc']['typed'] = False
    elif name == 'no_moe':
        o['enc']['use_moe'] = False
    elif name == 'no_avail':          # availability neither gates experts nor enters the propensity
        o['enc']['use_gate_av'] = False
        o['prop_av'] = False
    elif name == 'no_att':
        o['enc']['use_att'] = False
    elif name == 'dr_mlp':
        dr = 'mlp'
    elif name == 'dr_shift':
        dr = 'shift'
    elif name == 'attmoe':
        o['enc']['use_att'] = True
        o['enc']['use_moe'] = True
    elif name == 'no_moddrop':
        o['mod_drop'] = 0.0
    else:
        raise ValueError(name)
    return o, ssl, dr


def _g12(Sc):
    return Sc[:, 5]


def run(dtr, dva, evalsets, cfg, variant='full', seed=0, unlabeled=None, ssl_epochs=15, verbose=False,
        return_params=False, fit_kw=None, base_enc=None):
    opts, use_ssl, use_dr = variant_opts(variant, base_enc)
    fk = fit_kw or {}
    key = jax.random.PRNGKey(seed)
    p0 = twin.init(key, cfg)
    if use_ssl:
        pool = dtr if unlabeled is None else _concat(dtr, unlabeled)
        p0, _ = twin.fit(p0, pool, dva, cfg, opts, loss='ssl', epochs=ssl_epochs, seed=seed, patience=3)
    p, _ = twin.fit(p0, dtr, dva, cfg, opts, seed=seed + 1, **fk)
    out = {k: twin.predict(p, d, cfg, opts) for k, d in evalsets.items()}
    for k in out:
        out[k]['g12'] = _g12(out[k]['Sc'])
    if use_dr:
        # cross-fitted nuisance on the training split
        n = len(dtr['arm'])
        rng = np.random.default_rng(seed + 11)
        perm = rng.permutation(n)
        folds = [perm[: n // 2], perm[n // 2:]]
        nu = {k: np.zeros((n,) + v.shape[1:]) for k, v in [('rmst', np.zeros((1, cfg['K']))), ('p12', np.zeros((1, cfg['K']))),
                                                         ('prop', np.zeros((1, cfg['K']))), ('Sc', np.zeros((1, J)))]}
        for f in range(2):
            tr_i, ho_i = folds[1 - f], folds[f]
            pf, _ = twin.fit(copy.deepcopy(p0), subset(dtr, tr_i), dva, cfg, opts, seed=seed + 20 + f, **fk)
            pr = twin.predict(pf, subset(dtr, ho_i), cfg, opts)
            for k in nu:
                nu[k][ho_i] = pr[k]
        phi_r, phi_y = twin.pseudo_outcomes(dtr, nu)
        ftr = twin.predict(p, dtr, cfg, opts)
        out['_dr'] = dict(phi_r=phi_r, phi_y=phi_y, nu=nu, ftr={k: ftr[k] for k in ('rmst', 'p12', 's', 'prop')})
        if use_dr == 'mlp':
            pdr = twin.init_dr(jax.random.PRNGKey(seed + 5), cfg)
            pdr = twin.fit_dr(pdr, ftr['s'], ftr['rmst'], ftr['p12'], phi_r, phi_y, cfg, seed=seed)
        for k in out:
            if k == '_dr':
                continue
            if use_dr == 'mlp':
                r, y = twin.predict_dr(pdr, out[k]['s'], out[k]['rmst'], out[k]['p12'], cfg['K'])
            else:   # overlap-restricted additive DR shift per arm (pi_hat >= 0.05)
                sh_r = np.array([(phi_r[m, a] - ftr['rmst'][m, a]).mean() for a in range(cfg['K'])
                                 for m in [nu['prop'][:, a] >= 0.05]])
                sh_y = np.array([(phi_y[m, a] - ftr['p12'][m, a]).mean() for a in range(cfg['K'])
                                 for m in [nu['prop'][:, a] >= 0.05]])
                r = np.clip(out[k]['rmst'] + sh_r, 0, None)
                y = np.clip(out[k]['p12'] + sh_y, 0, 1)
            out[k]['rmst_dr'], out[k]['p12_dr'] = r, y
    else:
        for k in out:
            out[k]['rmst_dr'], out[k]['p12_dr'] = out[k]['rmst'], out[k]['p12']
    for k in out:   # drop heavy arrays
        for big in ('S_os', 'S_pfs', 'z'):
            out[k].pop(big, None)
    if return_params:
        return out, p, opts
    return out


def _concat(a, b):
    out = {}
    for k, v in a.items():
        if isinstance(v, dict):
            out[k] = {kk: np.concatenate([vv, b[k][kk]]) for kk, vv in v.items()}
        elif isinstance(v, np.ndarray) and k in b and isinstance(b[k], np.ndarray) and v.ndim >= 1 and b[k].shape[1:] == v.shape[1:] and len(v) == len(a['arm']):
            out[k] = np.concatenate([v, b[k]])
        else:
            out[k] = v
    return out
