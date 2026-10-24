"""Convert simulator or MSK-CHORD dictionaries into model-ready arrays.

All times are converted to months. Survival is discretised into J bins of
BW months up to the horizon TAU (default 18 bins x 2 months = 36 months).
"""
import numpy as np

BW, J = 2.0, 18
TAU = BW * J
H_PFS = 12.0                      # landmark for the binary durable-control endpoint


def discretise(t, e):
    """bin index (0..J) and event flag; t >= TAU -> bin J (alive past horizon)."""
    b = np.minimum((t // BW).astype(int), J)
    ev = (e == 1) & (t < TAU)
    return b, ev.astype(np.float32)


def binary_endpoint(t, e, h=H_PFS):
    """Y = 1{T >= h}; observed if T >= h or event before h."""
    y = (t >= h).astype(np.float32)
    obs = ((t >= h) | (e == 1)).astype(np.float32)
    return y, obs


def pack_events(events, cutoff, L, evdim, n_types):
    N = len(events)
    X = np.zeros((N, L, evdim), np.float32)
    Tm = np.zeros((N, L), np.float32)
    Ty = np.zeros((N, L), np.int32)
    M = np.zeros((N, L), np.float32)
    summ = np.zeros((N, n_types * 3), np.float32)
    for i, ev in enumerate(events):
        c = cutoff[i] if np.ndim(cutoff) else cutoff
        ev = [e for e in ev if e[0] <= c]
        ev = ev[-L:]
        for j, (t, ty, v) in enumerate(ev):
            v = np.asarray(v, np.float32)
            X[i, j, :len(v)] = v[:evdim]
            Tm[i, j] = t - c                       # time relative to prediction point (<= 0)
            Ty[i, j] = ty
            M[i, j] = 1
        for ty in range(n_types):
            sel = [e for e in ev if e[1] == ty]
            if sel:
                summ[i, ty * 3] = np.log1p(len(sel))
                summ[i, ty * 3 + 1] = float(np.asarray(sel[-1][2])[0])
                summ[i, ty * 3 + 2] = (sel[-1][0] - c) / 12.0
    return X, Tm, Ty, M, summ


def from_sim(D, cutoff=-1e-3, L=24):
    from lungsim import MODS, EV, EVDIM
    mods = {k: D['m_' + k].astype(np.float32) for k in MODS}
    av = np.stack([D['a_' + k] for k in MODS], 1).astype(np.float32)
    X, Tm, Ty, M, summ = pack_events(D['events'], cutoff, L, EVDIM, len(EV))
    out = dict(static=D['static'].astype(np.float32), mods=mods, mod_names=list(MODS), av=av,
               ev_x=X, ev_t=Tm, ev_ty=Ty, ev_m=M, ev_summ=summ, n_types=len(EV), evdim=EVDIM,
               arm=D['arm'].astype(np.int32), K=4,
               os_t=D['os_t'], os_e=D['os_e'], pfs_t=D['pfs_t'], pfs_e=D['pfs_e'], C=D['C'])
    # censoring indicator for the censoring model: censored = 1 - os_e
    out.update(_targets(out))
    # ground truth for evaluation
    out['T'] = D['T']; out['PFS'] = D['PFS']; out['resp'] = D['resp']
    out['truth_rmst'] = np.minimum(D['T'], TAU)
    out['truth_y'] = (D['PFS'] >= H_PFS).astype(np.float32)          # realised potential outcomes
    out['mu_true'] = D['rmst_true']                                      # E[min(T(a),36) | latent state]
    out['p_true'] = D['p12_true']
    out['site'] = D['site']; out['prop_true'] = D['prop']
    # phenotype labels (observable only where genomic / pdl1 testing exists)
    ph = np.stack([D['egfr'] | D['alk'], D['kras'], D['stk11'], D['pdl1'] >= 50, D['lusc']], 1).astype(np.float32)
    phm = np.stack([D['a_gen'], D['a_gen'], D['a_gen'], D['a_pdl1'], np.ones(len(ph))], 1).astype(np.float32)
    out['pheno'], out['pheno_m'], out['pheno_true'] = ph, phm, ph.copy()
    for k in ('egfr', 'alk', 'kras', 'stk11', 'pdl1', 'asian', 'male', 'age', 'ps', 'u', 'zimm', 'frail', 'luad'):
        out['g_' + k] = D[k]
    return out


def _targets(o):
    ob, oe = discretise(o['os_t'], o['os_e'])
    pb, pe = discretise(o['pfs_t'], o['pfs_e'])
    cb, ce = discretise(o['os_t'], 1 - o['os_e'])
    y, yobs = binary_endpoint(o['pfs_t'], o['pfs_e'])
    return dict(os_b=ob, os_ev=oe, pfs_b=pb, pfs_ev=pe, c_b=cb, c_ev=ce, y=y, y_obs=yobs,
                rmst_obs=np.minimum(o['os_t'], TAU), rmst_done=((o['os_e'] == 1) | (o['os_t'] >= TAU)).astype(np.float32))


def from_chord(R, cutoff_days=7.0, L=32, arms=(0, 1, 2)):
    """Real-world cohort; keeps only the chosen first-line arms (TKI excluded by default)."""
    keep = np.isin(R['arm'], arms)
    idx = np.where(keep)[0]
    S, A = R['static'].iloc[idx], R['avail'].iloc[idx]
    mod_cols = {
        'genomic': ['tmb', 'msi', 'purity'] + [c for c in S.columns if c.startswith(('amp_', 'del_'))],
        'pdl1': ['pdl1_pos'],
        'radiology': [c for c in S.columns if c.startswith('site_')] + ['n_sites'],
        'cea': ['cea_last', 'cea_slope'],
        'ecog': ['ecog'],
    }
    used = sum(mod_cols.values(), [])
    st_cols = [c for c in S.columns if c not in used]
    st = S[st_cols].astype(float).copy()
    order = R['order'][idx]
    st['era'] = (np.argsort(np.argsort(order)) / len(order) - 0.5) * 2
    mu, sd = st.mean(), st.std().replace(0, 1)
    stv = ((st - mu) / sd).fillna(0).values.astype(np.float32)
    mods, av = {}, []
    for k, cols in mod_cols.items():
        v = S[cols].astype(float)
        a = A[k].values.astype(float) if k in A else v.notna().any(axis=1).values.astype(float)
        if k == 'pdl1':
            v = v * 2 - 1
        m_, s_ = v.mean(), v.std().replace(0, 1)
        vz = ((v - m_) / s_) if k not in ('pdl1',) else v
        vz = vz.fillna(0).values * a[:, None]
        mods[k] = vz.astype(np.float32)
        av.append(a)
    av = np.stack(av, 1).astype(np.float32)
    ev_types = R['ev_types']
    tmap = {t: i for i, t in enumerate(ev_types)}
    events = []
    for i in idx:
        events.append([(t / 30.4375, tmap[ty], v) for (t, ty, v) in R['events'][R['pid'][i]]])
    evdim = 9
    X, Tm, Ty, M, summ = pack_events(events, cutoff_days / 30.4375, L, evdim, len(ev_types))
    arm_map = {a: j for j, a in enumerate(arms)}
    out = dict(static=stv, static_cols=list(st.columns), mods=mods, mod_names=list(mods), av=av,
               ev_x=X, ev_t=Tm, ev_ty=Ty, ev_m=M, ev_summ=summ, n_types=len(ev_types), evdim=evdim,
               arm=np.array([arm_map[a] for a in R['arm'][idx]], np.int32), K=len(arms),
               os_t=R['os_t'][idx] / 30.4375, os_e=R['os_e'][idx], pfs_t=R['pfs_t'][idx] / 30.4375,
               pfs_e=R['pfs_e'][idx], order=order, raw_static=S, raw_avail=A,
               events_raw=events)
    out.update(_targets(out))
    # phenotype heads on real data: PD-L1 positivity, TMB-high (>=10 mut/Mb)
    tmb = np.expm1(S['tmb'].values)
    ph = np.stack([np.nan_to_num(S['pdl1_pos'].values), np.nan_to_num(tmb >= 10), S['lusc'].values], 1).astype(np.float32)
    phm = np.stack([A['pdl1'].values, A['genomic'].values, np.ones(len(S))], 1).astype(np.float32)
    out['pheno'], out['pheno_m'] = ph, phm
    return out


def subset(d, idx):
    out = {}
    n = len(d['arm'])
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = {kk: vv[idx] for kk, vv in v.items()}
        elif isinstance(v, np.ndarray) and v.shape[:1] == (n,):
            out[k] = v[idx]
        elif hasattr(v, 'iloc') and len(v) == n:
            out[k] = v.iloc[idx]
        elif isinstance(v, list) and len(v) == n and k not in ('mod_names', 'static_cols'):
            out[k] = [v[i] for i in idx]
        else:
            out[k] = v
    return out


def flat(d):
    """Flattened feature matrix for tabular baselines."""
    parts = [d['static']] + [d['mods'][k] for k in d['mod_names']] + [d['av'], d['ev_summ']]
    return np.concatenate(parts, 1).astype(np.float32)
