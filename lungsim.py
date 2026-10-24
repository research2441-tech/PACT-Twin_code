"""LungTwinSim: semi-synthetic multi-site advanced NSCLC cohort with exact
counterfactual outcomes under four first-line strategies.

Time unit: months. Decision (first-line start) is t = 0. Tumour burden is the
log sum of longest diameters (SLD, mm). Each arm a has a closed-form burden
trajectory; death times under every arm share one exponential draw so the
potential outcomes are coupled (common random numbers).

Arms: 0 Chemo, 1 Chemo-IO, 2 IO-mono, 3 TKI.
Modalities (frozen foundation-encoder emulations): ct(32), pet(8), wsi(32),
gen(6), rna(2), ctdna(2), pdl1(3).
"""
import numpy as np

ARMS = ['Chemo', 'Chemo-IO', 'IO-mono', 'TKI']
MODS = {'ct': 32, 'pet': 8, 'wsi': 32, 'gen': 6, 'rna': 2, 'ctdna': 2, 'pdl1': 3}
EV = ['dx', 'ct', 'pet', 'labs', 'ctdna', 'biopsy', 'pdl1', 'ngs', 'rna', 'treat', 'prog']
EVDIM = 34
GRID = np.arange(0, 60.0001, 0.25)        # months
SCANS = np.arange(2, 60.0001, 2.0)        # response assessment every 2 months

SITES = {  # asian, never-smoker boost, ngs rate, ngs turnaround (d), io adoption,
           # pet, wsi, ctdna, rna rates, age shift, acquisition shift scale, dropout
    'A': dict(asian=.08, ngs=.80, tat=18, io=1.0, pet=.70, wsi=.80, ctdna=.45, rna=.30, age=0, shift=.0, drop=.010, n=2000),
    'B': dict(asian=.10, ngs=.70, tat=24, io=.9, pet=.60, wsi=.70, ctdna=.35, rna=.25, age=1, shift=.4, drop=.012, n=2000),
    'C': dict(asian=.15, ngs=.85, tat=16, io=1.1, pet=.75, wsi=.85, ctdna=.55, rna=.40, age=-1, shift=.6, drop=.008, n=2000),
    'D': dict(asian=.05, ngs=.60, tat=28, io=.8, pet=.50, wsi=.60, ctdna=.25, rna=.15, age=2, shift=.8, drop=.015, n=2000),
    'E': dict(asian=.70, ngs=.92, tat=14, io=.9, pet=.80, wsi=.90, ctdna=.60, rna=.40, age=-2, shift=1.0, drop=.010, n=2000),
    'F': dict(asian=.04, ngs=.40, tat=35, io=.7, pet=.40, wsi=.50, ctdna=.15, rna=.05, age=3, shift=1.4, drop=.018, n=2000),
}
TRAIN_SITES, EXT_SITES = ['A', 'B', 'C', 'D'], ['E', 'F']


def sig(x):
    return 1 / (1 + np.exp(-x))


class Proj:
    """Fixed random projections emulating frozen foundation encoders."""
    def __init__(self, seed=7):
        r = np.random.default_rng(seed)
        self.ct = r.normal(0, 1, (9, 32)) / 3
        self.pet = r.normal(0, 1, (3, 8)) / 2
        self.wsi = r.normal(0, 1, (9, 32)) / 3
        self.ct_site = {k: r.normal(0, 1, 32) for k in SITES}
        self.wsi_site = {k: r.normal(0, 1, 32) for k in SITES}
        self.ct_nl = r.normal(0, 1, (32, 32)) / 6
        self.wsi_nl = r.normal(0, 1, (32, 32)) / 6


def burden_traj(b0, g, kc, rc, ki, ri, kt, rt, ramp=1.0):
    """log-SLD on GRID for one arm given kill rates (vectorised over patients)."""
    t = GRID[None, :]
    out = b0[:, None] + g[:, None] * t
    out -= kc[:, None] * (1 - np.exp(-rc[:, None] * t)) / rc[:, None]
    rr = ri[:, None] + 1 / ramp
    out -= ki[:, None] * ((1 - np.exp(-ri[:, None] * t)) / ri[:, None] - (1 - np.exp(-rr * t)) / rr)
    out -= kt[:, None] * (1 - np.exp(-rt[:, None] * t)) / rt[:, None]
    return out


def simulate(seed=0, gamma_u=0.3, sites=None, proj=None, n_scale=1.0):
    rng = np.random.default_rng(seed)
    proj = proj or Proj()
    sites = sites or list(SITES)
    recs = []
    for s in sites:
        P = SITES[s]
        n = int(P['n'] * n_scale)
        recs.append(_site(rng, s, P, n, gamma_u, proj))
    D = {}
    for k in recs[0]:
        if isinstance(recs[0][k], list):
            D[k] = sum((r[k] for r in recs), [])
        else:
            D[k] = np.concatenate([r[k] for r in recs])
    return D


def _site(rng, s, P, n, gamma_u, proj):
    N = n
    age = rng.normal(67 + P['age'], 9, N).clip(35, 90)
    male = rng.random(N) < .52
    asian = rng.random(N) < P['asian']
    pnever = sig(-1.6 + 1.5 * asian + 0.7 * (~male))
    never = rng.random(N) < pnever
    current = (~never) & (rng.random(N) < .35)
    pack = np.where(never, 0, rng.gamma(4, 9, N))
    lusc = (~never) & (rng.random(N) < sig(-1.2 + 0.015 * pack))
    other = (~lusc) & (rng.random(N) < .08)
    luad = ~(lusc | other)
    egfr = luad & (rng.random(N) < sig(-2.4 + 1.9 * never + 1.3 * asian + 0.3 * (~male)))
    alk = luad & (~egfr) & (rng.random(N) < sig(-3.6 + 1.6 * never - 0.03 * (age - 67)))
    kras = luad & (~egfr) & (~alk) & (rng.random(N) < sig(-0.9 - 1.5 * never))
    stk11 = (~egfr) & (~alk) & (rng.random(N) < np.where(kras, .30, .07))
    tp53 = rng.random(N) < np.where(never, .35, .6)
    drv = egfr | alk
    ltmb = rng.normal(1.6 + 0.9 * (~never) + 0.004 * pack - 0.8 * drv, 0.6)
    tmb = np.exp(ltmb)
    tmbz = (ltmb - 2.2) / 0.8
    zimm = rng.normal(0, 1, N) + 0.35 * tmbz - 0.6 * drv - 0.5 * stk11
    pdl1 = 100 * sig(rng.normal(-1.0 + 1.2 * zimm, 1.3))
    pdz = (pdl1 - 30) / 30
    lg = rng.normal(np.log(0.11), 0.45, N) + 0.15 * tp53 + 0.1 * stk11
    g = np.exp(lg)
    gz = (lg - np.log(0.11)) / 0.45
    b0 = rng.normal(np.log(75), 0.45, N)
    liver = rng.random(N) < sig(-1.9 + 0.4 * gz)
    brain = rng.random(N) < sig(-1.4 + 0.6 * drv)
    bone = rng.random(N) < sig(-0.7 + 0.2 * gz)
    frail = rng.normal(0, 1, N)
    u = rng.normal(0, 1, N)                                    # hidden confounder
    ps_lat = 0.8 * frail + 0.5 * (b0 - np.log(75)) / .45 + 0.03 * (age - 67) + rng.normal(0, .7, N)
    ps = np.digitize(ps_lat, [-0.4, 1.0, 2.2])                 # 0..3

    # ---------- potential outcomes ----------
    one = np.ones(N)
    kc = 0.21 + 0.8 * g
    rc = 0.20 * one
    rim = sig(-0.4 + 1.5 * zimm + 0.6 * pdz + 0.35 * tmbz - 1.4 * stk11 - 1.8 * drv)
    ki = 0.34 * rim
    ri = 0.012 * one
    kt = np.where(egfr | alk, 0.33, 0.03)
    rt = np.where(alk, 0.045, 0.06)
    z = 0 * one
    trajs = np.stack([
        burden_traj(b0, g, kc, rc, z, ri, z, rt),
        burden_traj(b0, g, kc, rc, 0.75 * ki, ri, z, rt),
        burden_traj(b0, g, z, rc, ki, ri, z, rt),
        burden_traj(b0, g, z, rc, z, ri, kt, rt)], 1)          # N,4,G
    tox_scale = np.array([1.0, 1.3, 0.2, 0.25])
    tox = (0.45 * np.clip(frail, 0, None) + 0.25 * (age - 67) / 10 + 0.3 * (ps >= 2))[:, None] * tox_scale[None, :]
    tox_t = np.exp(-GRID / 6.0)[None, None, :] * tox[:, :, None]
    lin = (0.95 * (trajs - np.log(75)) + 0.35 * ps[:, None, None] + 0.25 * ((age - 67) / 10)[:, None, None]
           + 0.35 * liver[:, None, None] + 0.25 * brain[:, None, None] + gamma_u * u[:, None, None] + tox_t)
    haz = 0.032 * np.exp(lin)
    cum = np.cumsum(haz * 0.25, axis=2)
    Sgrid = np.exp(-cum)                                        # N,4,G survival given latent state
    i36, i12 = int(36 / 0.25), int(12 / 0.25)
    rmst_true = (np.concatenate([np.ones((N, 4, 1)), Sgrid[:, :, :i36 - 1]], 2) * 0.25).sum(2)
    E = rng.exponential(1, N)
    died = cum >= E[:, None, None]
    idx = np.where(died.any(2), died.argmax(2), len(GRID) - 1)
    T = GRID[idx] + 0.25 * rng.random((N, 1))
    T = np.where(died.any(2), T, 60.0 + 1e-3)
    # progression from scans
    sidx = (SCANS / 0.25).astype(int)
    bs = trajs[:, :, sidx]                                     # N,4,S
    base = trajs[:, :, :1]
    nadir = np.minimum.accumulate(np.concatenate([base, bs], 2), axis=2)[:, :, :-1]
    progm = bs >= nadir + np.log(1.2)
    pt = np.where(progm.any(2), SCANS[progm.argmax(2)], np.inf)
    PFS = np.minimum(pt, T)
    shrink = (bs - base)
    alive_scan = SCANS[None, None, :] <= np.minimum(T, pt)[:, :, None]
    best = np.where(alive_scan, shrink, np.inf).min(2)
    resp = (best <= np.log(0.7)).astype(int)
    p12_true = (pt >= 12) * Sgrid[:, :, i12]

    # ---------- modality availability ----------
    ngs_order = rng.random(N) < np.clip(P['ngs'] * (1 + 0.25 * luad + 0.2 * never - 0.35 * lusc), 0, .99)
    dx_lag = rng.uniform(1.0, 2.5, N)                          # months dx -> L1
    t_biopsy = -dx_lag + 0.25
    ngs_t = t_biopsy + rng.gamma(4, P['tat'] / 4, N) / 30.4
    gen_av = ngs_order & (ngs_t < -0.1)
    wsi_av = rng.random(N) < P['wsi']
    pdl1_av = rng.random(N) < .85
    pdl1_t = t_biopsy + rng.uniform(0.2, 0.6, N)
    pdl1_av = pdl1_av & (pdl1_t < 0)
    ctdna_av = rng.random(N) < P['ctdna']
    rna_av = rng.random(N) < P['rna']
    pet_av = rng.random(N) < P['pet']
    ct_av = np.ones(N, bool)
    drv_ct = ctdna_av & drv & (rng.random(N) < .7)             # driver detected in plasma

    # ---------- treatment policy (clinician) ----------
    pd_cat = np.digitize(pdl1, [1, 50])
    known_drv = (gen_av & drv) | drv_ct
    L = np.zeros((N, 4))
    L[:, 0] = 0.3 + 0.6 * (ps >= 2) + 0.03 * (age - 67) - 1.0 * P['io'] + 0.4 * (~pdl1_av)
    L[:, 1] = 1.0 * P['io'] + 0.5 * (pd_cat == 1) * pdl1_av + 0.3 * (gz > 0.5) - 0.5 * (ps >= 2)
    L[:, 2] = -0.7 + 1.0 * P['io'] + 1.6 * (pd_cat == 2) * pdl1_av + 0.6 * (ps >= 2) + 0.02 * (age - 67) - 0.6 * (~pdl1_av)
    L[:, 3] = -5.0 + 7.5 * known_drv + 1.6 * (never & asian & luad & ~gen_av & ~drv_ct) + 0.6 * (never & luad & ~gen_av)
    L[known_drv, :3] -= 2.5
    L[:, 0] += 0.5 * gamma_u * u                               # sicker-looking (hidden) -> chemo
    L[:, 2] -= 0.3 * gamma_u * u
    prop = np.exp(L - L.max(1, keepdims=True))
    prop /= prop.sum(1, keepdims=True)
    arm = np.array([rng.choice(4, p=p_) for p_ in prop])

    # ---------- censoring ----------
    admin = rng.uniform(8, 60, N)
    drop = rng.exponential(1 / (P['drop'] * np.exp(0.15 * ps)), N)
    C = np.minimum(admin, drop)
    Ta = T[np.arange(N), arm]
    Pa = PFS[np.arange(N), arm]
    os_t, os_e = np.minimum(Ta, C), (Ta <= C).astype(int)
    pfs_t, pfs_e = np.minimum(Pa, C), (Pa <= C).astype(int)

    # ---------- observed modalities at decision ----------
    def ct_emb(b, noise=0.35):
        feats = np.stack([b - np.log(75), liver, bone, brain, gz, lusc, luad, rng.normal(0, 1, N), rng.normal(0, 1, N)], 1)
        e = feats @ proj.ct
        e = np.tanh(e + e @ proj.ct_nl) + P['shift'] * 0.25 * proj.ct_site[s] + rng.normal(0, noise, (N, 32))
        return e
    b_dx = b0 - g * dx_lag
    b_st = b0 - g * 0.4
    sld_dx = b_dx + rng.normal(0, .08, N)
    sld_st = b_st + rng.normal(0, .08, N)
    ct_dx, ct_st = ct_emb(b_dx), ct_emb(b_st)
    pet = np.stack([gz + rng.normal(0, .5, N), b_st - np.log(75), (liver | bone).astype(float)], 1) @ proj.pet + rng.normal(0, .3, (N, 8))
    wf = np.stack([luad, lusc, other, zimm + rng.normal(0, .6, N), gz + rng.normal(0, .7, N), pdz + rng.normal(0, .8, N),
                   0.7 * egfr + 0.4 * alk + rng.normal(0, .5, N), 0.5 * kras + rng.normal(0, .6, N), rng.normal(0, 1, N)], 1)
    wsi = wf @ proj.wsi
    wsi = np.tanh(wsi + wsi @ proj.wsi_nl) + P['shift'] * 0.25 * proj.wsi_site[s] + rng.normal(0, .35, (N, 32))
    gen = np.stack([egfr, alk, kras, stk11, tp53, (ltmb - 2.2) / 0.8], 1).astype(float)
    rna = np.stack([zimm + rng.normal(0, .5, N), gz + rng.normal(0, .5, N)], 1)
    vaf = -4 + 0.9 * (b0 - np.log(75)) + rng.normal(0, .6, N)
    ctd = np.stack([vaf, drv_ct.astype(float)], 1)
    pdl = np.eye(3)[pd_cat]
    labs = lambda b: np.stack([0.4 * (b - np.log(75)) + 0.3 * gz + rng.normal(0, .3, N),
                               -0.3 * frail + rng.normal(0, .3, N), -0.3 * zimm + 0.2 * frail + rng.normal(0, .35, N)], 1)
    lab_dx, lab_pre = labs(b_dx), labs(b_st)

    static = np.stack([(age - 67) / 9, male, asian, never, current, pack / 40, ps == 1, ps == 2, ps >= 3,
                       liver, brain, bone, lab_pre[:, 0], lab_pre[:, 1], lab_pre[:, 2]], 1).astype(float)
    gen_full = gen.copy()
    mods = {'ct': ct_st, 'pet': pet, 'wsi': wsi, 'gen': gen, 'rna': rna, 'ctdna': ctd, 'pdl1': pdl}
    av = {'ct': ct_av, 'pet': pet_av, 'wsi': wsi_av, 'gen': gen_av, 'rna': rna_av, 'ctdna': ctdna_av, 'pdl1': pdl1_av}
    for k in mods:
        mods[k] = np.where(av[k][:, None], mods[k], 0.0)

    # ---------- event sequences (pre-decision, plus post-decision for landmarks) ----------
    def pad(v):
        o = np.zeros(EVDIM); o[:len(v)] = v; return o
    events = []
    # post-decision scans (factual arm) for dynamic prediction
    for i in range(N):
        e = [(-dx_lag[i], 0, pad([1.0])),
             (-dx_lag[i], 1, pad(np.r_[sld_dx[i] - np.log(75), ct_dx[i]])),
             (-dx_lag[i] + 0.05, 3, pad(lab_dx[i]))]
        if wsi_av[i]:
            e.append((t_biopsy[i], 5, pad(wsi[i])))
        if pdl1_av[i]:
            e.append((pdl1_t[i], 6, pad(pdl[i])))
        if ctdna_av[i]:
            e.append((-dx_lag[i] + 0.1, 4, pad(ctd[i])))
        if gen_av[i]:
            e.append((ngs_t[i], 7, pad(gen[i])))
        if rna_av[i]:
            e.append((t_biopsy[i] + 0.3, 8, pad(rna[i])))
        e.append((-0.4, 1, pad(np.r_[sld_st[i] - np.log(75), ct_st[i]])))
        if pet_av[i]:
            e.append((-0.4, 2, pad(pet[i])))
        e.append((-0.1, 3, pad(lab_pre[i])))
        # post decision
        a = arm[i]
        e.append((0.0, 9, pad(np.eye(4)[a])))
        if ngs_order[i] and not gen_av[i]:
            e.append((ngs_t[i], 7, pad(gen[i])))           # late-arriving genomics
        end = min(os_t[i], 24.0)
        for tt in SCANS:
            if tt > end:
                break
            b = trajs[i, a, int(tt / .25)] + rng.normal(0, .08)
            e.append((tt, 1, pad(np.r_[b - np.log(75), _ct_one(proj, s, P, rng, b, liver[i], bone[i], brain[i], gz[i], lusc[i], luad[i])])))
            if ctdna_av[i] and tt <= 4:
                e.append((tt - 1.0, 4, pad([-4 + 0.9 * (trajs[i, a, int((tt - 1) / .25)] - np.log(75)) + rng.normal(0, .6), drv_ct[i]])))
            if pfs_e[i] and abs(tt - pfs_t[i]) < 1e-6 and pfs_t[i] < os_t[i]:
                e.append((tt + 0.01, 10, pad([1.0])))
        e.sort(key=lambda x: x[0])
        events.append(e)

    return dict(site=np.array([s] * N), static=static, **{'m_' + k: v for k, v in mods.items()},
                **{'a_' + k: v.astype(float) for k, v in av.items()},
                events=events, arm=arm, prop=prop, os_t=os_t, os_e=os_e, pfs_t=pfs_t, pfs_e=pfs_e,
                C=C, T=T, PFS=PFS, resp=resp, rmst_true=rmst_true, p12_true=p12_true, egfr=egfr, alk=alk, kras=kras, stk11=stk11, luad=luad, lusc=lusc,
                pdl1=pdl1, zimm=zimm, g=g, u=u, frail=frail, asian=asian, male=male, age=age, ps=ps,
                ngs_order=ngs_order, drv_ct=drv_ct, gen_full=gen_full)


def _ct_one(proj, s, P, rng, b, liver, bone, brain, gz, lusc, luad):
    f = np.array([b - np.log(75), liver, bone, brain, gz, lusc, luad, rng.normal(), rng.normal()])
    e = f @ proj.ct
    return np.tanh(e + e @ proj.ct_nl) + P['shift'] * 0.25 * proj.ct_site[s] + rng.normal(0, .35, 32)


def summarize(D):
    import pandas as pd
    T, PFS = D['T'], D['PFS']
    print('N', len(T), 'arm dist', np.bincount(D['arm']) / len(T))
    print('median OS by arm (all pts)', np.median(T, 0).round(1), ' PFS', np.median(PFS, 0).round(1))
    best = np.argmax(np.minimum(T, 36), 1)
    print('best-arm (RMST36) dist', np.bincount(best, minlength=4) / len(T))
    y = (PFS >= 12).mean(0)
    print('P(PFS>=12) per arm', y.round(3), 'resp', D['resp'].mean(0).round(3))
    for nm, msk in [('EGFR/ALK', D['egfr'] | D['alk']), ('PDL1>=50 no drv', (D['pdl1'] >= 50) & ~(D['egfr'] | D['alk'])),
                    ('STK11', D['stk11'])]:
        print(nm, msk.mean().round(3), 'medOS', np.median(T[msk], 0).round(1), 'P(PFS12)', (PFS[msk] >= 12).mean(0).round(2))
    print('observed OS event rate', D['os_e'].mean().round(3), 'pfs', D['pfs_e'].mean().round(3))
    print('avail', {k: D['a_' + k].mean().round(2) for k in MODS})
    print('prop min per arm', D['prop'].min(0).round(4), 'frac prop<0.02', (D['prop'] < 0.02).mean(0).round(3))
    print('mean events', np.mean([len(e) for e in D['events']]), 'pre', np.mean([sum(1 for x in e if x[0] < 0) for e in D['events']]))


if __name__ == '__main__':
    import time
    t0 = time.time()
    D = simulate(0)
    print('time', time.time() - t0)
    summarize(D)
