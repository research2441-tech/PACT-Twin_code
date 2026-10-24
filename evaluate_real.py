"""Evaluation of the MSK-CHORD runs: factual discrimination, durable-control
calibration, off-policy value (AIPW x IPCW, held-out nuisance, bootstrap CI),
conformal checks on the observed arm, and missing-modality stress."""
import sys, glob, pickle, json
import numpy as np
from data_prep import from_chord, subset, TAU, J
from conformal import treatment_sets, decide
from sim_exp import cindex, ece
import twin


def ope_terms(d, nu, pol, clip_p=0.05, clip_g=0.1):
    """Per-patient AIPW x IPCW contributions for RMST_tau under policy pol (n,)."""
    n = len(pol)
    mu = nu['rmst'][np.arange(n), pol]
    pi = np.clip(nu['prop'][np.arange(n), d['arm']], clip_p, 1)
    G = np.clip(twin.g_at(nu['Sc'], np.where(d['os_t'] >= TAU, J, d['os_b'])), clip_g, 1)
    ipcw = d['rmst_done'] * d['rmst_obs'] / G
    return mu + (d['arm'] == pol) / pi * (ipcw - mu)


def boot_ci(x, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    bs = np.array([x[rng.integers(0, n, n)].mean() for _ in range(B)])
    return float(x.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def evaluate_split(R, d, alpha=0.1):
    sp = R['sp']
    dcal, dte = subset(d, sp['cal']), subset(d, sp['test'])
    nu = R['eval_nuisance']['test']
    n = len(dte['arm'])
    obs_terms = ope_terms(dte, nu, dte['arm'])
    res = {}
    for m, P in R['preds'].items():
        pc, pt = P['cal'], P['test']
        r = {}
        r['C_os'] = cindex(dte['os_t'], dte['os_e'], -pt['rmst'][np.arange(n), dte['arm']])
        r['C_pfs'] = cindex(dte['pfs_t'], dte['pfs_e'], -pt['p12'][np.arange(n), dte['arm']])
        ob = dte['y_obs'] > 0
        pf = pt['p12'][np.arange(n), dte['arm']]
        r['ece_y'] = ece(pf[ob], dte['y'][ob])
        r['brier_y'] = float(((pf[ob] - dte['y'][ob]) ** 2).mean())
        mu = pt['rmst']
        naive = mu.argmax(1)
        ov = np.where(pt['prop'] >= 0.05, mu, -np.inf).argmax(1)
        cal = dict(p=pc['p12'], pi=pc['prop'], g12=pc['g12'], arm=dcal['arm'], y=dcal['y'], y_obs=dcal['y_obs'])
        te = dict(p=pt['p12'], pi=pt['prop'], g12=pt['g12'])
        excl, assess, _ = treatment_sets(cal, te, alpha=alpha, simultaneous=False)
        rec_fb, reason_fb = decide(excl, assess, mu, 'fallback')
        rec_df, reason_df = decide(excl, assess, mu, 'defer')
        for nm, pol in [('naive', naive), ('overlap', ov), ('conf_fb', np.where(rec_fb >= 0, rec_fb, dte['arm'])),
                        ('conf_defer', np.where(rec_df >= 0, rec_df, dte['arm']))]:
            terms = ope_terms(dte, nu, pol)
            r['V_' + nm] = boot_ci(terms)[0]
            r['dV_' + nm] = boot_ci(terms - obs_terms)
            r['agree_' + nm] = float((pol == dte['arm']).mean())
            r['dist_' + nm] = np.bincount(pol, minlength=d['K']).tolist()
        r['V_obs'] = float(obs_terms.mean())
        r['defer'] = float((rec_df < 0).mean())
        r['defer_pos'] = float((reason_df == 1).mean())
        r['defer_fut'] = float((reason_df == 2).mean())
        r['set_size'] = float((~excl).sum(1).mean())
        r['unassess'] = (~assess).mean(0).tolist()
        # label-conditional exclusion on the observed arm: patients who achieved durable control
        a = dte['arm']
        succ = (dte['y_obs'] > 0) & (dte['y'] > 0)
        ex_obs = excl[np.arange(n), a]
        w = 1 / (np.clip(pt['prop'][np.arange(n), a], 1e-3, 1) * np.clip(pt['g12'], .05, 1))
        r['fer_obs_raw'] = float(ex_obs[succ].mean())
        r['fer_obs_w'] = float((w[succ] * ex_obs[succ]).sum() / w[succ].sum())
        r['n_succ'] = int(succ.sum())
        # stress sets
        for k in P:
            if k.startswith('test__'):
                ps = P[k]
                r['C_os__' + k[6:]] = cindex(dte['os_t'], dte['os_e'], -ps['rmst'][np.arange(n), dte['arm']])
                r['flip__' + k[6:]] = float((ps['rmst'].argmax(1) != naive).mean())
                r['flipov__' + k[6:]] = float((np.where(ps['prop'] >= .05, ps['rmst'], -np.inf).argmax(1) != ov).mean())
        res[m] = r
    return res


def main():
    R0 = pickle.load(open('data/chord_cohort.pkl', 'rb'))
    d = from_chord(R0)
    out = {}
    for f in sorted(glob.glob('runs/real_*.pkl')):
        R = pickle.load(open(f, 'rb'))
        out[R['tag']] = evaluate_split(R, d)
    json.dump(out, open('runs/real_eval.json', 'w'), indent=1, default=float)
    return out


if __name__ == '__main__':
    out = main()
    for tag, res in out.items():
        print('==', tag)
        for m, r in res.items():
            print(m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                      if k in ('C_os', 'C_pfs', 'ece_y', 'V_obs', 'V_naive', 'V_overlap', 'V_conf_fb', 'defer', 'set_size',
                               'fer_obs_raw', 'fer_obs_w', 'agree_conf_fb')})
