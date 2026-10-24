"""Shared helpers for the LungTwinSim experiments: cohorts, splits and metrics."""
import numpy as np
from lungsim import simulate, SITES, TRAIN_SITES, EXT_SITES, Proj
from data_prep import from_sim, subset, TAU, J
from conformal import treatment_sets, decide, evaluate_sets

N_TRAIN_SITE = 3000


def make_cohort(seed, gamma_u=0.3, L=16):
    proj = Proj(7)
    for s in TRAIN_SITES:
        SITES[s]['n'] = N_TRAIN_SITE
    D = simulate(seed, gamma_u=gamma_u, proj=proj)
    d = from_sim(D, L=L)
    # unlabeled archive from the training sites (outcomes never used)
    for s in TRAIN_SITES:
        SITES[s]['n'] = 1500
    Du = simulate(seed + 1000, gamma_u=gamma_u, sites=TRAIN_SITES, proj=proj)
    for s in TRAIN_SITES:
        SITES[s]['n'] = N_TRAIN_SITE
    du = from_sim(Du, L=L)
    rng = np.random.default_rng(seed)
    it = np.where(np.isin(d['site'], TRAIN_SITES))[0]
    rng.shuffle(it)
    n = len(it)
    a, b, c = int(.55 * n), int(.65 * n), int(.85 * n)
    sp = dict(train=it[:a], val=it[a:b], cal=it[b:c], test=it[c:])
    for s in EXT_SITES:
        sp['ext_' + s] = np.where(d['site'] == s)[0]
    return d, du, sp, D


def policy_value(truth, rec, fallback):
    r = np.where(rec >= 0, rec, fallback)
    return truth[np.arange(len(r)), r].mean()


def cate_metrics(pred, truth, ref=1, mask=None):
    """sqrt-PEHE of arm contrasts against a reference arm (RMST, months)."""
    m = np.ones(len(truth), bool) if mask is None else mask
    dp = pred - pred[:, [ref]]
    dt = truth - truth[:, [ref]]
    K = truth.shape[1]
    return np.array([np.sqrt(((dp[m, a] - dt[m, a]) ** 2).mean()) for a in range(K) if a != ref])


def overlap_mask(pi, eps=0.05):
    return pi >= eps


def eval_method(pr_cal, pr_te, dcal, dte, rmst_key='rmst', p_key='p12', alpha=0.1, prop_true=None, simultaneous=False,
                weighted=True, label_cond=True, use_g=True):
    """All selection and conformal metrics for one method on one evaluation set (simulation)."""
    T = dte['mu_true']            # noise-free conditional RMST (CATE / value truth)
    Y = dte['truth_y']            # realised durable-control potential outcomes (conformal target)
    mu = pr_te[rmst_key]
    out = {}
    pi_true = dte['prop_true']
    ov = pi_true >= 0.05
    out['pehe_all'] = cate_metrics(mu, T).mean()
    # PEHE restricted to (patient, arm) pairs with overlap: contrast against arm 1 when both overlap
    pe = []
    for a in [0, 2, 3]:
        m = ov[:, a] & ov[:, 1]
        if m.sum() > 20:
            pe.append(np.sqrt((((mu[m, a] - mu[m, 1]) - (T[m, a] - T[m, 1])) ** 2).mean()))
    out['pehe_ov'] = float(np.mean(pe))
    naive = mu.argmax(1)
    obs = dte['arm']
    out['v_oracle'] = T.max(1).mean()
    out['v_obs'] = T[np.arange(len(T)), obs].mean()
    out['v_naive'] = T[np.arange(len(T)), naive].mean()
    pi_hat = pr_te['prop']
    restr = np.where(pi_hat >= 0.05, mu, -np.inf).argmax(1)
    out['v_overlap'] = T[np.arange(len(T)), restr].mean()
    out['acc_best'] = (naive == T.argmax(1)).mean()
    out['C_os'] = cindex(dte['os_t'], dte['os_e'], -mu[np.arange(len(T)), obs])
    gc = pr_cal['g12'] if use_g else np.ones(len(dcal['arm']))
    gt = pr_te['g12'] if use_g else np.ones(len(dte['arm']))
    cal = dict(p=pr_cal[p_key], pi=pr_cal['prop'], g12=gc, arm=dcal['arm'], y=dcal['y'], y_obs=dcal['y_obs'])
    te = dict(p=pr_te[p_key], pi=pr_te['prop'], g12=gt)
    excl, assess, _ = treatment_sets(cal, te, alpha=alpha, simultaneous=simultaneous, weighted=weighted, label_cond=label_cond)
    ev = evaluate_sets(excl, assess, Y)
    out.update({'fer_' + str(a): ev['fer'][a] for a in range(len(ev['fer']))})
    out['fer_max'] = ev['fer'].max()
    out['fwer'] = ev['fwer']
    out['power'] = ev['power'].mean()
    out['set_size'] = ev['size']
    rec, reason = decide(excl, assess, mu, 'defer')
    keep = rec >= 0
    out['defer'] = 1 - keep.mean()
    out['defer_pos'] = (reason == 1).mean()
    out['defer_fut'] = (reason == 2).mean()
    out['v_conf_hybrid'] = policy_value(T, rec, obs)               # deferred -> clinician choice
    out['regret_conf_kept'] = (T.max(1)[keep] - T[np.arange(len(T))[keep], rec[keep]]).mean() if keep.any() else np.nan
    out['regret_naive'] = (T.max(1) - T[np.arange(len(T)), naive]).mean()
    out['regret_obs_kept'] = (T.max(1)[keep] - T[np.arange(len(T))[keep], obs[keep]]).mean() if keep.any() else np.nan
    out['regret_naive_kept'] = (T.max(1)[keep] - T[np.arange(len(T))[keep], naive[keep]]).mean() if keep.any() else np.nan
    rec2, _ = decide(excl, assess, mu, 'fallback')
    out['v_conf_fallback'] = policy_value(T, rec2, obs)
    # success-probability calibration for the factual arm and counterfactual arms
    p = pr_te[p_key]
    out['brier_cf'] = ((p - Y) ** 2).mean()
    out['ece_cf'] = ece(p.ravel(), Y.ravel())
    out['p_rmse'] = np.sqrt(((p - dte['p_true']) ** 2).mean())
    return out, dict(excl=excl, assess=assess, rec=rec, reason=reason)


def ece(p, y, bins=10):
    e = np.clip((p * bins).astype(int), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = e == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return tot


def cindex(t, e, risk):
    from sksurv.metrics import concordance_index_censored
    return concordance_index_censored(e.astype(bool), t, risk)[0]
