"""Label-conditional, propensity- and censoring-weighted conformal treatment sets.

For arm a the calibration pool is {i : A_i = a, success observed (Y_i = 1)}.
Score s(x) = -p_a(x) (low predicted durable-control probability = nonconforming).
Weights w(x) = 1 / (pi_a(x) G(12 | x)) correct the shift from
P(x | A = a, C >= 12, Y(a) = 1) to P(x | Y(a) = 1).
Arm a is excluded for x when s(x) exceeds the weighted (1 - alpha_a) quantile.
If the normalised test weight alone exceeds alpha_a, the quantile is +inf: the
arm cannot be excluded and is flagged as not assessable (positivity failure).
"""
import numpy as np


def weighted_threshold(cal_s, cal_w, test_w, alpha):
    o = np.argsort(cal_s)
    s, w = cal_s[o], cal_w[o]
    cw = np.cumsum(w)
    W = cw[-1] if len(cw) else 0.0
    need = (1 - alpha) * (W + test_w)
    k = np.searchsorted(cw, need, side='left')
    thr = np.where(k < len(s), s[np.minimum(k, len(s) - 1)] if len(s) else np.inf, np.inf)
    return thr


def treatment_sets(cal, test, alpha=0.1, simultaneous=True, weighted=True, label_cond=True,
                   min_pi=1e-3, min_g=0.05):
    """cal/test: dicts with p (n,K) success prob, pi (n,K) propensity, g12 (n,) censoring survival at 12 m,
    and for cal: arm, y, y_obs. Returns excluded (n,K) bool, assessable (n,K) bool, thresholds."""
    K = test['p'].shape[1]
    a_k = alpha / K if simultaneous else alpha
    n = len(test['p'])
    excl = np.zeros((n, K), bool)
    assess = np.ones((n, K), bool)
    thr_all = np.zeros((n, K))
    for a in range(K):
        if label_cond:
            m = (cal['arm'] == a) & (cal['y_obs'] == 1) & (cal['y'] == 1)
            s_cal = -cal['p'][m, a]
        else:   # marginal two-label set: score = 1 - p(y_true)
            m = (cal['arm'] == a) & (cal['y_obs'] == 1)
            py = np.where(cal['y'][m] == 1, cal['p'][m, a], 1 - cal['p'][m, a])
            s_cal = 1 - py
        if weighted:
            w_cal = 1 / (np.clip(cal['pi'][m, a], min_pi, 1) * np.clip(cal['g12'][m], min_g, 1))
            w_te = 1 / (np.clip(test['pi'][:, a], min_pi, 1) * np.clip(test['g12'], min_g, 1))
        else:
            w_cal = np.ones(m.sum())
            w_te = np.ones(n)
        thr = weighted_threshold(s_cal, w_cal, w_te, a_k)
        thr_all[:, a] = thr
        assess[:, a] = np.isfinite(thr)
        if label_cond:
            excl[:, a] = (-test['p'][:, a] > thr)
        else:
            excl[:, a] = ((1 - test['p'][:, a]) > thr)          # label 1 not in set
    return excl, assess, thr_all


def decide(excl, assess, mu, mode='defer'):
    """Recommendation from conformal sets.

    Gamma = arms not excluded. Candidate = argmax of mu over Gamma.
    mode 'defer'   : defer if candidate is not assessable (positivity) or Gamma empty (futility)
    mode 'fallback': choose best assessable non-excluded arm; defer only if none exists
    Returns rec (n,) with -1 for deferral, and reason codes (0 rec, 1 positivity, 2 futility)."""
    n, K = mu.shape
    inset = ~excl
    m = np.where(inset, mu, -np.inf)
    cand = m.argmax(1)
    empty = ~inset.any(1)
    rec = cand.copy()
    reason = np.zeros(n, int)
    if mode == 'defer':
        pos = ~assess[np.arange(n), cand] & ~empty
        rec[pos] = -1
        reason[pos] = 1
    else:
        m2 = np.where(inset & assess, mu, -np.inf)
        c2 = m2.argmax(1)
        none = ~(inset & assess).any(1)
        rec = np.where(none, -1, c2)
        reason[none & ~empty] = 1
    rec[empty] = -1
    reason[empty] = 2
    return rec, reason


def evaluate_sets(excl, assess, Ytrue):
    """Ytrue (n,K) binary counterfactual durable-control indicators (simulation only)."""
    K = Ytrue.shape[1]
    pos = Ytrue == 1
    fer = [(excl[:, a] & pos[:, a]).sum() / max(pos[:, a].sum(), 1) for a in range(K)]
    power = [(excl[:, a] & ~pos[:, a]).sum() / max((~pos[:, a]).sum(), 1) for a in range(K)]
    any_miss = (excl & pos).any(1).mean()
    return dict(fer=np.array(fer), power=np.array(power), fwer=any_miss,
                size=(~excl).sum(1).mean(), unassess=(~assess).mean(0))
