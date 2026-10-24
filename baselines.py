"""Comparison methods for counterfactual treatment selection.

Tabular (flattened multimodal features + availability + event summaries):
  CoxT      per-arm Cox proportional hazards (ridge)
  RSFT      per-arm random survival forest
  DRGB      DR-learner: cross-fitted Cox outcome/censoring + logistic propensity,
            gradient-boosted regression of AIPW x IPCW pseudo-outcomes
Neural (discrete-time hazards, same bins as the twin):
  SMLP      S-learner MLP with the arm as an input (association model)
  TARNet    shared trunk + arm-specific heads
  CFR       TARNet + linear-MMD balancing penalty on the representation
  CRN       recurrent event encoder + gradient-reversal balancing (longitudinal)
"""
import numpy as np
import jax, jax.numpy as jnp
import optax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from data_prep import flat, J, BW, TAU
import twin

EDGES = BW * np.arange(1, J + 1)


def _surv_at(fns):
    return np.stack([np.clip(f(EDGES), 0, 1) if True else None for f in fns])


def _sf_eval(model, X):
    fns = model.predict_survival_function(X)
    out = np.zeros((len(X), J))
    for i, f in enumerate(fns):
        x = np.clip(EDGES, f.x[0], f.x[-1])
        out[i] = f(x)
        out[i, EDGES < f.x[0]] = 1.0
    return out


def _rmst(S):
    Sp = np.concatenate([np.ones((len(S), 1)), S[:, :-1]], 1)
    return (BW * (Sp + S) / 2).sum(1)


def _fit_surv(kind, X, t, e, seed):
    y = Surv.from_arrays(e.astype(bool), np.maximum(t, 1e-3))
    if kind == 'cox':
        m = CoxPHSurvivalAnalysis(alpha=10.0, ties='efron', n_iter=200)
    else:
        m = RandomSurvivalForest(n_estimators=100, min_samples_leaf=25, max_features='sqrt', n_jobs=2, random_state=seed)
    m.fit(X, y)
    return m


class TabularT:
    """Per-arm survival models for OS and PFS; logistic propensity; Cox censoring."""
    def __init__(self, kind='cox', seed=0):
        self.kind, self.seed = kind, seed

    def fit(self, d):
        X = flat(d)
        self.sc = StandardScaler().fit(X)
        X = self.sc.transform(X)
        K = d['K']
        self.os, self.pfs = [], []
        for a in range(K):
            m = d['arm'] == a
            self.os.append(_fit_surv(self.kind, X[m], d['os_t'][m], d['os_e'][m], self.seed))
            self.pfs.append(_fit_surv(self.kind, X[m], d['pfs_t'][m], d['pfs_e'][m], self.seed))
        self.prop = LogisticRegression(C=0.5, max_iter=2000).fit(X, d['arm'])
        self.cens = _fit_surv('cox', X, d['os_t'], 1 - d['os_e'], self.seed)
        return self

    def predict(self, d):
        X = self.sc.transform(flat(d))
        K = len(self.os)
        So = np.stack([_sf_eval(self.os[a], X) for a in range(K)], 1)
        Sp = np.stack([_sf_eval(self.pfs[a], X) for a in range(K)], 1)
        Sc = _sf_eval(self.cens, X)
        rm = np.stack([_rmst(So[:, a]) for a in range(K)], 1)
        return dict(rmst=rm, p12=Sp[:, :, 5], prop=self.prop.predict_proba(X), Sc=Sc, g12=Sc[:, 5])


class DRGB:
    """DR-learner with cross-fitted tabular nuisance and boosted second stage."""
    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, d):
        from data_prep import subset
        n = len(d['arm'])
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        folds = [perm[: n // 2], perm[n // 2:]]
        K = d['K']
        nu = dict(rmst=np.zeros((n, K)), p12=np.zeros((n, K)), prop=np.zeros((n, K)), Sc=np.zeros((n, J)))
        for f in range(2):
            m = TabularT('cox', self.seed).fit(subset(d, folds[1 - f]))
            pr = m.predict(subset(d, folds[f]))
            for k in nu:
                nu[k][folds[f]] = pr[k]
        phi_r, phi_y = twin.pseudo_outcomes(d, nu)
        self.base = TabularT('cox', self.seed).fit(d)
        X = self.base.sc.transform(flat(d))
        self.gr = [HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                                 random_state=self.seed).fit(X, phi_r[:, a]) for a in range(K)]
        self.gy = [HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                                 random_state=self.seed).fit(X, phi_y[:, a]) for a in range(K)]
        return self

    def predict(self, d):
        b = self.base.predict(d)
        X = self.base.sc.transform(flat(d))
        b['rmst'] = np.clip(np.stack([g.predict(X) for g in self.gr], 1), 0, TAU)
        b['p12'] = np.clip(np.stack([g.predict(X) for g in self.gy], 1), 0, 1)
        return b


# ------------------------------------------------------------------ neural baselines
def _init_nn(key, din, d, K, kind, cfg=None):
    ks = iter(jax.random.split(key, 50))
    D = twin._dense
    p = {'t1': D(next(ks), din + (K if kind == 'smlp' else 0), d), 't2': D(next(ks), d, d)}
    if kind == 'smlp':
        p['o_os'], p['o_pfs'] = D(next(ks), d, J, 0.5), D(next(ks), d, J, 0.5)
        p['o_os']['b'] = jnp.full(J, -2.5); p['o_pfs']['b'] = jnp.full(J, -2.5)
    else:
        def armhead(out):
            return {'W1': jax.random.normal(next(ks), (K, d, d)) * np.sqrt(1 / d), 'b1': jnp.zeros((K, d)),
                    'W2': jax.random.normal(next(ks), (K, d, out)) * np.sqrt(1 / d) * 0.5, 'b2': jnp.full((K, out), -2.5)}
        p['h_os'], p['h_pfs'] = armhead(J), armhead(J)
    p['prop'] = D(next(ks), d, K)
    p['cens'] = D(next(ks), d, J, 0.5)
    p['cens']['b'] = jnp.full(J, -3.0)
    if kind == 'crn':
        p['ev_W'] = jax.random.normal(next(ks), (cfg['evdim'] + cfg['n_types'], d)) * 0.1
        p['gru'] = {n: D(next(ks), 2 * d, d) for n in ('z', 'r', 'h')}
        p['merge'] = D(next(ks), 2 * d, d)
    return p


@jax.custom_vjp
def grad_rev(x, lam):
    return x


def _gr_fwd(x, lam):
    return x, lam


def _gr_bwd(lam, g):
    return (-lam * g, 0.0)


grad_rev.defvjp(_gr_fwd, _gr_bwd)


def _rep(p, X, b, kind, cfg):
    h = jax.nn.gelu(twin.dense(p['t1'], X))
    h = jax.nn.gelu(twin.dense(p['t2'], h))
    if kind == 'crn':
        oh = jax.nn.one_hot(b['ev_ty'], cfg['n_types'])
        xe = jnp.concatenate([b['ev_x'], oh], -1) @ p['ev_W']
        def step(hc, inp):
            x, m = inp
            hn = twin.gru(p['gru'], hc, x)
            return jnp.where(m[:, None] > 0, hn, hc), None
        hseq, _ = jax.lax.scan(step, h, (jnp.swapaxes(xe, 0, 1), b['ev_m'].T))
        h = jax.nn.gelu(twin.dense(p['merge'], jnp.concatenate([h, hseq], -1)))
    return h


def _nn_out(p, X, b, arm_oh, kind, cfg):
    if kind != 'smlp':
        r = _rep(p, X, b, kind, cfg)
    if kind == 'smlp':
        K = arm_oh.shape[1]
        outs_os, outs_pfs = [], []
        for a in range(K):
            oh = jnp.zeros((X.shape[0], K)).at[:, a].set(1.0)
            h = jax.nn.gelu(twin.dense(p['t1'], jnp.concatenate([X, oh], -1)))
            h = jax.nn.gelu(twin.dense(p['t2'], h))
            outs_os.append(twin.dense(p['o_os'], h)); outs_pfs.append(twin.dense(p['o_pfs'], h))
        os_, pfs_ = jnp.stack(outs_os, 1), jnp.stack(outs_pfs, 1)
        r = jax.nn.gelu(twin.dense(p['t2'], jax.nn.gelu(twin.dense(p['t1'], jnp.concatenate([X, arm_oh], -1)))))
    else:
        os_, pfs_ = twin.armheads(p['h_os'], r), twin.armheads(p['h_pfs'], r)
    return r, os_, pfs_


def _mmd(r, arm_oh):
    mu_all = r.mean(0)
    cnt = arm_oh.sum(0) + 1e-3
    mus = (arm_oh.T @ r) / cnt[:, None]
    return (((mus - mu_all) ** 2).sum(1) * (cnt > 5)).sum()


class NeuralCATE:
    def __init__(self, kind='tarnet', d=64, seed=0, lam_bal=1.0):
        self.kind, self.d, self.seed, self.lam = kind, d, seed, lam_bal

    def fit(self, dtr, dva, epochs=60, bs=128, lr=7e-4, patience=6):
        self.sc = StandardScaler().fit(flat(dtr))
        self.cfg = dict(evdim=dtr['evdim'], n_types=dtr['n_types'])
        K = dtr['K']
        self.K = K
        p = _init_nn(jax.random.PRNGKey(self.seed), flat(dtr).shape[1], self.d, K, self.kind, self.cfg)
        kind, cfg, lam = self.kind, self.cfg, self.lam

        def loss(p, b, key):
            aoh = jax.nn.one_hot(b['arm'], K)
            r, os_, pfs_ = _nn_out(p, b['X'], b, aoh, kind, cfg)
            r = jax.random.bernoulli(key, 0.8, r.shape) * r / 0.8
            sel = lambda x: (x * aoh[..., None]).sum(1)
            l = twin.surv_nll(sel(os_), b['os_b'], b['os_ev']).mean() + twin.surv_nll(sel(pfs_), b['pfs_b'], b['pfs_ev']).mean()
            rp = grad_rev(r, 0.3) if kind == 'crn' else jax.lax.stop_gradient(r)
            lp = optax.softmax_cross_entropy_with_integer_labels(twin.dense(p['prop'], rp), b['arm']).mean()
            lc = twin.surv_nll(twin.dense(p['cens'], jax.lax.stop_gradient(r)), b['c_b'], b['c_ev']).mean()
            bal = _mmd(r, aoh) if kind == 'cfr' else 0.0
            return l + 0.5 * lp + 0.3 * lc + lam * bal, l

        tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr, weight_decay=1e-3))
        st = tx.init(p)

        @jax.jit
        def upd(p, st, b, key):
            (l, f), g = jax.value_and_grad(loss, has_aux=True)(p, b, key)
            u, st = tx.update(g, st, p)
            return optax.apply_updates(p, u), st

        @jax.jit
        def evl(p, b):
            return loss(p, b, jax.random.PRNGKey(0))[1]
        Btr, Bva = self._batch(dtr), self._batch(dva)
        n = len(dtr['arm'])
        rng = np.random.default_rng(self.seed)
        key = jax.random.PRNGKey(self.seed + 1)
        best, bestp, bad = np.inf, p, 0
        for ep in range(epochs):
            perm = rng.permutation(n)
            for i in range(0, n - bs + 1, bs):
                idx = jnp.asarray(perm[i:i + bs])
                key, sk = jax.random.split(key)
                p, st = upd(p, st, {k: v[idx] for k, v in Btr.items()}, sk)
            v = float(evl(p, Bva))
            if v < best - 1e-4:
                best, bestp, bad = v, p, 0
            else:
                bad += 1
                if bad >= patience:
                    break
        self.p = bestp
        return self

    def _batch(self, d):
        b = {k: jnp.asarray(d[k]) for k in ('arm', 'os_b', 'os_ev', 'pfs_b', 'pfs_ev', 'c_b', 'c_ev', 'ev_x', 'ev_ty', 'ev_m')}
        b['X'] = jnp.asarray(self.sc.transform(flat(d)).astype(np.float32))
        return b

    def predict(self, d):
        b = self._batch(d)
        aoh = jax.nn.one_hot(b['arm'], self.K)
        r, os_, pfs_ = _nn_out(self.p, b['X'], b, aoh, self.kind, self.cfg)
        So, Sp = np.asarray(twin.surv_from_logits(os_)), np.asarray(twin.surv_from_logits(pfs_))
        Sc = np.asarray(twin.surv_from_logits(twin.dense(self.p['cens'], r)))
        prop = np.asarray(jax.nn.softmax(twin.dense(self.p['prop'], r), -1))
        # all neural baselines share the same logistic propensity model as the tabular ones
        prop = self._lr.predict_proba(self.sc.transform(flat(d)))
        return dict(rmst=np.stack([_rmst(So[:, a]) for a in range(self.K)], 1), p12=Sp[:, :, 5], prop=prop, Sc=Sc, g12=Sc[:, 5])

    def fit_all(self, dtr, dva):
        self.fit(dtr, dva)
        self._lr = LogisticRegression(C=0.5, max_iter=2000).fit(self.sc.transform(flat(dtr)), dtr['arm'])
        return self


def guideline_policy(d, sim=True):
    """Rule-based first-line policy (simplified guideline logic).
    Driver known -> TKI; PD-L1 >= 50% and PS <= 1 -> IO-mono; PS <= 1 -> Chemo-IO;
    PS >= 2 -> IO-mono if PD-L1 >= 50% else Chemo."""
    n = len(d['arm'])
    gen, av = d['mods']['gen'], d['av']
    names = d['mod_names']
    known = ((av[:, names.index('gen')] > 0) & ((gen[:, 0] > 0))) | (d['mods']['ctdna'][:, 1] > 0)
    pd_hi = (av[:, names.index('pdl1')] > 0) & (d['mods']['pdl1'][:, 2] > 0)
    st = d['static']
    ps2 = (st[:, 7] + st[:, 8]) > 0
    rec = np.where(ps2, np.where(pd_hi, 2, 0), np.where(pd_hi, 2, 1))
    rec = np.where(known, 3, rec)
    return rec
