"""Causal lung digital twin (JAX).

Components
  * modality adapters over frozen-encoder embeddings
  * availability-gated mixture-of-experts fusion with masked cross-modal attention
  * event-typed continuous-time state transition (relaxation toward the fused state)
  * arm-specific discrete-time hazard heads (OS, PFS), response head, propensity head
    conditioned on the availability pattern, censoring head, phenotype heads
  * self-supervised pretraining: masked-modality reconstruction + next-event prediction
  * doubly robust potential-outcome heads fitted on cross-fitted pseudo-outcomes
"""
import numpy as np
import jax, jax.numpy as jnp
import optax
from functools import partial
from data_prep import J, BW, TAU

jax.config.update('jax_platform_name', 'cpu')


# ---------------------------------------------------------------- init
def _dense(key, i, o, scale=1.0):
    return {'W': jax.random.normal(key, (i, o)) * scale * np.sqrt(2.0 / (i + o)), 'b': jnp.zeros(o)}


def _ln(d):
    return {'g': jnp.ones(d), 'b': jnp.zeros(d)}


def init(key, cfg):
    d, E, K, M = cfg['d'], cfg['E'], cfg['K'], len(cfg['mod_dims'])
    ks = iter(jax.random.split(key, 200))
    p = {}
    p['static'] = _dense(next(ks), cfg['ds'], d)
    p['mod'] = [{'l1': _dense(next(ks), dm, d), 'l2': _dense(next(ks), d, d)} for dm in cfg['mod_dims']]
    p['mod_emb'] = jax.random.normal(next(ks), (M + 1, d)) * 0.1
    p['att'] = {n: _dense(next(ks), d, d) for n in ('q', 'k', 'v', 'o')}
    p['ln1'], p['ln2'], p['ln3'], p['ln4'] = _ln(d), _ln(d), _ln(d), _ln(d)
    p['ffn'] = {'l1': _dense(next(ks), d, 2 * d), 'l2': _dense(next(ks), 2 * d, d)}
    p['gate'] = _dense(next(ks), d + M, E, 0.5)
    p['experts'] = [{'l1': _dense(next(ks), d, 2 * d), 'l2': _dense(next(ks), 2 * d, d)} for _ in range(E)]
    T, V = cfg['n_types'], cfg['evdim']
    p['ev_W'] = jax.random.normal(next(ks), (T, V, d)) * np.sqrt(1.0 / V)
    p['ev_emb'] = jax.random.normal(next(ks), (T, d)) * 0.1
    p['ev_lam'] = jnp.full((T,), -1.0)
    p['lam_final'] = jnp.array(-1.0)
    p['gru'] = {n: _dense(next(ks), 2 * d, d) for n in ('z', 'r', 'h')}
    p['merge'] = _dense(next(ks), 2 * d, d)
    # heads
    def armhead(out):
        return {'W1': jax.random.normal(next(ks), (K, d, d)) * np.sqrt(1 / d), 'b1': jnp.zeros((K, d)),
                'W2': jax.random.normal(next(ks), (K, d, out)) * np.sqrt(1 / d) * 0.5, 'b2': jnp.full((K, out), -2.5 if out == J else 0.0)}
    p['h_os'], p['h_pfs'], p['h_resp'] = armhead(J), armhead(J), armhead(1)
    p['h_prop'] = {'l1': _dense(next(ks), d + M, d), 'l2': _dense(next(ks), d, K, 0.5)}
    p['h_cens'] = {'l1': _dense(next(ks), d, d), 'l2': _dense(next(ks), d, J, 0.5)}
    p['h_cens']['l2']['b'] = jnp.full(J, -3.0)
    p['h_pheno'] = _dense(next(ks), d, cfg['n_pheno'])
    # self-supervised decoders
    p['dec'] = [_dense(next(ks), d, dm) for dm in cfg['mod_dims']]
    p['nxt_type'] = _dense(next(ks), d, T)
    p['nxt_dt'] = _dense(next(ks), d, 1)
    return p


def init_dr(key, cfg):
    d, K = cfg['d'], cfg['K']
    k1, k2, k3 = jax.random.split(key, 3)
    return {'l1': _dense(k1, d + K * 2, d), 'l2': _dense(k2, d, d), 'o_r': _dense(k3, d, 2 * K, 0.3)}


# ---------------------------------------------------------------- blocks
def dense(p, x):
    return x @ p['W'] + p['b']


def ln(p, x):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + 1e-5) * p['g'] + p['b']


def mlp(p, x):
    return dense(p['l2'], jax.nn.gelu(dense(p['l1'], x)))


def attention(p, x, mask, nh=4):
    B, N, d = x.shape
    hd = d // nh
    q = dense(p['q'], x).reshape(B, N, nh, hd)
    k = dense(p['k'], x).reshape(B, N, nh, hd)
    v = dense(p['v'], x).reshape(B, N, nh, hd)
    lg = jnp.einsum('bqhd,bkhd->bhqk', q, k) / np.sqrt(hd)
    lg = jnp.where(mask[:, None, None, :] > 0, lg, -1e9)
    a = jax.nn.softmax(lg, -1)
    o = jnp.einsum('bhqk,bkhd->bqhd', a, v).reshape(B, N, d)
    return dense(p['o'], o), a


def gru(p, h, x):
    hx = jnp.concatenate([h, x], -1)
    z = jax.nn.sigmoid(dense(p['z'], hx))
    r = jax.nn.sigmoid(dense(p['r'], hx))
    hh = jnp.tanh(dense(p['h'], jnp.concatenate([r * h, x], -1)))
    return (1 - z) * h + z * hh


# ---------------------------------------------------------------- encoder
def encode(p, b, cfg, key=None, mod_drop=0.0, use_events=True, use_moe=True, use_gate_av=True,
           typed=True, use_att=True):
    """Return fused state z, decision state s, per-step hidden states, gate weights."""
    xs = b['static']
    B = xs.shape[0]
    toks = [dense(p['static'], xs) + p['mod_emb'][0]]
    masks = [jnp.ones(B)]
    av = b['av']
    if key is not None and mod_drop > 0:
        keep = jax.random.bernoulli(key, 1 - mod_drop, av.shape)
        av = av * keep
    for m, name in enumerate(cfg['mod_names']):
        toks.append(mlp(p['mod'][m], b['mods'][name]) + p['mod_emb'][m + 1])
        masks.append(av[:, m])
    X = jnp.stack(toks, 1)
    Mk = jnp.stack(masks, 1)
    if use_att:
        a, att = attention(p['att'], ln(p['ln1'], X), Mk)
        X = X + a
        X = X + mlp(p['ffn'], ln(p['ln2'], X))
        # attended static token plus an availability-masked pooling path so every
        # observed modality reaches the fused state even when attention is weak
        cls = X[:, 0] + (X * Mk[..., None]).sum(1) / Mk.sum(1, keepdims=True)
    else:  # masked mean pooling (late fusion)
        att = None
        cls = (X * Mk[..., None]).sum(1) / Mk.sum(1, keepdims=True)
    if use_moe:
        gin = jnp.concatenate([cls, av if use_gate_av else jnp.zeros_like(av)], -1)
        g = jax.nn.softmax(dense(p['gate'], gin), -1)
        ex = jnp.stack([mlp(e, cls) for e in p['experts']], 1)
        z = ln(p['ln3'], cls + (g[..., None] * ex).sum(1))
    else:
        g = None
        z = ln(p['ln3'], cls + mlp(p['experts'][0], cls))
    if not use_events:
        return z, z, None, g, att, av
    ty = b['ev_ty']
    oh = jax.nn.one_hot(ty, cfg['n_types'])
    if typed:
        xall = jnp.einsum('blv,tvd->bltd', b['ev_x'], p['ev_W'])
        xe = (xall * oh[..., None]).sum(2) + oh @ p['ev_emb']
        lam = jax.nn.softplus(oh @ p['ev_lam'])
    else:
        xe = b['ev_x'] @ p['ev_W'][0] + oh @ p['ev_emb']
        lam = jnp.full(ty.shape, jax.nn.softplus(p['ev_lam'][0]))
    tt, mm = b['ev_t'], b['ev_m']

    def step(carry, inp):
        h, tprev = carry
        x, t, m, l = inp
        dt = jnp.clip(t - tprev, 0, None)
        hd = z + (h - z) * jnp.exp(-l * dt)[:, None]
        hn = gru(p['gru'], hd, x)
        h2 = jnp.where(m[:, None] > 0, hn, h)
        tp = jnp.where(m > 0, t, tprev)
        return (h2, tp), h2

    t0 = tt[:, 0]
    (h, tl), hs = jax.lax.scan(step, (z, t0), (jnp.swapaxes(xe, 0, 1), tt.T, mm.T, lam.T))
    dt = jnp.clip(0.0 - tl, 0, None)
    h = z + (h - z) * jnp.exp(-jax.nn.softplus(p['lam_final']) * dt)[:, None]
    s = ln(p['ln4'], z + dense(p['merge'], jnp.concatenate([z, h], -1)))
    return z, s, jnp.swapaxes(hs, 0, 1), g, att, av


def armheads(p, s):
    h = jax.nn.gelu(jnp.einsum('bd,kde->bke', s, p['W1']) + p['b1'][None])
    return jnp.einsum('bke,keo->bko', h, p['W2']) + p['b2'][None]


def heads(p, s, av):
    return dict(os=armheads(p['h_os'], s), pfs=armheads(p['h_pfs'], s), resp=armheads(p['h_resp'], s)[..., 0],
                prop=mlp(p['h_prop'], jnp.concatenate([s, av], -1)), cens=mlp(p['h_cens'], s),
                pheno=dense(p['h_pheno'], s))


# ---------------------------------------------------------------- losses
def surv_nll(logits, b, ev):
    h = jax.nn.sigmoid(logits)
    k = jnp.arange(J)[None]
    lt = (k < b[:, None]).astype(jnp.float32)
    eq = (k == b[:, None]).astype(jnp.float32) * ev[:, None]
    ll = (lt * jnp.log(1 - h + 1e-7)).sum(1) + (eq * jnp.log(h + 1e-7)).sum(1)
    return -ll


def bce(logit, y):
    return optax.sigmoid_binary_cross_entropy(logit, y)


def sup_loss(p, b, cfg, key, opts):
    k1, k2 = jax.random.split(key)
    z, s, hs, g, _, av = encode(p, b, cfg, k1, opts['mod_drop'], **opts['enc'])
    if opts.get('dropout', 0) > 0 and opts.get('train', True):
        keep = jax.random.bernoulli(k2, 1 - opts['dropout'], s.shape)
        s = s * keep / (1 - opts['dropout'])
    H = heads(p, s, b['av'] if opts.get('prop_av', True) else jnp.zeros_like(b['av']))
    a = b['arm']
    sel = lambda x: jnp.take_along_axis(x, a[:, None, None], 1)[:, 0]
    l_os = surv_nll(sel(H['os']), b['os_b'], b['os_ev']).mean()
    l_pfs = surv_nll(sel(H['pfs']), b['pfs_b'], b['pfs_ev']).mean()
    l_resp = (bce(jnp.take_along_axis(H['resp'], a[:, None], 1)[:, 0], b['resp_f']) * b['resp_m']).sum() / (b['resp_m'].sum() + 1)
    l_prop = optax.softmax_cross_entropy_with_integer_labels(H['prop'], a).mean()
    l_cens = surv_nll(H['cens'], b['c_b'], b['c_ev']).mean()
    l_ph = (bce(H['pheno'], b['pheno']) * b['pheno_m']).sum() / (b['pheno_m'].sum() + 1)
    lb = 0.0
    if g is not None:
        lb = cfg['E'] * ((g.mean(0)) ** 2).sum()
    w = opts['w']
    tot = (l_os + w['pfs'] * l_pfs + w['resp'] * l_resp + w['prop'] * l_prop + w['cens'] * l_cens
           + w['pheno'] * l_ph + 0.01 * lb)
    return tot, dict(os=l_os, pfs=l_pfs, prop=l_prop, cens=l_cens, ph=l_ph, fact=l_os + l_pfs)


def ssl_loss(p, b, cfg, key, opts):
    k1, k2 = jax.random.split(key)
    drop = jax.random.bernoulli(k1, 0.4, b['av'].shape) * b['av']
    bb = dict(b)
    bb['av'] = b['av'] - drop
    z, s, hs, g, _, _ = encode(p, bb, cfg, None, 0.0, **opts['enc'])
    rec = 0.0
    for m, name in enumerate(cfg['mod_names']):
        e = ((dense(p['dec'][m], z) - b['mods'][name]) ** 2).mean(-1)
        rec = rec + (e * drop[:, m]).sum() / (drop[:, m].sum() + 1)
    nxt = 0.0
    if hs is not None:
        hprev = hs[:, :-1]
        tgt_ty = b['ev_ty'][:, 1:]
        m = b['ev_m'][:, 1:]
        ce = optax.softmax_cross_entropy_with_integer_labels(dense(p['nxt_type'], hprev), tgt_ty)
        dtt = jnp.log1p(jnp.clip(b['ev_t'][:, 1:] - b['ev_t'][:, :-1], 0, None))
        se = (dense(p['nxt_dt'], hprev)[..., 0] - dtt) ** 2
        nxt = ((ce + se) * m).sum() / (m.sum() + 1)
    return rec + nxt, dict(rec=rec, nxt=nxt)


# ---------------------------------------------------------------- training
def to_jax(d, idx=None):
    keys = ['static', 'av', 'ev_x', 'ev_t', 'ev_ty', 'ev_m', 'arm', 'os_b', 'os_ev', 'pfs_b', 'pfs_ev',
            'c_b', 'c_ev', 'pheno', 'pheno_m']
    sl = (lambda x: x) if idx is None else (lambda x: x[idx])
    b = {k: jnp.asarray(sl(d[k])) for k in keys}
    b['mods'] = {k: jnp.asarray(sl(v)) for k, v in d['mods'].items()}
    if 'resp' in d:
        r = sl(d['resp'])
        a = sl(d['arm'])
        b['resp_f'] = jnp.asarray(r[np.arange(len(a)), a].astype(np.float32))
        b['resp_m'] = jnp.asarray(np.ones(len(a), np.float32))
    else:
        b['resp_f'] = jnp.zeros(len(sl(d['arm'])))
        b['resp_m'] = jnp.zeros(len(sl(d['arm'])))
    return b


def batches(n, bs, rng):
    perm = rng.permutation(n)
    for i in range(0, n - bs + 1 if n >= bs else 1, bs):
        yield perm[i:i + bs]


def _take(b, idx):
    out = {k: v[idx] for k, v in b.items() if k != 'mods'}
    out['mods'] = {k: v[idx] for k, v in b['mods'].items()}
    return out


DEFAULT_OPTS = dict(mod_drop=0.25, dropout=0.2, enc=dict(use_events=True, use_moe=True, use_gate_av=True, typed=True, use_att=False),
                    w=dict(pfs=1.0, resp=0.3, prop=0.5, cens=0.3, pheno=0.3), prop_av=True)


def fit(p, dtr, dva, cfg, opts=None, loss='sup', epochs=60, bs=128, lr=7e-4, seed=0, patience=6, verbose=False, wd=1e-3):
    opts = opts or DEFAULT_OPTS
    lf = sup_loss if loss == 'sup' else ssl_loss
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr, weight_decay=wd))
    st = tx.init(p)

    @jax.jit
    def upd(p, st, b, key):
        (l, aux), gr = jax.value_and_grad(lambda pp: lf(pp, b, cfg, key, opts), has_aux=True)(p)
        u, st = tx.update(gr, st, p)
        return optax.apply_updates(p, u), st, l

    @jax.jit
    def evl(p, b):
        l, aux = lf(p, b, cfg, jax.random.PRNGKey(0), dict(opts, mod_drop=0.0, train=False))
        return aux['fact'] + 0.3 * aux['prop'] if loss == 'sup' else l

    Btr, Bva = to_jax(dtr), to_jax(dva)
    n = len(dtr['arm'])
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    best, bestp, bad = np.inf, p, 0
    for ep in range(epochs):
        for idx in batches(n, bs, rng):
            key, sk = jax.random.split(key)
            p, st, l = upd(p, st, _take(Btr, jnp.asarray(idx)), sk)
        v = float(evl(p, Bva))
        if verbose:
            print(ep, round(float(l), 4), round(v, 4))
        if v < best - 1e-4:
            best, bestp, bad = v, p, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return bestp, best


# ---------------------------------------------------------------- prediction
def surv_from_logits(lg):
    h = jax.nn.sigmoid(lg)
    return jnp.cumprod(1 - h, -1)


def rmst(S):
    Sp = jnp.concatenate([jnp.ones(S.shape[:-1] + (1,)), S[..., :-1]], -1)
    return (BW * (Sp + S) / 2).sum(-1)


@partial(jax.jit, static_argnums=(2, 3))
def _predict(p, b, cfg_t, enc_t):
    cfg = dict(cfg_t)
    enc = dict(enc_t)
    z, s, hs, g, att, _ = encode(p, b, cfg, None, 0.0, **enc)
    H = heads(p, s, b['av_prop'] if 'av_prop' in b else b['av'])
    So, Sp = surv_from_logits(H['os']), surv_from_logits(H['pfs'])
    Sc = surv_from_logits(H['cens'])
    return dict(S_os=So, S_pfs=Sp, rmst=rmst(So), p12=Sp[..., 5], prop=jax.nn.softmax(H['prop'], -1),
                Sc=Sc, pheno=jax.nn.sigmoid(H['pheno']), resp=jax.nn.sigmoid(H['resp']), s=s, z=z,
                gate=g if g is not None else jnp.zeros((s.shape[0], 1)))


def freeze(cfg):
    return tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in cfg.items())


def predict(p, d, cfg, opts=None, bs=1024):
    opts = opts or DEFAULT_OPTS
    n = len(d['arm'])
    outs = []
    B = to_jax(d)
    if not opts.get('prop_av', True):
        B['av_prop'] = jnp.zeros_like(B['av'])
    for i in range(0, n, bs):
        idx = jnp.arange(i, min(n, i + bs))
        o = _predict(p, _take(B, idx), freeze(cfg), freeze(opts['enc']))
        outs.append({k: np.asarray(v) for k, v in o.items()})
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


# ---------------------------------------------------------------- doubly robust stage
def g_at(Sc, b):
    """censoring survival just before bin b (product over bins < b)."""
    Sp = np.concatenate([np.ones((len(Sc), 1)), Sc], 1)
    return Sp[np.arange(len(Sc)), np.clip(b, 0, J)]


def pseudo_outcomes(d, nu, clip_p=0.05, clip_g=0.1):
    """AIPW (treatment) x IPCW (censoring) pseudo-outcomes for RMST_tau and Y = 1{PFS >= 12}."""
    n, K = len(d['arm']), nu['rmst'].shape[1]
    pi = np.clip(nu['prop'], clip_p, 1)
    A = np.eye(K)[d['arm']]
    G_r = np.clip(g_at(nu['Sc'], np.where(d['os_t'] >= TAU, J, d['os_b'])), clip_g, 1)
    b12 = np.minimum(d['pfs_b'], 6)
    G_y = np.clip(g_at(nu['Sc'], b12), clip_g, 1)
    ipcw_r = d['rmst_done'] * d['rmst_obs'] / G_r
    ipcw_y = d['y_obs'] * d['y'] / G_y
    phi_r = nu['rmst'] + A / pi * (ipcw_r[:, None] - nu['rmst'])
    phi_y = nu['p12'] + A / pi * (ipcw_y[:, None] - nu['p12'])
    return phi_r, phi_y


def fit_dr(pdr, s, base_r, base_y, phi_r, phi_y, cfg, epochs=200, lr=3e-3, seed=0):
    """Second-stage regression of pseudo-outcomes on the frozen twin state.

    The head outputs an additive correction to the plug-in estimates so the DR
    stage starts at the plug-in solution and moves only where the pseudo-outcomes
    disagree with it."""
    K = cfg['K']
    X = jnp.asarray(np.concatenate([s, base_r / TAU, base_y], 1))
    Yr, Yy = jnp.asarray(phi_r / TAU), jnp.asarray(phi_y)
    br, by = jnp.asarray(base_r / TAU), jnp.asarray(base_y)
    tx = optax.adamw(lr, weight_decay=1e-3)
    st = tx.init(pdr)

    def lf(q):
        o = dense(q['o_r'], jax.nn.gelu(dense(q['l2'], jax.nn.gelu(dense(q['l1'], X)))))
        r, y = br + o[:, :K], by + o[:, K:]
        return ((r - Yr) ** 2).mean() + ((y - Yy) ** 2).mean() + 1e-2 * (o ** 2).mean()

    @jax.jit
    def upd(q, st):
        l, g = jax.value_and_grad(lf)(q)
        u, st = tx.update(g, st, q)
        return optax.apply_updates(q, u), st, l
    for _ in range(epochs):
        pdr, st, l = upd(pdr, st)
    return pdr


def predict_dr(pdr, s, base_r, base_y, K):
    X = jnp.asarray(np.concatenate([s, base_r / TAU, base_y], 1))
    o = np.asarray(dense(pdr['o_r'], jax.nn.gelu(dense(pdr['l2'], jax.nn.gelu(dense(pdr['l1'], X))))))
    return np.clip(base_r + TAU * o[:, :K], 0, TAU), np.clip(base_y + o[:, K:], 0, 1)


def make_cfg(data, **kw):
    d = data
    cfg = dict(d=64, E=4, K=d["K"], ds=d["static"].shape[1], mod_dims=[d["mods"][k].shape[1] for k in d["mod_names"]],
               mod_names=list(d['mod_names']), n_types=d['n_types'], evdim=d['evdim'], n_pheno=d['pheno'].shape[1])
    cfg.update(kw)
    return cfg
