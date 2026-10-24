"""Aggregate the LungTwinSim runs into the numbers, tables and figure data used in the manuscript."""
import glob, pickle, json, itertools
import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare
from sim_exp import make_cohort, subset, eval_method, policy_value, cindex
from conformal import treatment_sets, decide, evaluate_sets
from baselines import guideline_policy
from run_sim_main import drop_mods, reveal_gen, STRESS

SETS = ['test', 'ext_E', 'ext_F']
METHODS = ['Twin:full', 'CoxT', 'RSFT', 'DRGB', 'SMLP', 'TARNet', 'CFR', 'CRN']
ABL = ['Twin:full', 'Twin:no_ssl', 'Twin:no_events', 'Twin:untyped', 'Twin:no_moe', 'Twin:no_avail', 'Twin:attmoe',
       'Twin:no_moddrop', 'Twin:dr_mlp', 'Twin:dr_shift']
KEYS = ['pehe_ov', 'pehe_all', 'v_naive', 'v_overlap', 'v_conf_fallback', 'v_conf_hybrid', 'regret_conf_kept', 'regret_obs_kept',
        'defer', 'defer_pos', 'defer_fut', 'C_os', 'acc_best', 'brier_cf', 'ece_cf', 'fer_max', 'fwer', 'power', 'set_size',
        'v_obs', 'v_oracle', 'fer_0', 'fer_1', 'fer_2', 'fer_3']


def g_true(dd):
    from lungsim import SITES
    rate = np.array([SITES[x]['drop'] for x in dd['site']])
    return (48 / 52) * np.exp(-rate * np.exp(0.15 * dd['g_ps']) * 12)


def oracle_w(P, dd):
    Q = dict(P)
    Q['prop'] = dd['prop_true']
    Q['g12'] = g_true(dd)
    return Q


def holm(ps):
    o = np.argsort(ps)
    m = len(ps)
    adj = np.zeros(m)
    run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i])
        adj[i] = min(1, run)
    return adj


def main():
    runs = {}
    for f in sorted(glob.glob('runs/sim_seed[0-9].pkl')):
        R = pickle.load(open(f, 'rb'))
        runs[R['seed']] = R
    seeds = sorted(runs)
    res = {}            # res[method][set] -> list over seeds of metric dicts
    extra = dict(conf_variants={}, alpha_sweep={}, local_recal={}, stress={}, voi=[], subgroup={}, timing={}, conf_example=None,
                 policies={})
    for s in seeds:
        R = runs[s]
        d, du, sp, D = make_cohort(s)
        sets = {k: subset(d, sp[k]) for k in ['cal'] + SETS}
        for m in dict.fromkeys(METHODS + ABL):
            if m not in R['preds']:
                continue
            P = R['preds'][m]
            kk = dict(rmst_key='rmst_dr', p_key='p12_dr') if m.startswith('Twin:dr') else {}
            for st in SETS:
                r, aux = eval_method(P['cal'], P[st], sets['cal'], sets[st], **kk)
                rs, _ = eval_method(P['cal'], P[st], sets['cal'], sets[st], simultaneous=True, **kk)
                r['fwer_simul'] = rs['fwer']; r['fer_max_simul'] = rs['fer_max']; r['set_size_simul'] = rs['set_size']
                res.setdefault(m, {}).setdefault(st, []).append(r)
        for st in SETS:
            T = sets[st]['mu_true']
            gp = guideline_policy(sets[st])
            extra['policies'].setdefault(st, []).append(dict(obs=T[np.arange(len(T)), sets[st]['arm']].mean(), oracle=T.max(1).mean(),
                                                             guideline=T[np.arange(len(T)), gp].mean(),
                                                             best_single=T.mean(0).max()))
        # conformal variants and alpha sweep (twin, internal test + external)
        P = R['preds']['Twin:full']
        for st in SETS:
            for vn, kw in [('ours', {}), ('unweighted', dict(weighted=False)), ('marginal', dict(label_cond=False)),
                           ('no_censor_w', dict(use_g=False)), ('simultaneous', dict(simultaneous=True))]:
                r, _ = eval_method(P['cal'], P[st], sets['cal'], sets[st], **kw)
                extra['conf_variants'].setdefault(vn, {}).setdefault(st, []).append(r)
            pc_o, pt_o = oracle_w(P['cal'], sets['cal']), oracle_w(P[st], sets[st])
            r, _ = eval_method(pc_o, pt_o, sets['cal'], sets[st])
            extra['conf_variants'].setdefault('oracle_weights', {}).setdefault(st, []).append(r)
            for a in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
                for vn, kw in [('ours', {}), ('unweighted', dict(weighted=False)), ('marginal', dict(label_cond=False))]:
                    r, _ = eval_method(P['cal'], P[st], sets['cal'], sets[st], alpha=a, **kw)
                    extra['alpha_sweep'].setdefault(vn, {}).setdefault(st, {}).setdefault(a, []).append(
                        dict(fer=[r['fer_%d' % k] for k in range(4)], power=r['power'], size=r['set_size'], defer=r['defer']))
        # site-local recalibration on external sites (30 % of the site as calibration)
        for st in ['ext_E', 'ext_F']:
            rng = np.random.default_rng(s + 500)
            n = len(sets[st]['arm'])
            perm = rng.permutation(n)
            lc, ev = perm[: int(.3 * n)], perm[int(.3 * n):]
            de, dl = subset(sets[st], ev), subset(sets[st], lc)
            Pe = {k: v[ev] for k, v in P[st].items()}
            Pl = {k: v[lc] for k, v in P[st].items()}
            r_int, _ = eval_method(P['cal'], Pe, sets['cal'], de)
            r_loc, _ = eval_method(Pl, Pe, dl, de)
            extra['local_recal'].setdefault(st, []).append(dict(internal=r_int, local=r_loc))
        # stress tests
        for m in ['Twin:full', 'CoxT', 'TARNet', 'CRN']:
            P = R['preds'][m]
            for k in list(STRESS) + ['reveal']:
                key = 'test__' + k
                if key not in P:
                    continue
                dd = drop_mods(sets['test'], STRESS[k]) if k != 'reveal' else reveal_gen(sets['test'], D, sp['test'])
                r, aux = eval_method(P['cal'], P[key], sets['cal'], dd)
                extra['stress'].setdefault(m, {}).setdefault(k, []).append(r)
        # value of information: positivity deferrals before / after revealing NGS
        P = R['preds']['Twin:full']
        for st, key in [('test', 'test__reveal'), ('ext_F', 'ext_F__reveal')]:
            dd = reveal_gen(sets[st], D, sp[st])
            mask = dd['revealed']
            r0, a0 = eval_method(P['cal'], P[st], sets['cal'], sets[st])
            r1, a1 = eval_method(P['cal'], P[key], sets['cal'], dd)
            T = sets[st]['mu_true']
            drv = (sets[st]['g_egfr'] | sets[st]['g_alk'])
            obs = sets[st]['arm']
            v = lambda rec: T[np.arange(len(T)), np.where(rec >= 0, rec, obs)]
            extra['voi'].append(dict(seed=s, set=st, n_missing=int(mask.sum()),
                                     pos_before=float((a0['reason'][mask] == 1).mean()), pos_after=float((a1['reason'][mask] == 1).mean()),
                                     fut_before=float((a0['reason'][mask] == 2).mean()), fut_after=float((a1['reason'][mask] == 2).mean()),
                                     rec_before=float((a0['rec'][mask] >= 0).mean()), rec_after=float((a1['rec'][mask] >= 0).mean()),
                                     v_before=float(v(a0['rec'])[mask].mean()), v_after=float(v(a1['rec'])[mask].mean()),
                                     tki_drv_before=float((a0['rec'][mask & drv] == 3).mean()) if (mask & drv).any() else np.nan,
                                     tki_drv_after=float((a1['rec'][mask & drv] == 3).mean()) if (mask & drv).any() else np.nan,
                                     tki_nodrv_after=float((a1['rec'][mask & ~drv] == 3).mean()),
                                     oracle=float(T.max(1)[mask].mean()), obs=float(T[np.arange(len(T)), obs][mask].mean())))
        # subgroup analysis (internal test + site E pooled)
        for m in ['Twin:full', 'CoxT', 'TARNet']:
            P = R['preds'][m]
            for st in ['test', 'ext_E']:
                dte = sets[st]
                _, aux = eval_method(P['cal'], P[st], sets['cal'], dte)
                T = dte['mu_true']
                rec = aux['rec']
                excl = aux['excl']
                Y = dte['truth_y']
                groups = {'Female': ~dte['g_male'], 'Male': dte['g_male'], 'Asian': dte['g_asian'], 'Non-Asian': ~dte['g_asian'],
                          'Age >= 70': dte['g_age'] >= 70, 'Age < 70': dte['g_age'] < 70, 'ECOG >= 2': dte['g_ps'] >= 2}
                pol = np.where(rec >= 0, rec, dte['arm'])
                for g, msk in groups.items():
                    pos = (Y == 1) & msk[:, None]
                    extra['subgroup'].setdefault(m, {}).setdefault(g, []).append(dict(
                        regret=float((T.max(1) - T[np.arange(len(T)), pol])[msk].mean()),
                        regret_obs=float((T.max(1) - T[np.arange(len(T)), dte['arm']])[msk].mean()),
                        fer=float((excl & pos).sum() / max(pos.sum(), 1)), defer=float((rec[msk] < 0).mean()), n=int(msk.sum())))
        extra['timing'][s] = R['timing']
        if s == seeds[0]:
            P = R['preds']['Twin:full']
            extra['conf_example'] = dict(cal_p=P['cal']['p12'], cal_pi=P['cal']['prop'], cal_g=P['cal']['g12'], cal_arm=sets['cal']['arm'],
                                         cal_y=sets['cal']['y'], cal_yobs=sets['cal']['y_obs'], te_p=P['test']['p12'], te_pi=P['test']['prop'],
                                         te_g=P['test']['g12'], te_drv=(sets['test']['g_egfr'] | sets['test']['g_alk']),
                                         te_gen=sets['test']['av'][:, sets['test']['mod_names'].index('gen')])
    # ---- summaries
    def agg(lst, k):
        v = np.array([x[k] for x in lst], float)
        return float(np.nanmean(v)), float(np.nanstd(v))
    summary = {m: {st: {k: agg(res[m][st], k) for k in KEYS + ['fwer_simul', 'fer_max_simul', 'set_size_simul']} for st in SETS} for m in res}
    # paired statistics: twin vs each comparator over (seed, set) blocks
    stats = {}
    for metric, better in [('pehe_ov', 'low'), ('v_overlap', 'high'), ('v_conf_fallback', 'high'), ('C_os', 'high')]:
        ps, comp = [], []
        for m in METHODS[1:]:
            a = np.array([x[metric] for st in SETS for x in res['Twin:full'][st]])
            b = np.array([x[metric] for st in SETS for x in res[m][st]])
            try:
                p = wilcoxon(a, b).pvalue
            except ValueError:
                p = 1.0
            ps.append(p)
            comp.append(dict(method=m, diff=float((a - b).mean()), p=float(p), wins=int(((a < b) if better == 'low' else (a > b)).sum()), n=len(a)))
        adj = holm(np.array(ps))
        for c, q in zip(comp, adj):
            c['p_holm'] = float(q)
        blocks = np.array([[x[metric] for st in SETS for x in res[m][st]] for m in METHODS])
        fr = friedmanchisquare(*blocks)
        ranks = np.argsort(np.argsort(blocks if better == 'low' else -blocks, axis=0), axis=0) + 1
        stats[metric] = dict(pairs=comp, friedman_chi2=float(fr.statistic), friedman_p=float(fr.pvalue),
                             mean_rank={m: float(r) for m, r in zip(METHODS, ranks.mean(1))})
    # ablation paired tests vs full (pehe_ov, v_conf_fallback, fer_max)
    abl_stats = {}
    for m in ABL[1:]:
        if m not in res:
            continue
        for metric in ['pehe_ov', 'v_conf_fallback', 'C_os']:
            a = np.array([x[metric] for st in SETS for x in res['Twin:full'][st]])
            b = np.array([x[metric] for st in SETS for x in res[m][st]])
            try:
                p = wilcoxon(a, b).pvalue
            except ValueError:
                p = 1.0
            abl_stats.setdefault(m, {})[metric] = dict(diff=float((b - a).mean()), p=float(p))
    out = dict(seeds=seeds, summary=summary, stats=stats, abl_stats=abl_stats,
               policies={st: {k: agg(extra['policies'][st], k) for k in extra['policies'][st][0]} for st in SETS},
               conf_variants={vn: {st: {k: agg(v[st], k) for k in ['fer_max', 'fwer', 'power', 'set_size', 'defer', 'v_conf_fallback',
                                                                  'fer_0', 'fer_1', 'fer_2', 'fer_3']} for st in SETS}
                              for vn, v in extra['conf_variants'].items()},
               alpha_sweep={vn: {st: {str(a): dict(fer=np.mean([x['fer'] for x in L], 0).tolist(), power=float(np.mean([x['power'] for x in L])),
                                                    size=float(np.mean([x['size'] for x in L])), defer=float(np.mean([x['defer'] for x in L])))
                                      for a, L in v[st].items()} for st in SETS} for vn, v in extra['alpha_sweep'].items()},
               local_recal={st: {w: {k: agg([x[w] for x in L], k) for k in ['fer_max', 'fer_0', 'fer_1', 'fer_2', 'fer_3', 'power', 'set_size', 'defer', 'v_conf_fallback']}
                                 for w in ('internal', 'local')} for st, L in extra['local_recal'].items()},
               stress={m: {k: {kk: agg(L, kk) for kk in ['pehe_ov', 'v_overlap', 'v_conf_fallback', 'C_os', 'defer', 'fer_max']} for k, L in v.items()}
                       for m, v in extra['stress'].items()},
               voi=extra['voi'],
               subgroup={m: {g: {k: agg(L, k) for k in ['regret', 'regret_obs', 'fer', 'defer', 'n']} for g, L in v.items()} for m, v in extra['subgroup'].items()},
               timing={m: float(np.mean([extra['timing'][s][m] for s in seeds if m in extra['timing'][s]])) for m in extra['timing'][seeds[0]]})
    json.dump(out, open('runs/sim_summary.json', 'w'), indent=1, default=float)
    pickle.dump(extra['conf_example'], open('runs/conf_example.pkl', 'wb'))
    return out


if __name__ == '__main__':
    o = main()
    for m in o['summary']:
        print(m, {k: round(o['summary'][m]['test'][k][0], 3) for k in ['pehe_ov', 'v_overlap', 'v_conf_fallback', 'C_os', 'defer', 'fer_max', 'fwer_simul', 'power']})
    print(json.dumps(o['stats'], indent=0)[:3000])
    print(o['policies'])
