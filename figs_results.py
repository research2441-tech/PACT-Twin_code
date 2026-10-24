"""Result figures (grayscale, single-column width unless noted). Reads runs/*.json|pkl."""
import json, pickle, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'DejaVu Serif', 'mathtext.fontset': 'dejavuserif', 'font.size': 7, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'legend.frameon': False, 'axes.spines.top': False,
                     'axes.spines.right': False, 'hatch.linewidth': 0.5})
GRAYS = ['#000000', '#4d4d4d', '#808080', '#b3b3b3', '#d9d9d9', '#ffffff']
HATCH = ['', '////', '....', 'xxxx', '\\\\\\\\', '----', 'oo', '++']
MARK = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
LS = ['-', '--', ':', '-.', (0, (5, 1, 1, 1)), (0, (1, 1))]
W1 = 3.45
NAMES = {'Twin:full': 'PACT-Twin', 'CoxT': 'Cox-T', 'RSFT': 'RSF-T', 'DRGB': 'DR-GB', 'SMLP': 'S-MLP', 'TARNet': 'TARNet',
         'CFR': 'CFR', 'CRN': 'CRN'}
SETN = {'test': 'Internal', 'ext_E': 'Site E', 'ext_F': 'Site F'}
OUT = 'figs/'


def save(fig, name):
    fig.savefig(OUT + name, dpi=600)
    plt.close(fig)
    print('saved', name)


def fig_conformal_mechanism():
    """Fig. 2: (a) normalised test weight vs propensity for TKI; (b) outcome of the set rule by arm and true label."""
    from conformal import treatment_sets
    E = pickle.load(open('runs/conf_example.pkl', 'rb'))
    S = json.load(open('runs/sim_summary.json'))
    cal = dict(p=E['cal_p'], pi=E['cal_pi'], g12=E['cal_g'], arm=E['cal_arm'], y=E['cal_y'], y_obs=E['cal_yobs'])
    te = dict(p=E['te_p'], pi=E['te_pi'], g12=E['te_g'])
    excl, assess, thr = treatment_sets(cal, te, alpha=0.1)
    a = 3
    m = (cal['arm'] == a) & (cal['y_obs'] == 1) & (cal['y'] == 1)
    wc = 1 / (np.clip(cal['pi'][m, a], 1e-3, 1) * np.clip(cal['g12'][m], .05, 1))
    wt = 1 / (np.clip(te['pi'][:, a], 1e-3, 1) * np.clip(te['g12'], .05, 1))
    po = wt / (wc.sum() + wt)
    fig, ax = plt.subplots(1, 2, figsize=(W1 * 2.05, 2.35), gridspec_kw=dict(wspace=0.38))
    gen = E['te_gen'] > 0
    for msk, mk, fc, lab in [(gen, 'o', 'white', 'NGS before $t=0$'), (~gen, 's', '#808080', 'no NGS before $t=0$')]:
        ax[0].scatter(np.clip(te['pi'][msk, a], 1e-3, 1), po[msk], s=6, marker=mk, facecolors=fc, edgecolors='black', linewidths=0.35, label=lab)
    ax[0].axhspan(0.1, 1.5, color='#e6e6e6', zorder=0, lw=0)
    ax[0].text(0.3, 0.45, 'not assessable', fontsize=6.3, ha='center', fontstyle='italic')
    ax[0].axhline(0.1, color='black', lw=0.8, ls='--')
    ax[0].text(1.2e-3, 0.115, r'$\alpha=0.1$', fontsize=6.5, va='bottom')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_ylim(3e-3, 1.5)
    ax[0].set_xlabel(r'estimated propensity $\hat{\pi}_{\mathrm{TKI}}(o)$')
    ax[0].set_ylabel(r'normalised test weight $p_{o}(o)$')
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=2, fontsize=6.3, handletextpad=0.3, columnspacing=1.0, markerscale=1.5)
    ax[0].text(-0.2, 1.17, '(a)', transform=ax[0].transAxes, fontsize=8, fontweight='bold')
    # (b) composition by arm and true label (seed 0, internal test)
    import sys
    sys.path.insert(0, '.')
    from sim_exp import make_cohort, subset
    d, du, sp, D = make_cohort(0)
    Y = subset(d, sp['test'])['truth_y']
    arms = ['C', 'C-IO', 'IO', 'TKI']
    xs, lab = [], []
    comp = []
    for k in range(4):
        for yv in (1, 0):
            mm = Y[:, k] == yv
            ex = (excl[mm, k] & assess[mm, k]).mean()
            na = (~assess[mm, k]).mean()
            kept = 1 - ex - na
            comp.append((kept, ex, na))
            lab.append(f'{arms[k]}\n$Y{{=}}{yv}$')
    comp = np.array(comp)
    x = np.arange(len(comp))
    bottom = np.zeros(len(comp))
    for j, (nm, fc, h) in enumerate([('retained, assessable', 'white', ''), ('excluded', '#4d4d4d', ''), ('not assessable', '#bfbfbf', '////')]):
        ax[1].bar(x, comp[:, j], bottom=bottom, color=fc, edgecolor='black', lw=0.4, hatch=h, label=nm, width=0.72)
        bottom += comp[:, j]
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(lab, fontsize=5.8)
    ax[1].set_ylabel('share of patients')
    ax[1].set_ylim(0, 1)
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=3, fontsize=6.2, handlelength=1.2, columnspacing=0.8)
    ax[1].text(-0.2, 1.17, '(b)', transform=ax[1].transAxes, fontsize=8, fontweight='bold')
    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.2)
    save(fig, 'fig2_conformal_mechanism.png')


def grouped(ax, methods, vals, errs, ylabel, sets=('test', 'ext_E', 'ext_F'), ref=None):
    n = len(methods)
    width = 0.8 / len(sets)
    x = np.arange(n)
    for j, st in enumerate(sets):
        ax.bar(x + (j - (len(sets) - 1) / 2) * width, [vals[m][st] for m in methods], width, yerr=[errs[m][st] for m in methods],
               color=['white', '#9a9a9a', '#3d3d3d'][j], edgecolor='black', lw=0.4, hatch=['', '', ''][j],
               error_kw=dict(lw=0.5, capsize=1.2, capthick=0.5), label=SETN[st])
    ax.set_xticks(x)
    ax.set_xticklabels([NAMES[m] for m in methods], rotation=35, ha='right', fontsize=6.3)
    ax.set_ylabel(ylabel)
    if ref is not None:
        ax.axhline(ref, color='black', lw=0.6)


def fig_main():
    S = json.load(open('runs/sim_summary.json'))
    M = ['Twin:full', 'CoxT', 'RSFT', 'DRGB', 'SMLP', 'TARNet', 'CFR', 'CRN']
    fig, ax = plt.subplots(1, 2, figsize=(W1 * 2.05, 2.45), gridspec_kw=dict(wspace=0.3))
    sm = S['summary']
    v = {m: {st: sm[m][st]['pehe_ov'][0] for st in SETN} for m in M}
    e = {m: {st: sm[m][st]['pehe_ov'][1] for st in SETN} for m in M}
    grouped(ax[0], M, v, e, 'overlap PEHE (months)')
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=6.3)
    ax[0].text(-0.16, 1.14, '(a)', transform=ax[0].transAxes, fontsize=8, fontweight='bold')
    pol = S['policies']
    M2 = M + ['guideline', 'oracle']
    v = {m: {st: sm[m][st]['v_conf_fallback'][0] - pol[st]['obs'][0] for st in SETN} for m in M}
    e = {m: {st: sm[m][st]['v_conf_fallback'][1] for st in SETN} for m in M}
    for r in ('guideline', 'oracle'):
        v[r] = {st: pol[st][r][0] - pol[st]['obs'][0] for st in SETN}
        e[r] = {st: pol[st][r][1] for st in SETN}
    NAMES['guideline'], NAMES['oracle'] = 'Guideline', 'Oracle'
    grouped(ax[1], M2, v, e, 'value gain over practice (months)', ref=0)
    ax[1].axvline(7.5, color='#808080', lw=0.5, ls=':')
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=6.3)
    ax[1].text(-0.16, 1.14, '(b)', transform=ax[1].transAxes, fontsize=8, fontweight='bold')
    fig.subplots_adjust(left=0.07, right=0.99, top=0.83, bottom=0.25)
    save(fig, 'fig3_main.png')


def fig_alpha():
    S = json.load(open('runs/sim_summary.json'))
    A = S['alpha_sweep']
    fig, ax = plt.subplots(1, 2, figsize=(W1 * 2.05, 2.35), gridspec_kw=dict(wspace=0.32))
    labs = {'ours': 'weighted, label-conditional', 'unweighted': 'unweighted', 'marginal': 'weighted, marginal'}
    for k, vn in enumerate(['ours', 'unweighted', 'marginal']):
        for j, st in enumerate(['test', 'ext_F']):
            al = sorted(A[vn][st], key=float)
            xs = [float(a) for a in al]
            fer = [max(A[vn][st][a]['fer']) for a in al]
            ax[0].plot(xs, fer, color=['black', '#555555', '#999999'][k], ls=['-', '--'][j], marker=MARK[k], ms=3, lw=0.8,
                       mfc='white' if j else None, label=f'{labs[vn]}, {SETN[st]}')
            if j == 0:
                ax[1].plot(xs, [A[vn][st][a]['power'] for a in al], color=['black', '#555555', '#999999'][k], marker=MARK[k], ms=3, lw=0.8,
                           label=labs[vn])
    ax[0].plot([0.05, 0.3], [0.05, 0.3], color='black', lw=0.5, ls=':')
    ax[0].text(0.3, 0.255, 'nominal', fontsize=6, ha='right', va='top')
    ax[0].set_xlabel(r'level $\alpha$')
    ax[0].set_ylabel('maximum FER over arms')
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.32), ncol=2, fontsize=5.6, handlelength=2.2, columnspacing=0.8)
    ax[0].text(-0.17, 1.25, '(a)', transform=ax[0].transAxes, fontsize=8, fontweight='bold')
    ax[1].set_xlabel(r'level $\alpha$')
    ax[1].set_ylabel('power (futile arms excluded)')
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=2, fontsize=5.8)
    ax[1].text(-0.17, 1.25, '(b)', transform=ax[1].transAxes, fontsize=8, fontweight='bold')
    fig.subplots_adjust(left=0.08, right=0.99, top=0.76, bottom=0.17)
    save(fig, 'fig4_alpha.png')


def fig_stress():
    S = json.load(open('runs/sim_summary.json'))
    st = S['stress']
    keys = ['none', 'drop_pdl1', 'drop_wsi', 'drop_ctdna', 'drop_pet_rna', 'drop_gen', 'ct_ehr_only']
    lab = ['none', 'PD-L1', 'WSI', 'ctDNA', 'PET+RNA', 'NGS', 'all but\nCT+EHR']
    M = ['Twin:full', 'CoxT', 'TARNet', 'CRN']
    fig, ax = plt.subplots(1, 2, figsize=(W1 * 2.05, 2.4), gridspec_kw=dict(wspace=0.32, width_ratios=[1.35, 1]))
    x = np.arange(len(keys))
    w = 0.2
    for j, m in enumerate(M):
        vals = [S['summary'][m]['test']['pehe_ov'][0] if k == 'none' else st[m][k]['pehe_ov'][0] for k in keys]
        ax[0].bar(x + (j - 1.5) * w, vals, w, color=['white', '#bdbdbd', '#7a7a7a', '#2e2e2e'][j], edgecolor='black', lw=0.4,
                  hatch=['', '', '', ''][j], label=NAMES[m])
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(lab, fontsize=6.0)
    ax[0].set_ylabel('overlap PEHE (months)')
    ax[0].set_xlabel('modality withheld at prediction time')
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4, fontsize=6.2)
    ax[0].text(-0.13, 1.14, '(a)', transform=ax[0].transAxes, fontsize=8, fontweight='bold')
    V = S['voi']
    for k, stn in enumerate(['test', 'ext_F']):
        L = [v for v in V if v['set'] == stn]
        comp_b = np.array([[np.mean([v['rec_before'] for v in L]), np.mean([v['pos_before'] for v in L]), np.mean([v['fut_before'] for v in L])],
                           [np.mean([v['rec_after'] for v in L]), np.mean([v['pos_after'] for v in L]), np.mean([v['fut_after'] for v in L])]])
        for r in range(2):
            bottom = 0
            for j, (nm, fc, h) in enumerate([('recommended', 'white', ''), ('positivity deferral', '#9a9a9a', '////'),
                                             ('futility deferral', '#3d3d3d', '')]):
                ax[1].bar(k * 2.6 + r, comp_b[r, j], bottom=bottom, color=fc, edgecolor='black', lw=0.4, hatch=h, width=0.8,
                          label=nm if (k == 0 and r == 0) else None)
                bottom += comp_b[r, j]
    ax[1].set_xticks([0, 1, 2.6, 3.6])
    ax[1].set_xticklabels(['before', 'after', 'before', 'after'], fontsize=6.2)
    ax[1].text(0.5, -0.27, 'Internal', ha='center', transform=ax[1].get_xaxis_transform(), fontsize=6.6)
    ax[1].text(3.1, -0.27, 'Site F', ha='center', transform=ax[1].get_xaxis_transform(), fontsize=6.6)
    ax[1].set_ylabel('share of unsequenced patients')
    ax[1].set_ylim(0, 1)
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.27), ncol=2, fontsize=6.0, columnspacing=0.8)
    ax[1].text(-0.2, 1.14, '(b)', transform=ax[1].transAxes, fontsize=8, fontweight='bold')
    fig.subplots_adjust(left=0.07, right=0.99, top=0.8, bottom=0.27)
    save(fig, 'fig5_stress_voi.png')


def fig_sensitivity():
    G = {}
    for f in glob.glob('runs/sens_gamma_*.json'):
        G.update(json.load(open(f)))
    S = json.load(open('runs/sens_summary.json'))
    fig, ax = plt.subplots(1, 2, figsize=(W1 * 2.05, 2.3), gridspec_kw=dict(wspace=0.32))
    for j, m in enumerate(['Twin', 'CoxT', 'TARNet']):
        gs = sorted(S['gamma'][m], key=float)
        ax[0].errorbar([float(g) for g in gs], [S['gamma'][m][g]['pehe_ov'][0] for g in gs], yerr=[S['gamma'][m][g]['pehe_ov'][1] for g in gs],
                       color=['black', '#555555', '#999999'][j], marker=MARK[j], ms=3.5, lw=0.8, capsize=1.5, elinewidth=0.5,
                       label={'Twin': 'PACT-Twin', 'CoxT': 'Cox-T', 'TARNet': 'TARNet'}[m])
    ax[0].set_xlabel(r'unmeasured confounding $\gamma$')
    ax[0].set_ylabel('overlap PEHE (months)')
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=6.3)
    ax[0].text(-0.17, 1.14, '(a)', transform=ax[0].transAxes, fontsize=8, fontweight='bold')
    for j, m in enumerate(['Twin:full', 'Twin:no_ssl', 'CoxT']):
        fs = sorted(S['label'][m], key=float)
        ax[1].errorbar([100 * float(f) for f in fs], [S['label'][m][f]['pehe_ov'][0] for f in fs], yerr=[S['label'][m][f]['pehe_ov'][1] for f in fs],
                       color=['black', '#555555', '#999999'][j], marker=MARK[j], ms=3.5, lw=0.8, capsize=1.5, elinewidth=0.5, ls=LS[j],
                       label={'Twin:full': 'PACT-Twin', 'Twin:no_ssl': 'without SSL', 'CoxT': 'Cox-T'}[m])
    ax[1].set_xscale('log')
    ax[1].set_xticks([10, 25, 50, 100])
    ax[1].set_xticklabels(['10', '25', '50', '100'])
    ax[1].set_xlabel('labelled training patients (%)')
    ax[1].set_ylabel('overlap PEHE (months)')
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=6.3)
    ax[1].text(-0.17, 1.14, '(b)', transform=ax[1].transAxes, fontsize=8, fontweight='bold')
    fig.subplots_adjust(left=0.08, right=0.99, top=0.83, bottom=0.18)
    save(fig, 'fig6_sensitivity.png')


def fig_landmark():
    L = json.load(open('runs/landmark_summary.json'))
    fig, ax = plt.subplots(1, 2, figsize=(W1 * 2.05, 2.55), gridspec_kw=dict(wspace=0.3))
    for k, kind in enumerate(['sim', 'real']):
        D = L[kind]
        lm = sorted(D, key=float)
        for j, m in enumerate(['Twin-updated', 'Twin-baseline', 'CoxT-LOCF', 'CoxT-baseline']):
            ax[k].errorbar([float(x) for x in lm], [D[x][m][0] for x in lm], yerr=[D[x][m][1] for x in lm], color=['black', 'black', '#808080', '#808080'][j],
                           ls=['-', '--', '-', '--'][j], marker=MARK[j], ms=3.5, lw=0.8, capsize=1.5, elinewidth=0.5,
                           mfc=['black', 'white', '#808080', 'white'][j], label={'Twin-updated': 'twin, updated', 'Twin-baseline': 'twin, decision time',
                                                                              'CoxT-LOCF': 'Cox, last value', 'CoxT-baseline': 'Cox, decision time'}[m])
        ax[k].set_xlabel('landmark after first-line start (months)')
        ax[k].set_ylabel('residual OS concordance')
        ax[k].set_xticks([float(x) for x in lm])
        ax[k].set_title(('(a) LungTwinSim' if k == 0 else '(b) MSK-CHORD'), fontsize=7.5, pad=4, loc='left', fontweight='bold')
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=4, fontsize=6.3)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.8, bottom=0.17)
    save(fig, 'fig7_landmark.png')


def fig_real():
    R = json.load(open('runs/real_eval.json'))
    RD = json.load(open('runs/real_descriptive.json'))
    fig, ax = plt.subplots(1, 2, figsize=(W1 * 2.05, 2.45), gridspec_kw=dict(wspace=0.38, width_ratios=[1, 1.3]))
    arms = ['Chemo', 'Chemo-IO', 'IO-mono']
    feats = [('ngs_before', 'NGS'), ('pdl1_tested', 'PD-L1'), ('ecog_av', 'ECOG'), ('radiology_av', 'radiology'), ('cea_av', 'CEA')]
    x = np.arange(len(feats))
    for j, a in enumerate(arms):
        ax[0].bar(x + (j - 1) * 0.26, [RD['table'][a][f] for f, _ in feats], 0.26, color=['white', '#9a9a9a', '#3d3d3d'][j],
                  edgecolor='black', lw=0.4, label=a)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([n for _, n in feats], fontsize=6.3)
    ax[0].set_ylabel('available before first line')
    ax[0].set_ylim(0, 1.0)
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=6.3)
    ax[0].text(-0.2, 1.14, '(a)', transform=ax[0].transAxes, fontsize=8, fontweight='bold')
    M = ['Twin:full', 'CoxT', 'RSFT', 'DRGB', 'SMLP', 'TARNet', 'CFR', 'CRN']
    tags = sorted(t for t in R if t.startswith('r'))
    for i, m in enumerate(M):
        ds = [R[t][m]['dV_conf_fb'] for t in tags]
        mu = [d[0] for d in ds]
        y = len(M) - 1 - i
        for k, (t, d) in enumerate(zip(tags, ds)):
            ax[1].plot([d[1], d[2]], [y + (k - 2) * 0.13] * 2, color='#9a9a9a', lw=0.5)
            ax[1].plot(d[0], y + (k - 2) * 0.13, marker='o', ms=2.2, color='#555555', mew=0.4)
        ax[1].plot(np.mean(mu), y, marker='D', ms=4, color='black')
    ax[1].axvline(0, color='black', lw=0.6, ls='--')
    ax[1].set_yticks(range(len(M)))
    ax[1].set_yticklabels([NAMES[m] for m in M[::-1]], fontsize=6.3)
    ax[1].set_xlabel('off-policy RMST gain over practice (months)')
    ax[1].text(-0.28, 1.14, '(b)', transform=ax[1].transAxes, fontsize=8, fontweight='bold')
    from matplotlib.lines import Line2D
    hl = [Line2D([], [], marker='o', ls='-', color='#555555', ms=2.5, lw=0.5, label='random split, 95% CI'),
          Line2D([], [], marker='D', ls='', color='black', ms=3.5, label='mean of five splits')]
    ax[1].legend(handles=hl, loc='upper center', bbox_to_anchor=(0.45, 1.2), ncol=2, fontsize=5.9, columnspacing=0.7, handlelength=1.6)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.83, bottom=0.18)
    save(fig, 'fig8_real.png')


def fig_attr():
    A = pickle.load(open('runs/attr_seed0.pkl', 'rb'))
    fig, ax = plt.subplots(1, 2, figsize=(W1 * 2.05, 2.35), gridspec_kw=dict(wspace=0.85, width_ratios=[1, 1]))
    g = list(A['A'])
    M = np.array([A['A'][k] for k in g])
    cols = ['C vs C-IO', 'IO vs C-IO', 'TKI vs C-IO']
    im = ax[0].imshow(M, cmap='Greys', aspect='auto', vmin=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax[0].text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=5.8, color='white' if M[i, j] > 0.6 * M.max() else 'black')
    ax[0].set_yticks(range(len(g)))
    ax[0].set_yticklabels(g, fontsize=6.2)
    ax[0].set_xticks(range(3))
    ax[0].set_xticklabels(cols, rotation=25, ha='right', fontsize=6.2)
    ax[0].set_title('mean |change in contrast| (months)', fontsize=6.6, pad=3)
    ax[0].text(-0.62, 1.08, '(a)', transform=ax[0].transAxes, fontsize=8, fontweight='bold')
    gg = list(A['G'])
    G = np.array([A['G'][k] for k in gg])
    ax[1].imshow(G, cmap='Greys', aspect='auto', vmin=0, vmax=1)
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            ax[1].text(j, i, f'{G[i, j]:.2f}', ha='center', va='center', fontsize=5.8, color='white' if G[i, j] > 0.6 else 'black')
    ax[1].set_yticks(range(len(gg)))
    ax[1].set_yticklabels(gg, fontsize=6.2)
    ax[1].set_xticks(range(3))
    ax[1].set_xticklabels(cols, rotation=25, ha='right', fontsize=6.2)
    ax[1].set_title('|Spearman| with true contrast', fontsize=6.6, pad=3)
    ax[1].text(-0.55, 1.08, '(b)', transform=ax[1].transAxes, fontsize=8, fontweight='bold')
    fig.subplots_adjust(left=0.2, right=0.99, top=0.86, bottom=0.2)
    save(fig, 'fig9_attribution.png')


if __name__ == '__main__':
    import sys
    for f in sys.argv[1:]:
        globals()[f]()
