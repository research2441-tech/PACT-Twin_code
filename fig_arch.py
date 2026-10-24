"""Fig. 1: architecture of the positivity-aware conformal lung twin (grayscale, print size 7.2 in)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family': 'DejaVu Serif', 'mathtext.fontset': 'dejavuserif'})
TS, BS = 6.3, 5.6
fig = plt.figure(figsize=(7.2, 4.15), dpi=600)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 72)
ax.set_ylim(0, 41.5)
ax.axis('off')
BOX = {}


def box(key, x, y, w, h, title, lines=(), fill='#ffffff', ls='-', title_at=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.6', fc=fill, ec='black', lw=0.8, ls=ls))
    BOX[key] = (x, y, w, h)
    ty = y + h - 1.25 if title_at is None else title_at
    ax.text(x + w / 2, ty, title, ha='center', va='top', fontsize=TS, fontweight='bold')
    for k, ln in enumerate(lines):
        ax.text(x + w / 2, ty - 2.15 - 1.55 * k, ln, ha='center', va='top', fontsize=BS)


def side(key, s, f=0.5):
    x, y, w, h = BOX[key]
    return {'l': (x, y + h * f), 'r': (x + w, y + h * f), 't': (x + w * f, y + h), 'b': (x + w * f, y)}[s]


def arrow(p, q, ls='-'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>,head_length=0.36,head_width=0.2', lw=0.8, ls=ls, color='black',
                                 shrinkA=0, shrinkB=0.4, mutation_scale=10))


def poly(points, ls='-'):
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color='black', lw=0.8, ls=ls, solid_capstyle='butt', dash_capstyle='butt')
    arrow(points[-2], points[-1], ls=ls)


for x, t in [(6.75, '(a) Decision-time inputs'), (21.75, '(b) Multimodal state'), (37.25, '(c) Longitudinal twin'),
             (51.75, '(d) Causal heads'), (65.35, '(e) Conformal decision')]:
    ax.text(x, 40.7, t, ha='center', va='center', fontsize=6.4, fontstyle='italic')
ax.plot([16.0, 18.4], [38.6, 38.6], color='black', lw=0.8)
ax.text(18.7, 38.6, 'inference path', va='center', fontsize=5.6)
ax.plot([27.4, 29.8], [38.6, 38.6], color='black', lw=0.8, ls='--')
ax.text(30.1, 38.6, 'training / gradient path', va='center', fontsize=5.6)

# (a) inputs
box('ehr', 0.6, 31.9, 12.3, 6.7, 'Structured EHR  $\\mathbf{v}_i$', ['age, sex, smoking, ECOG,', 'metastases, LDH, albumin'], fill='#f2f2f2')
box('mod', 0.6, 14.8, 12.3, 16.4, 'Frozen encoder outputs', fill='#f2f2f2')
mods = ['CT', 'PET', 'WSI', 'NGS', 'RNA', 'ctDNA', 'PD-L1']
for k, m in enumerate(mods):
    yy = 27.3 - k * 1.75
    ax.add_patch(FancyBboxPatch((1.4, yy - 0.66), 7.6, 1.32, boxstyle='round,pad=0.01,rounding_size=0.3', fc='white', ec='black', lw=0.5))
    ax.text(5.2, yy, f'{m}  ' + '$\\mathbf{x}_i^{(%d)}$' % (k + 1), ha='center', va='center', fontsize=5.5)
    ax.add_patch(FancyBboxPatch((9.6, yy - 0.58), 2.5, 1.16, boxstyle='round,pad=0.01,rounding_size=0.25', fc='#d9d9d9', ec='black', lw=0.5))
    ax.text(10.85, yy, '$r_{i%d}$' % (k + 1), ha='center', va='center', fontsize=5.4)
ax.text(10.85, 15.65, 'availability', ha='center', va='center', fontsize=5.2, fontstyle='italic')
box('ev', 0.6, 3.0, 12.3, 10.4, 'Event history  $\\mathcal{H}_i$', fill='#f2f2f2')
ax.plot([1.6, 11.4], [7.4, 7.4], color='black', lw=0.7)
for xx, lab in [(2.2, 'dx'), (4.4, 'CT'), (6.6, 'bx'), (8.8, 'NGS'), (11.0, 'CT')]:
    ax.plot([xx, xx], [6.95, 7.85], color='black', lw=0.7)
    ax.text(xx, 8.35, lab, ha='center', va='bottom', fontsize=5.2)
ax.text(11.0, 6.5, '$t\\!<\\!0$', ha='center', va='top', fontsize=5.2)
ax.text(6.75, 4.6, '$(t_{ij},\\,\\kappa_{ij},\\,\\mathbf{e}_{ij})$', ha='center', va='center', fontsize=5.6)

# (b) fusion
box('adp', 15.6, 26.2, 12.3, 9.2, 'Modality adapters', ['token $\\mathbf{u}_{im}$ per modality', 'masked pooling by $r_{im}$', '(Eqs. 3-4)'])
box('moe', 15.6, 12.4, 12.3, 10.4, 'Availability-gated MoE', ['gate $\\mathbf{g}_i$ from $(\\bar{\\mathbf{u}}_i,\\mathbf{r}_i)$', '$E$ experts, fused $\\mathbf{z}_i$', '(Eq. 5)'])
box('ssl', 15.6, 3.0, 12.3, 7.7, 'SSL pretraining', ['masked-modality recon.', '+ next-event prediction', '(Eq. 15)'], fill='#f2f2f2', ls='--')

# (c) twin
box('gru', 31.1, 21.4, 12.3, 14.0, 'Event-typed relaxation', ['type embedding $\\mathbf{W}_{\\kappa}\\mathbf{e}_{ij}$', 'decay to $\\mathbf{z}_i$ at rate $\\lambda_{\\kappa}$',
                                                          'GRU update per event', '(Eqs. 6-8)'])
box('state', 31.1, 12.4, 12.3, 6.4, 'Decision state  $\\mathbf{s}_i$', ['$\\mathrm{LN}(\\mathbf{z}_i+\\mathbf{W}_s[\\mathbf{z}_i;\\mathbf{h}_i(0)])$', '(Eq. 9)'], fill='#e6e6e6')

# (d) heads and objective
box('haz', 46.1, 29.0, 11.3, 6.8, 'Arm-specific hazards', ['$\\eta_{a,k}(\\mathbf{s}_i)$: OS, PFS', '$\\hat{\\mu}_a$, $\\hat{p}_a$  (Eqs. 10-11)'])
box('cens', 46.1, 23.0, 11.3, 4.6, 'Censoring', ['$\\hat{G}(t\\mid\\mathbf{s}_i)$  (Eq. 14)'])
box('prop', 46.1, 16.9, 11.3, 4.7, 'Propensity', ['$\\hat{\\pi}_a(\\mathbf{s}_i,\\mathbf{r}_i)$  (Eq. 13)'])
box('aux', 46.1, 10.8, 11.3, 4.7, 'Auxiliary heads', ['phenotype, response'])
box('loss', 46.1, 3.0, 11.3, 6.2, 'Multi-task objective', ['$\\mathcal{L}$ over all heads', '(Eq. 16)'], fill='#f2f2f2')

# (e) conformal decision
box('calib', 60.1, 29.5, 10.5, 6.7, 'Calibration pool $\\mathcal{C}_a$', ['arm-$a$ patients with', 'durable control'], fill='#f2f2f2')
box('conf', 60.1, 17.2, 10.5, 10.6, 'Conformal sets', ['score $-\\hat{p}_a$', 'weight $1/(\\hat{\\pi}_a\\hat{G})$', '$q_a=\\infty$: not assessable', '(Eqs. 18-21)'])
box('dec', 60.1, 3.0, 10.5, 12.0, 'Decision rule', ['recommend $\\hat{a}\\in\\Gamma_{\\alpha}$', 'or defer: positivity', '(acquire modality)', 'or futility', '(Eq. 23)'], fill='#e6e6e6')

# forward (inference) arrows
arrow((12.9, 35.2), (15.6, 33.4))                          # EHR -> adapters
arrow((12.9, 25.6), (15.6, 28.4))                          # embeddings -> adapters
arrow((12.9, 17.6), (15.6, 16.2))                          # r_i -> MoE gate
ax.text(13.1, 18.3, '$\\mathbf{r}_i$', ha='left', va='bottom', fontsize=5.8)
arrow(side('adp', 'b'), side('moe', 't'))                  # adapters -> MoE
poly([(27.9, 17.6), (29.0, 17.6), (29.0, 26.5), (31.1, 26.5)])     # z_i -> twin
ax.text(28.1, 18.1, '$\\mathbf{z}_i$', ha='left', va='bottom', fontsize=5.8)
poly([(6.75, 3.0), (6.75, 1.4), (30.2, 1.4), (30.2, 23.0), (31.1, 23.0)])   # events -> twin
arrow(side('gru', 'b'), side('state', 't'))                # twin -> decision state
poly([(43.4, 15.6), (44.75, 15.6), (44.75, 32.4), (46.1, 32.4)])   # state -> hazards
arrow((44.75, 25.3), (46.1, 25.3))                          # -> censoring
arrow((44.75, 19.25), (46.1, 19.25))                        # -> propensity
poly([(44.75, 15.6), (44.75, 13.15), (46.1, 13.15)])       # -> auxiliary
arrow(side('aux', 'b'), side('loss', 't'))                 # heads -> objective
arrow((57.4, 31.0), (60.1, 26.6))                          # p_hat -> sets
arrow((57.4, 25.3), (60.1, 23.0))                          # G_hat -> sets
arrow((57.4, 19.25), (60.1, 19.25))                        # pi_hat -> sets
arrow(side('calib', 'b'), side('conf', 't'))               # calibration pool -> sets
arrow(side('conf', 'b'), side('dec', 't'))                 # sets -> decision
poly([(53.5, 35.8), (53.5, 37.3), (71.5, 37.3), (71.5, 9.0), (70.6, 9.0)])   # mu_hat -> decision
ax.text(62.5, 37.75, '$\\hat{\\mu}_a$', ha='center', va='bottom', fontsize=5.8)

# training paths (dashed): reverse-mode gradient from the objective back into the twin; SSL initialisation
poly([(46.1, 6.1), (38.6, 6.1), (38.6, 12.4)], ls='--')
ax.text(42.35, 6.6, '$\\nabla_{\\theta}\\mathcal{L}$', ha='center', va='bottom', fontsize=5.8)
ax.text(42.35, 5.6, 'reverse-mode', ha='center', va='top', fontsize=5.0, fontstyle='italic')
arrow(side('ssl', 't'), side('moe', 'b'), ls='--')
ax.text(22.3, 11.55, 'init. $\\theta$', ha='left', va='center', fontsize=5.4, fontstyle='italic')
fig.savefig('figs/fig1_architecture.png', dpi=600)
print('saved')
