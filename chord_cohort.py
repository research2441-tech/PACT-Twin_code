"""Build the MSK-CHORD de novo stage IV NSCLC first-line cohort.

Every feature is restricted to information time-stamped on or before the
first-line start (plus a 7-day reporting grace) so that the decision-time
twin state never sees the future. Post-decision events are kept separately
for landmark (dynamic) prediction.

Output: chord_cohort.pkl with keys
  static   : DataFrame of decision-time features (NaN = missing)
  avail    : DataFrame of modality availability indicators
  events   : dict patient -> list of (day_rel_L1, type, vector)
  arm      : first-line arm index (0 Chemo, 1 Chemo-IO, 2 IO-mono, 3 TKI)
  os_t, os_e, pfs_t, pfs_e (days from L1)
  order    : patient sequence number (temporal proxy)
"""
import sys, pickle, re
import numpy as np, pandas as pd

D = sys.argv[1] if len(sys.argv) > 1 else 'data'
def rd(f):
    # clinical files carry '#'-prefixed header lines; timeline files carry
    # '#RRGGBB' colour codes inside rows, so comment parsing must be off there
    if f.startswith('data_clinical'):
        return pd.read_csv(f"{D}/{f}", sep='\t', comment='#', low_memory=False)
    return pd.read_csv(f"{D}/{f}", sep='\t', low_memory=False)

IO = {'PEMBROLIZUMAB', 'NIVOLUMAB', 'ATEZOLIZUMAB', 'DURVALUMAB', 'IPILIMUMAB',
      'CEMIPLIMAB', 'TREMELIMUMAB'}
TKI = {'OSIMERTINIB', 'ERLOTINIB', 'GEFITINIB', 'AFATINIB', 'DACOMITINIB',
       'ALECTINIB', 'CRIZOTINIB', 'BRIGATINIB', 'LORLATINIB', 'CERITINIB',
       'CAPMATINIB', 'TEPOTINIB', 'SELPERCATINIB', 'PRALSETINIB', 'DABRAFENIB',
       'TRAMETINIB', 'SOTORASIB', 'ADAGRASIB', 'ENTRECTINIB', 'LAROTRECTINIB',
       'MOBOCERTINIB', 'AMIVANTAMAB', 'CABOZANTINIB'}
CHEMO_CLASS = {'Chemo'}
ARMS = ['Chemo', 'Chemo-IO', 'IO-mono', 'TKI']
CNA_GENES = ['EGFR', 'MET', 'ERBB2', 'MDM2', 'CDK4', 'MYC', 'NKX2-1', 'KRAS',
             'CDKN2A', 'CDKN2B', 'PTEN', 'RB1', 'SMARCA4', 'STK11']
SITES = ['Adrenal Glands', 'Bone', 'CNS/Brain', 'Liver', 'Lymph Node', 'Pleura',
         'Lung', 'Intra Abdominal', 'Other']
EV_TYPES = ['dx', 'seq', 'pdl1', 'ecog', 'cea', 'site', 'presence', 'prog',
            'rad', 'surg', 'treat']


def main():
    pt = rd('data_clinical_patient.txt').set_index('PATIENT_ID')
    sm = rd('data_clinical_sample.txt')
    ns = sm[sm.CANCER_TYPE == 'Non-Small Cell Lung Cancer'].copy()
    P = set(ns.PATIENT_ID)
    sp = rd('data_timeline_specimen.txt')
    sp = sp[sp.SAMPLE_ID.isin(ns.SAMPLE_ID)]
    seqday = sp.groupby('SAMPLE_ID').START_DATE.min()
    ns['seqday'] = ns.SAMPLE_ID.map(seqday)
    ns = ns.sort_values(['PATIENT_ID', 'seqday']).groupby('PATIENT_ID').head(1).set_index('PATIENT_ID')

    dx = rd('data_timeline_diagnosis.txt')
    dx = dx[dx.PATIENT_ID.isin(P) & (dx.SUBTYPE == 'Primary') &
            dx.DX_DESCRIPTION.str.contains('LUNG', na=False)]
    dx = dx.sort_values('START_DATE').groupby('PATIENT_ID').head(1).set_index('PATIENT_ID')
    dx = dx[dx.STAGE_CDM_DERIVED == 'Stage 4']

    tx = rd('data_timeline_treatment.txt')
    tx = tx[tx.PATIENT_ID.isin(dx.index)]
    systemic = tx[tx.SUBTYPE.isin(['Chemo', 'Immuno', 'Targeted', 'Biologic', 'Investigational'])]
    rows = []
    for pid, g in systemic.groupby('PATIENT_ID'):
        d0 = dx.at[pid, 'START_DATE']
        g = g[g.START_DATE >= d0 - 14]
        if len(g) == 0:
            continue
        s1 = g.START_DATE.min()
        if s1 > d0 + 365:
            continue
        w = g[g.START_DATE <= s1 + 28]
        ag = set(w.AGENT.str.upper())
        if 'INVESTIGATIONAL' in ag or (w.SUBTYPE == 'Investigational').any():
            continue
        has_io = len(ag & IO) > 0
        has_tki = len(ag & TKI) > 0
        has_chemo = (w.SUBTYPE == 'Chemo').any()
        if has_tki:
            a = 3
        elif has_io and has_chemo:
            a = 1
        elif has_io:
            a = 2
        elif has_chemo:
            a = 0
        else:
            continue
        # time of second-line start: first systemic agent not in L1 set after s1+28
        later = g[(g.START_DATE > s1 + 28) & (~g.AGENT.str.upper().isin(ag))]
        s2 = later.START_DATE.min() if len(later) else np.nan
        rows.append((pid, d0, s1, a, s2))
    co = pd.DataFrame(rows, columns=['pid', 'd0', 's1', 'arm', 's2']).set_index('pid')
    co = co[co.index.isin(ns.index) & co.index.isin(pt.index)]
    osd = pt.loc[co.index, 'OS_MONTHS'] * 30.4375
    co['os_t'] = osd - co.s1
    co['os_e'] = pt.loc[co.index, 'OS_STATUS'].str.startswith('1').astype(int)
    co = co[co.os_t > 7]

    # ---------- static decision-time features ----------
    S = pd.DataFrame(index=co.index)
    p = pt.loc[co.index]
    S['male'] = (p.GENDER == 'Male').astype(float)
    S['race_white'] = (p.RACE == 'White').astype(float)
    S['race_black'] = (p.RACE.str.contains('Black', na=False)).astype(float)
    S['race_asian'] = (p.RACE.str.contains('Asian', na=False)).astype(float)
    S['hispanic'] = (p.ETHNICITY.str.contains('Hispanic', na=False) & ~p.ETHNICITY.str.contains('Non', na=False)).astype(float)
    age = pd.to_numeric(p.CURRENT_AGE_DEID, errors='coerce')
    S['age'] = (age - (co.os_t / 365.25)).clip(20, 95)
    smk = p.SMOKING_PREDICTIONS_3_CLASSES.fillna('Unknown')
    S['smoke_former'] = (smk == 'Former').astype(float)
    S['smoke_current'] = (smk == 'Current').astype(float)
    S['smoke_never'] = (smk == 'Never').astype(float)
    S['prior_med'] = (p.PRIOR_MED_TO_MSK.str.contains('Prior medications', na=False)).astype(float)
    S['dx_to_l1'] = (co.s1 - co.d0) / 30.4
    n = ns.loc[co.index]
    det = n.CANCER_TYPE_DETAILED.fillna('')
    S['luad'] = det.str.contains('Adenocarcinoma').astype(float)
    S['lusc'] = det.str.contains('Squamous').astype(float)
    S['primary_sample'] = (n.SAMPLE_TYPE == 'Primary').astype(float)

    A = pd.DataFrame(index=co.index)
    # genomic modality available only if sequenced >=14 days before L1 start
    A['genomic'] = ((co.s1 - n.seqday) >= 14).astype(float)
    S['tmb'] = np.log1p(pd.to_numeric(n.TMB_NONSYNONYMOUS, errors='coerce'))
    S['msi'] = pd.to_numeric(n.MSI_SCORE, errors='coerce').clip(0, 40)
    S['purity'] = pd.to_numeric(n.TUMOR_PURITY, errors='coerce') / 100
    cna = pd.read_csv(f"{D}/data_cna.txt", sep='\t', comment='#', low_memory=False)
    cna = cna[cna.Hugo_Symbol.isin(CNA_GENES)].set_index('Hugo_Symbol')
    cols = [s for s in n.SAMPLE_ID if s in cna.columns]
    cm = cna[cols].T
    for gname in CNA_GENES:
        v = cm[gname] if gname in cm.columns else pd.Series(dtype=float)
        sid = n.SAMPLE_ID
        val = sid.map(v)
        if gname in ('CDKN2A', 'CDKN2B', 'PTEN', 'RB1', 'SMARCA4', 'STK11'):
            S['del_' + gname] = (val <= -2).astype(float).where(val.notna())
        else:
            S['amp_' + gname] = (val >= 2).astype(float).where(val.notna())
    gcols = ['tmb', 'msi', 'purity'] + [c for c in S.columns if c.startswith(('amp_', 'del_'))]
    S.loc[A.genomic == 0, gcols] = np.nan

    # ---------- longitudinal sources ----------
    def tl(f):
        t = rd(f)
        return t[t.PATIENT_ID.isin(co.index)].copy()
    pdl = tl('data_timeline_pdl1.txt')
    ecog = tl('data_timeline_performance_status.txt')
    cea = tl('data_timeline_cea_labs.txt')
    site = tl('data_timeline_tumor_sites.txt')
    pres = tl('data_timeline_cancer_presence.txt')
    prog = tl('data_timeline_progression.txt')
    rad = tl('data_timeline_radiation.txt')
    surg = tl('data_timeline_surgery.txt')
    for t in (pdl, ecog, cea, site, pres, prog, rad, surg):
        t['rel'] = t.START_DATE - t.PATIENT_ID.map(co.s1)
    cea['val'] = np.log1p(pd.to_numeric(cea.RESULT, errors='coerce').clip(0, 5000))
    ecog['val'] = pd.to_numeric(ecog.ECOG, errors='coerce')

    # PD-L1
    pre = pdl[pdl.rel <= 7].sort_values('rel').groupby('PATIENT_ID').tail(1)
    A['pdl1'] = co.index.isin(pre.PATIENT_ID).astype(float)
    S['pdl1_pos'] = pre.set_index('PATIENT_ID').PDL1_POSITIVE.reindex(co.index).map({'Yes': 1.0, 'No': 0.0})
    # ECOG
    pre = ecog[(ecog.rel <= 7) & (ecog.rel >= -90)].dropna(subset=['val']).sort_values('rel').groupby('PATIENT_ID').tail(1)
    A['ecog'] = co.index.isin(pre.PATIENT_ID).astype(float)
    S['ecog'] = pre.set_index('PATIENT_ID').val.reindex(co.index)
    # CEA
    c = cea[(cea.rel <= 7) & (cea.rel >= -180)].dropna(subset=['val'])
    A['cea'] = co.index.isin(c.PATIENT_ID).astype(float)
    last = c.sort_values('rel').groupby('PATIENT_ID').tail(1).set_index('PATIENT_ID').val
    S['cea_last'] = last.reindex(co.index)

    def slope(g):
        if len(g) < 2 or g.rel.max() - g.rel.min() < 14:
            return np.nan
        return np.polyfit(g.rel / 30.4, g.val, 1)[0]
    S['cea_slope'] = c.groupby('PATIENT_ID').apply(slope, include_groups=False).reindex(co.index)
    # radiology (NLP) tumour sites
    s_ = site[(site.rel <= 7) & (site.rel >= -120)]
    A['radiology'] = co.index.isin(s_.PATIENT_ID).astype(float)
    for nm in SITES:
        has = set(s_[s_.TUMOR_SITE == nm].PATIENT_ID)
        S['site_' + re.sub('[^a-z]', '', nm.lower())] = co.index.isin(has).astype(float)
    sc = [c for c in S.columns if c.startswith('site_')]
    S.loc[A.radiology == 0, sc] = np.nan
    S['n_sites'] = S[sc].sum(axis=1).where(A.radiology == 1)
    S['prior_rad'] = co.index.isin(rad[rad.rel < 0].PATIENT_ID).astype(float)
    S['prior_surg'] = co.index.isin(surg[surg.rel < 0].PATIENT_ID).astype(float)

    # ---------- PFS ----------
    pg = prog[(prog.PROGRESSION == 'Y') & (prog.rel > 28)]
    first = pg.groupby('PATIENT_ID').rel.min().reindex(co.index)
    pfs_t = np.minimum(first.fillna(np.inf), co.os_t)
    pfs_e = ((first <= co.os_t) | (co.os_e == 1)).astype(int)
    co['pfs_t'] = pfs_t
    co['pfs_e'] = pfs_e

    # ---------- event sequences (relative to L1, days) ----------
    ev = {pid: [] for pid in co.index}
    def add(df, typ, fn, lo=-365):
        for r in df[(df.rel >= lo)].itertuples():
            ev[r.PATIENT_ID].append((float(r.rel), typ, fn(r)))
    add(pdl, 'pdl1', lambda r: [1.0 if r.PDL1_POSITIVE == 'Yes' else -1.0])
    add(ecog.dropna(subset=['val']), 'ecog', lambda r: [r.val / 4])
    add(cea.dropna(subset=['val']), 'cea', lambda r: [r.val / 5])
    site['k'] = site.TUMOR_SITE.map({s: i for i, s in enumerate(SITES)}).fillna(8)
    add(site, 'site', lambda r: list(np.eye(len(SITES))[int(r.k)]))
    pres['p'] = pd.to_numeric(pres.NLP_HAS_CANCER_PROBABILITY, errors='coerce').fillna(0.5)
    add(pres, 'presence', lambda r: [r.p])
    prog['p'] = pd.to_numeric(prog.NLP_PROGRESSION_PROBABILITY, errors='coerce').fillna(0.5)
    add(prog, 'prog', lambda r: [r.p])
    add(rad, 'rad', lambda r: [1.0])
    add(surg, 'surg', lambda r: [1.0])
    for pid in co.index:
        ev[pid].append((float(co.at[pid, 'd0'] - co.at[pid, 's1']), 'dx', [1.0]))
        ev[pid].append((float(ns.at[pid, 'seqday'] - co.at[pid, 's1']), 'seq', [1.0]))
        ev[pid].sort(key=lambda x: x[0])

    order = pd.Series([int(re.sub(r'\D', '', i)) for i in co.index], index=co.index)
    out = dict(static=S, avail=A, events=ev, arm=co.arm.values.astype(int),
               os_t=co.os_t.values.astype(float), os_e=co.os_e.values.astype(int),
               pfs_t=co.pfs_t.values.astype(float), pfs_e=co.pfs_e.values.astype(int),
               s2=(co.s2 - co.s1).values, order=order.values, pid=list(co.index),
               arms=ARMS, ev_types=EV_TYPES)
    pickle.dump(out, open('chord_cohort.pkl', 'wb'))
    print('patients', len(co))
    print(pd.Series(co.arm).map(dict(enumerate(ARMS))).value_counts().to_string())
    print('availability\n', A.mean().round(3).to_string())
    print('OS events', co.os_e.mean().round(3), 'median OS d', np.median(co.os_t))
    print('PFS events', co.pfs_e.mean().round(3))


if __name__ == '__main__':
    main()
