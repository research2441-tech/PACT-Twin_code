"""Descriptive real-world analysis: cohort table and how strongly modality availability predicts first-line arm."""
import pickle, json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from data_prep import from_chord
R = pickle.load(open('data/chord_cohort.pkl', 'rb'))
d = from_chord(R)
S, A = d['raw_static'], d['raw_avail']
arms = ['Chemo', 'Chemo-IO', 'IO-mono']
out = {'n': int(len(d['arm'])), 'arm_n': np.bincount(d['arm']).tolist()}
tab = {}
for j, a in enumerate(arms):
    m = d['arm'] == j
    s = S[m]
    tab[a] = dict(n=int(m.sum()), age=float(np.nanmedian(s['age'])), male=float(s['male'].mean()), never=float(s['smoke_never'].mean()),
                  luad=float(s['luad'].mean()), lusc=float(s['lusc'].mean()), pdl1_tested=float(A['pdl1'][m].mean()),
                  pdl1_pos=float(np.nanmean(s['pdl1_pos'])), ngs_before=float(A['genomic'][m].mean()), ecog_av=float(A['ecog'][m].mean()),
                  radiology_av=float(A['radiology'][m].mean()), cea_av=float(A['cea'][m].mean()),
                  os_events=float(d['os_e'][m].mean()), med_fu=float(np.median(d['os_t'][m])))
out['table'] = tab
cv = StratifiedKFold(5, shuffle=True, random_state=0)
era = d['static'][:, d['static_cols'].index('era')][:, None]
for nm, X in [('availability', A.values), ('availability+era', np.c_[A.values, era]), ('era', era),
              ('all features', np.c_[d['static'], np.concatenate([d['mods'][k] for k in d['mod_names']], 1), d['av']])]:
    pr = cross_val_predict(LogisticRegression(max_iter=3000, C=1.0), X, d['arm'], cv=cv, method='predict_proba')
    out['auc_' + nm] = float(roc_auc_score(d['arm'], pr, multi_class='ovr', average='macro'))
# events per patient and modality availability overall
out['avail_overall'] = A.mean().to_dict()
out['events_pre'] = float(d['ev_m'].sum(1).mean())
out['y12'] = float(d['y'][d['y_obs'] > 0].mean())
json.dump(out, open('runs/real_descriptive.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
