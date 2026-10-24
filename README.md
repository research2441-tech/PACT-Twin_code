# PACT-Twin

Positivity-aware conformal treatment twin for first-line therapy selection in advanced non-small-cell lung cancer.

This repository holds all code behind the manuscript: the semi-synthetic multi-site benchmark (LungTwinSim), the MSK-CHORD
cohort builder, the twin model, seven comparison methods, the weighted label-conditional conformal treatment sets, every
experiment, and the scripts that regenerate the figures and summary tables. All files sit in one folder. Scripts create
`data/`, `runs/` and `figs/` next to themselves when they run.

## Requirements

Python 3.11 or newer on a CPU is enough; no GPU is used.

```
pip install -r requirements.txt
```

## Data

* **LungTwinSim** is generated on the fly by `lungsim.py` (seeds 0-4 for evaluation, 99 for development).
* **MSK-CHORD** (Jee et al., Nature 2024) is public through the cBioPortal datahub under CC BY-NC-ND 4.0.
  `download_msk_chord.sh` fetches the clinical and timeline tables into `data/`, and `chord_cohort.py` builds the
  de novo stage IV first-line cohort (`data/chord_cohort.pkl`). Mutation-level files are not needed.

## Reproducing the results

```
bash run_all.sh            # full protocol (about 6 CPU hours on two cores)
QUICK=1 bash run_all.sh    # one seed and one real split, for a smoke test
```

| Step | Script | Output |
|---|---|---|
| Real cohort | `download_msk_chord.sh`, `chord_cohort.py` | `data/chord_cohort.pkl` |
| Hyperparameter selection | `tune.py` (sim, seed 99), `tune_real.py` (real dev splits) | `runs/tune_*.json` |
| Main simulation | `run_sim_main.py SEED` | `runs/sim_seedS.pkl` |
| Real-world splits | `run_real.py r0..r4, t0` | `runs/real_*.pkl` |
| Sensitivity | `run_sens.py gamma|label SEED` | `runs/sens_*.pkl` |
| Dynamic updating | `run_landmark.py sim|real SEED` | `runs/landmark_*.pkl` |
| Attribution, efficiency | `run_attr.py` | `runs/attr_seed0.pkl` |
| Summaries | `analysis_sim.py`, `analysis_extra.py sens landmark`, `evaluate_real.py`, `real_descriptive.py`, `describe_sim.py` | `runs/*.json` |
| Figures | `fig_arch.py`, `figs_results.py ...` | `figs/*.png` |

## Main modules

* `lungsim.py`: six-site simulator with closed-form tumour-burden trajectories, coupled potential outcomes,
  modality availability, sequencing turnaround and a clinician policy with unmeasured confounding.
* `data_prep.py`: decision-time packing of modalities, availability and event histories; survival discretisation.
* `twin.py`: availability-gated mixture-of-experts fusion, event-typed relaxation recurrence, arm-specific hazard,
  propensity, censoring and auxiliary heads, self-supervised pretraining (JAX).
* `conformal.py`: weighted label-conditional treatment sets, non-assessability flags and the decision rule.
* `baselines.py`: Cox-T, RSF-T, DR-GB, S-MLP, TARNet, CFR, CRN and the guideline policy.

## Licence

MIT for the code. MSK-CHORD data remain under their own licence and are not redistributed here.
