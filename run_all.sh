#!/usr/bin/env bash
# Full PACT-Twin protocol. QUICK=1 runs one seed and one real split as a smoke test.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data runs figs
export PYTHONPATH=.
if [ ! -f data/chord_cohort.pkl ]; then
  bash download_msk_chord.sh
  python chord_cohort.py data
  mv chord_cohort.pkl data/
fi
SEEDS="0 1 2 3 4"; SPLITS="r0 r1 r2 r3 r4 t0"; SSEEDS="0 1 2"
if [ "${QUICK:-0}" = "1" ]; then SEEDS="0"; SPLITS="r0"; SSEEDS="0"; fi
python tune.py
python tune_real.py
for s in $SEEDS; do python run_sim_main.py "$s"; done
for t in $SPLITS; do python run_real.py "$t"; done
for s in $SSEEDS; do
  python run_landmark.py real "$s"; python run_landmark.py sim "$s"
  python run_sens.py gamma "$s"; python run_sens.py label "$s"
done
python run_attr.py
python describe_sim.py; python real_descriptive.py
python analysis_sim.py; python analysis_extra.py sens landmark; python evaluate_real.py
python fig_arch.py
python figs_results.py fig_conformal_mechanism fig_main fig_alpha fig_stress fig_sensitivity fig_landmark fig_real fig_attr
echo "done: summaries in runs/, figures in figs/"
