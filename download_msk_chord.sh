#!/usr/bin/env bash
# Fetch the MSK-CHORD (msk_chord_2024) clinical and timeline tables from the cBioPortal datahub (Git LFS media URLs).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
B="https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/msk_chord_2024"
for f in data_clinical_patient.txt data_clinical_sample.txt data_cna.txt data_timeline_treatment.txt \
         data_timeline_progression.txt data_timeline_cea_labs.txt data_timeline_pdl1.txt data_timeline_tumor_sites.txt \
         data_timeline_diagnosis.txt data_timeline_specimen.txt data_timeline_radiation.txt data_timeline_surgery.txt \
         data_timeline_cancer_presence.txt data_timeline_performance_status.txt; do
  curl -fsSL -o "data/$f" "$B/$f"
  echo "fetched $f"
done
