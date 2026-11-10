#!/usr/bin/env bash
# Copy this file to data_paths.local.sh and set the paths available locally.
# The local file is intentionally ignored because these paths are specific to
# one workstation. Exporters should consume these variables, not hardcode paths.

export MIMIC_III_1_4_ROOT=/path/to/mimic-iii/1.4
export MIMIC_IV_0_4_ROOT=/path/to/mimic-iv/0.4
export MIMIC_IV_1_0_ROOT=/path/to/mimic-iv/1.0
export MIMIC_IV_2_0_ROOT=/path/to/mimic-iv/2.0
export MIMIC_IV_2_2_ROOT=/path/to/mimic-iv/2.2
export MIMIC_IV_3_0_ROOT=/path/to/mimic-iv/3.0
export MIMIC_IV_3_1_ROOT=/path/to/mimic-iv/3.1

# Optional modality-specific releases used by some papers.
export MIMIC_IV_NOTE_2_2_ROOT=/path/to/mimic-iv-note/2.2
export MIMIC_IV_EXT_INSTR_1_0_ROOT=/path/to/mimic-iv-ext-instr/1.0.0
export MIMIC_IV_ED_2_0_ROOT=/path/to/mimic-iv-ed/2.0
export MIMIC_IV_ED_2_2_ROOT=/path/to/mimic-iv-ed/2.2
export MIMIC_IV_FHIR_1_0_ROOT=/path/to/mimic-iv-on-fhir/1.0/fhir
export MIMIC_IV_ECG_1_0_ROOT=/path/to/mimic-iv-ecg/1.0
export MIMIC_CXR_2_1_0_ROOT=/path/to/mimic-cxr/2.1.0
export MIMIC_CXR_JPG_2_0_0_ROOT=/path/to/exact-mimic-cxr-jpg/2.0.0-payload
export MIMIC_CXR_JPG_2_1_0_ROOT=/path/to/mimic-cxr-jpg/2.1.0
