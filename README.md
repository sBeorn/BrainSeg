# BrainSeg — Thalamic Nuclei Segmentation Pipeline

Thalamic nuclei segmentation and morphometric analysis pipeline using atlas-based label propagation with ANTsPy nonlinear registration.

---

## Overview

This pipeline performs atlas-to-subject registration, label propagation, and comprehensive morphometric analysis of thalamic substructures from T1/T2 MRI data. It is designed for research use and produces structured, reproducible outputs for expert review.

The pipeline currently covers 38 thalamic labels across 5 IXI subjects using a single MNI152 atlas, with FreeSurfer pseudo-ground-truth integration underway.

---

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| İP-1 | `ip1_preproc.py` | N4 bias correction, brain masking (ANTsPyNet), resampling to 1mm isotropic |
| İP-2 | `ip2_warp.py` | Atlas→Subject nonlinear registration (3 candidates: W1/W2/W3), Jacobian QC |
| İP-3 | `ip3_propagate.py` | Label propagation (NN interpolation) + full morphometrics |
| İP-4 | `ip4_refine.py` | Boundary refinement (3 candidates: R1/R2/R3) |
| İP-5 | `ip5_freesurfer_gt.py` | FreeSurfer pseudo-GT generation + BrainSeg comparison |
| — | `leave_one_out_val.py` | LOO cross-validation (proxy DSC) |
| — | `compare_candidates.py` | Warp/refine candidate ranking |
| — | `generate_project_report.py` | Full HTML report generation |

**Best configuration:** W2 (SyN + Mattes MI) + R1 (ham propagation)

---

## Morphometrics

For each label and subject, the pipeline computes:

**Geometric** — volume (mm³), surface area, sphericity/compactness, elongation, flatness (PCA eigenvalues), skeleton length, bounding box fill ratio, connected components

**Positional** — centroid (mm), L/R volume ratio, midline distance, centroid mirror error

**Intensity/Texture** — T1 mean/std/IQR/MAD, gradient energy, LoG energy, GLCM (contrast, homogeneity, entropy, correlation), T1/T2 ratio

**Quality tiers** — TIER-1 (reliable), TIER-2 (use with caution), TIER-3 (unreliable)

---

## Results (W2, 5 subjects, 38 labels)

| Metric | Value |
|--------|-------|
| Mean LOO DSC | 0.514 (proxy) |
| DSC ≥ 0.70 labels | 4/38 (11%) — PuM, RN, PuL, MDpc |
| DSC 0.50–0.70 | 16/38 (42%) |
| DSC < 0.50 | 18/38 (47%) |
| Large structures (≥200 mm³) mean DSC | 0.639 |
| Small structures (<100 mm³) mean DSC | 0.392 |
| Cross-subject consistency | 0.786 |
| TIER-1 / TIER-2 / TIER-3 | 169 / 13 / 23 |

> **Note:** All DSC values are LOO proxy metrics — independent FreeSurfer validation is in progress.

---

## FreeSurfer Integration

FreeSurfer 7.4.1 is installed (WSL/Ubuntu 22.04). The integration code is complete:

- `_wsl_path()` / `_wsl_cmd()` wrappers handle Windows→WSL path conversion
- `run_pipeline.py --steps ip5` runs the full FreeSurfer workflow
- `mri_segment_thalamic_nuclei` produces thalamic nuclei segmentation
- 24 of 38 labels are directly comparable with FreeSurfer output

`recon-all` (~6–10h per subject) has not yet been executed.

---

## Repository Structure

```
scripts/
├── config.py                      # Central configuration
├── run_pipeline.py                # Master pipeline runner
├── ip1_preproc.py                 # Preprocessing
├── ip2_warp.py                    # Registration
├── ip3_propagate.py               # Label propagation + morphometrics
├── ip4_refine.py                  # Boundary refinement
├── ip5_freesurfer_gt.py           # FreeSurfer pseudo-GT
├── leave_one_out_val.py           # LOO validation
├── compare_candidates.py          # Candidate comparison
├── generate_project_report.py     # HTML report generator
├── generate_morphometrics_report.py
└── utils/
    ├── metrics.py                 # Morphometrics computation
    ├── nifti_utils.py             # NIfTI I/O utilities
    └── reporter.py                # Step reporting

outputs/
├── brainseg_project_report.html   # Main report
├── morphometrics_summary/         # Morphometrics reports + CSV
├── expert_review/                 # Per-subject label reliability
├── loo_validation/                # LOO DSC results
└── <subject>/                     # Per-subject pipeline outputs
```

> NIfTI files (.nii.gz), raw MRI data, and QC images are excluded from this repository (see `.gitignore`).

---

## Usage

```bash
# Full pipeline
python scripts/run_pipeline.py

# Single step
python scripts/run_pipeline.py --steps ip1
python scripts/run_pipeline.py --steps ip2 --warp W2
python scripts/run_pipeline.py --steps ip3 --candidate W2

# FreeSurfer (after recon-all)
python scripts/run_pipeline.py --steps ip5 --skip-recon

# Generate reports
python scripts/generate_project_report.py
python scripts/generate_morphometrics_report.py
```

---

## Dependencies

- Python 3.9+
- ANTsPy
- ANTsPyNet
- nibabel
- numpy, scipy, pandas
- scikit-image
- matplotlib
- FreeSurfer 7.4+ (İP-5 only, WSL on Windows)

---

## Limitations

- All validation is LOO proxy — no manual ground-truth annotation
- Small structures (<100 mm³): DSC < 0.50, TIER-3 in most subjects
- 14 labels (STh, RN, Hb, AD, Pv, etc.) have no FreeSurfer equivalent
- Single atlas — label fusion not yet implemented
- 5 subjects only — statistical power is limited

---

## License

MIT License — see [LICENSE](LICENSE)

---

*Research use only. Not validated for clinical decision-making.*
