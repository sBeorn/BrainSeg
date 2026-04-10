"""
BrainSeg Pipeline — Merkezi Konfigürasyon
Tüm scriptler bu dosyayı import eder; yol ve parametre değişiklikleri
yalnızca buradan yapılır.
"""

import os
import platform

# ─── Platform ───────────────────────────────────────────────────────────────
IS_WINDOWS = platform.system() == "Windows"

# ─── Proje Kök Dizini ───────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ─── Veri Yolları ───────────────────────────────────────────────────────────
DATA_DIR      = os.path.join(PROJECT_ROOT, "BrainSeg", "Data")
ATLAS_DIR     = os.path.join(DATA_DIR, "Atlas")
PATIENT_DIR   = os.path.join(DATA_DIR, "Patient")

ATLAS_T1      = os.path.join(ATLAS_DIR, "MNI152_T1_1mm.nii.gz")
LEFT_LABELS   = os.path.join(ATLAS_DIR, "left-vols-1mm")
RIGHT_LABELS  = os.path.join(ATLAS_DIR, "right-vols-1mm")

# ─── Çıktı Dizini ───────────────────────────────────────────────────────────
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, "outputs")
ATLAS_OUT     = os.path.join(OUTPUT_DIR, "atlas")

# ─── Hasta Listesi ──────────────────────────────────────────────────────────
SUBJECTS = [
    "IXI002-Guys-0828",
    "IXI012-HH-1211",
    "IXI013-HH-1212",
    "IXI015-HH-1258",
    "IXI016-Guys-0697",
]

# ─── Warp Adayları ──────────────────────────────────────────────────────────
WARP_CANDIDATES = {
    "W1": {
        "type_of_transform": "SyNCC",
        "description": "Affine + SyN, CC similarity (standart)",
        "syn_metric": "CC",
        "syn_metric_params": [4],          # radius
        "grad_step": 0.1,
        "flow_sigma": 3.0,
        "total_sigma": 0.0,
        "aff_metric": "mattes",
        "syn_iterations": [100, 70, 50, 20],
    },
    "W2": {
        "type_of_transform": "SyN",
        "description": "Affine + SyN, MI similarity (multi-modal)",
        "syn_metric": "mattes",
        "syn_metric_params": [32],
        "grad_step": 0.1,
        "flow_sigma": 3.0,
        "total_sigma": 0.0,
        "aff_metric": "mattes",
        "syn_iterations": [100, 70, 50, 20],
    },
    "W3": {
        "type_of_transform": "SyNCC",
        "description": "Affine + SyN (güçlü reg.), CC — küçük yapı overfit'ini önler",
        "syn_metric": "CC",
        "syn_metric_params": [4],
        "grad_step": 0.05,                 # Daha küçük adım
        "flow_sigma": 4.0,                 # Daha fazla smoothing
        "total_sigma": 1.0,
        "aff_metric": "mattes",
        "syn_iterations": [200, 100, 70, 20],
    },
}

# ─── Refine Adayları ────────────────────────────────────────────────────────
REFINE_CANDIDATES = {
    "R1": {
        "method": "gaussian_boundary",
        "description": "Gaussian intensity-based local boundary refinement",
        "sigma": 1.0,
        "n_iter": 5,
    },
    "R2": {
        "method": "active_contour",
        "description": "Graph-cut / active contour, T1+T2 joint",
        "alpha": 0.015,
        "sigma": 3.0,
        "n_iter": 200,
    },
    "R3": {
        "method": "morphological",
        "description": "Morphological correction only (baseline)",
        "morph_op": "closing",
        "radius": 1,
    },
}

# ─── Label Grupları ─────────────────────────────────────────────────────────
LABEL_GROUPS = {
    "coarse": ["global", "thalamus_body", "MAX_VOLUME"],
    "mid":    ["MDmc", "MDpc", "LP", "Pf", "CM", "VLa", "VLpd", "VLpv", "VAmc", "VApc"],
    "fine":   ["STh", "RN", "Hb", "LGNmc", "LGNpc", "mtt", "SG", "Li", "Pv"],
}

# ─── Pseudo-label dışlamaları (morfometrik analize dahil edilmez) ─────────────
# Bunlar bütün talamus veya hacim referansı için atlastaki sentetik etiketlerdir.
EXCLUDE_LABELS = {"MAX_VOLUME", "global", "thalamus_body"}

# ─── TIER Sınıflandırma Eşikleri ────────────────────────────────────────────
# Morphometrics kalite skoruna göre label güvenilirlik sınıfı belirlenir.
# Bu değerleri değiştirmek tüm pipeline çıktılarını etkiler.
TIER_THRESHOLDS = {
    "tier1_min_quality":  0.60,   # quality_score >= bu → TIER-1
    "tier2_min_quality":  0.40,   # quality_score >= bu → TIER-2, altı → TIER-3
    "tier1_min_atlas_mm3": 50,    # atlas referans hacmi minimum (mm³)
    "tier1_min_retention": 45.0,  # propagated hacmin korunma oranı minimum (%)
    "tier1_max_lr_ratio":  3.0,   # sol/sağ hacim oranı maksimum (simetri)
    "loo_dsc_good":        0.70,  # LOO DSC ≥ bu → klinik kabul eşiği
    "loo_dsc_acceptable":  0.50,  # LOO DSC ≥ bu → kabul edilebilir
}

# ─── Kalite Eşikleri ────────────────────────────────────────────────────────
THRESHOLDS = {
    "jacobian_neg_ratio_fail":   0.001,   # > %0.1 → aday elenir
    "jacobian_neg_ratio_warn":   0.0005,
    "inverse_consistency_fail":  1.0,     # > 1 mm → uyarı
    "lr_volume_ratio_min":       0.80,
    "lr_volume_ratio_max":       1.20,
    "volume_drift_fail":         0.15,    # > %15 → uyarı
    "vessel_overlap_fail":       0,       # herhangi bir overlap → hata
    "max_connected_components":  1,       # label parçalanmaması
}

# ─── FreeSurfer ─────────────────────────────────────────────────────────────
# Windows'ta WSL içindeki kuruluma işaret eder; Linux/WSL'de standart env değişkeni.
FREESURFER_HOME = os.environ.get(
    "FREESURFER_HOME",
    os.path.expanduser("~/freesurfer"),
)

# ─── QC Görsel Ayarları ─────────────────────────────────────────────────────
QC_DPI        = 300
QC_FIGSIZE    = (12, 4)
QC_N_SLICES   = 3       # Her eksende gösterilecek kesit sayısı

# ─── Yeniden Örnekleme ──────────────────────────────────────────────────────
TARGET_SPACING = (1.0, 1.0, 1.0)   # mm, isotropic

# ─── Birleşik Pipeline Skoru Ağırlıkları ────────────────────────────────────
SCORE_WEIGHTS = {
    "warp":      0.30,
    "propagate": 0.25,
    "refine":    0.25,
    "stability": 0.10,
    "runtime":   0.10,
}
