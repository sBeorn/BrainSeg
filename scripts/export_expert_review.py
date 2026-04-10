"""
Uzman İncelemesi İçin Temiz Veri Üretimi
==========================================
Her hasta için 3D Slicer'da açılabilir çıktılar üretir:

1. Birlesik parcellation NIfTI  (her label = ayrı integer ID)
2. 3D Slicer renk tablosu       (.ctbl)
3. Cok dilimli QC gorselleri    (axial / coronal / sagittal)
4. Guvenilirlik siniflandirmasi (CSV + Excel)
5. Uzman ozet HTML raporu

Guvenilirlik katmanlari (cok kriterli — morphometrics + LOO DSC entegre):
  TIER-1 (Guvenilir)  : atlas >= 200 mm3 VE retention >= 65%
                        VE LOO DSC >= 0.6 (varsa)
                        VE L/R simetri OK VE T1 yogunluk OK
  TIER-2 (Belirsiz)   : atlas >= 50  mm3 VE retention >= 45%
                        VEYA kriter puan 0.45-0.65 arasi
  TIER-3 (Guvenilmez) : diger (parcellation'a dahil edilmez)

Kullanim:
    python scripts/export_expert_review.py
    python scripts/export_expert_review.py --warp W2 --subject IXI002-Guys-0828
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import SUBJECTS, OUTPUT_DIR, LEFT_LABELS, RIGHT_LABELS

# ── Cikti dizini ──────────────────────────────────────────────────────────────
EXPORT_DIR = os.path.join(OUTPUT_DIR, "expert_review")

# ── Guvenilirlik esikleri ─────────────────────────────────────────────────────
TIER1_MIN_ATLAS    = 200   # mm3
TIER1_MIN_RETAIN   = 65    # %
TIER2_MIN_ATLAS    = 50    # mm3
TIER2_MIN_RETAIN   = 45    # %

# Morphometrics esikleri
LR_RATIO_MAX       = 3.0   # L/R hacim orani maksimumu (>= bu = asimetrik)
LR_RATIO_MIN       = 0.33  # L/R hacim orani minimuumu
T1_MEAN_MIN        = -0.4  # T1 ortalama yogunluk minimuumu (< bu = arka plan)
MIRROR_ERR_WARN    = 8.0   # mm — centroid simetri uyari esigi
LOO_DSC_TIER1      = 0.6   # LOO DSC >= bu ise TIER-1 destekli
LOO_DSC_TIER2      = 0.4   # LOO DSC >= bu ise TIER-2 destekli
COMPACTNESS_MIN    = 0.12  # kompaktlik minimuumu (< bu = cok duzensiz sekil)
T1_OUTLIER_WARN    = 0.15  # T1 outlier orani uyari esigi
T1_GRADIENT_MIN    = 0.05  # T1 gradient enerji minimuumu (< bu = sinir belirsiz)
BBOX_FILL_MIN      = 0.05  # bbox doluluk orani minimuumu
GLCM_HOMO_MIN      = 0.25  # GLCM homojenlik minimuumu

# LOO sonuc dosyasi
LOO_DIR = os.path.join(OUTPUT_DIR, "loo_validation")

# ── Renk paleti (TIER-1: koyu, TIER-2: orta, TIER-3: acik) ───────────────────
# Her label icin RGB (0-255) — 3D Slicer uyumlu
LABEL_COLORS = [
    (255, 50,  50),  (50,  150, 255), (50,  200, 50),  (255, 180, 0),
    (180, 50,  255), (0,   220, 200), (255, 100, 0),   (100, 100, 255),
    (200, 200, 0),   (0,   180, 100), (255, 0,   150), (100, 200, 255),
    (255, 140, 140), (140, 255, 140), (140, 140, 255), (255, 220, 140),
    (200, 140, 255), (140, 255, 220), (255, 180, 100), (180, 255, 100),
    (100, 180, 255), (255, 100, 180), (180, 100, 255), (100, 255, 180),
    (220, 100, 100), (100, 220, 100), (100, 100, 220), (220, 220, 100),
    (220, 100, 220), (100, 220, 220), (200, 160, 120), (160, 200, 120),
    (120, 160, 200), (200, 120, 160), (160, 120, 200), (120, 200, 160),
    (240, 180, 180), (180, 240, 180), (180, 180, 240), (240, 240, 180),
    (240, 180, 240),
]

# ── LOO DSC yukle ────────────────────────────────────────────────────────────

def load_loo_dsc():
    """outputs/loo_validation/loo_label_dsc.csv -> {label_name: mean_dsc}"""
    csv_path = os.path.join(LOO_DIR, "loo_label_dsc.csv")
    if not os.path.exists(csv_path):
        return {}
    loo = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                loo[row["label"]] = float(row["mean_dsc"])
            except (KeyError, ValueError):
                pass
    return loo


# ── Morphometrics yukle ───────────────────────────────────────────────────────

def load_morphometrics(subject_id, warp):
    """ip3 raporundan TUM morphometrics yukle -> {label: {tum metrikler}}"""
    rp = os.path.join(OUTPUT_DIR, subject_id, "propagate", warp,
                      f"report_IP3_Propagate_{warp}.json")
    if not os.path.exists(rp):
        return {}
    try:
        with open(rp, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return {}

    morph_data = d.get("metrics", {}).get("morphometrics", {})
    result = {}
    for label_name, m in morph_data.items():
        pos   = m.get("positional",     {})
        int_l = m.get("intensity_left",  {})
        int_r = m.get("intensity_right", {})
        geo_l = m.get("geometric_left",  {})
        geo_r = m.get("geometric_right", {})

        # compactness > 1 kucuk label marching-cubes artifakti — ust sinirla
        comp_l = min(geo_l.get("compactness", 0.0) or 0.0, 1.0)
        comp_r = min(geo_r.get("compactness", 0.0) or 0.0, 1.0)

        result[label_name] = {
            # Pozisyonel
            "lr_volume_ratio":      pos.get("lr_volume_ratio",          1.0),
            "centroid_mirror_err":  pos.get("centroid_mirror_error_mm", 0.0),
            "midline_dist_left":    pos.get("midline_distance_mm_left",  0.0),
            "midline_dist_right":   pos.get("midline_distance_mm_right", 0.0),
            # Geometrik
            "compactness_left":     comp_l,
            "compactness_right":    comp_r,
            "bbox_fill_left":       geo_l.get("bbox_fill_ratio",  0.5) or 0.5,
            "bbox_fill_right":      geo_r.get("bbox_fill_ratio",  0.5) or 0.5,
            "surface_area_left":    geo_l.get("surface_area_mm2", 0.0),
            "surface_area_right":   geo_r.get("surface_area_mm2", 0.0),
            "cc_left":              geo_l.get("connected_components", 1),
            "cc_right":             geo_r.get("connected_components", 1),
            # Intensite — temel
            "t1_mean_left":         int_l.get("T1_mean",           0.0),
            "t1_mean_right":        int_r.get("T1_mean",           0.0),
            "t1_std_left":          int_l.get("T1_std",            0.0),
            "t1_std_right":         int_r.get("T1_std",            0.0),
            "t1_outlier_left":      int_l.get("T1_outlier_ratio",  0.0) or 0.0,
            "t1_outlier_right":     int_r.get("T1_outlier_ratio",  0.0) or 0.0,
            "t1_gradient_left":     int_l.get("T1_gradient_energy",0.0) or 0.0,
            "t1_gradient_right":    int_r.get("T1_gradient_energy",0.0) or 0.0,
            "t1_log_left":          int_l.get("T1_LoG_energy",     0.0) or 0.0,
            "t1_log_right":         int_r.get("T1_LoG_energy",     0.0) or 0.0,
            # Texture (GLCM — sadece buyuk labellar icin mevcut)
            "glcm_homo_left":       int_l.get("GLCM_homogeneity"),
            "glcm_homo_right":      int_r.get("GLCM_homogeneity"),
            "glcm_entropy_left":    int_l.get("GLCM_entropy"),
            "glcm_entropy_right":   int_r.get("GLCM_entropy"),
            # T1/T2 orani (T2 mevcutsa)
            "t1_t2_ratio_left":     int_l.get("T1_mean_T2_mean_ratio"),
            "t1_t2_ratio_right":    int_r.get("T1_mean_T2_mean_ratio"),
        }
    return result


def compute_cross_subject_consistency(subjects, warp):
    """
    Tum hastalar uzerinde her label icin cross-subject tutarlilik skoru hesapla.

    Her label icin 6 temel metrigin CV (std/mean) degerini hesapla.
    Dusuk CV → tutarli propagasyon → yuksek guvenilirlik.

    Donus: {label_name: consistency_score (0-1)}
    """
    from collections import defaultdict
    label_data = defaultdict(lambda: defaultdict(list))

    for sid in subjects:
        morph = load_morphometrics(sid, warp)
        for label, m in morph.items():
            for key in ["lr_volume_ratio", "compactness_left", "compactness_right",
                        "t1_mean_left", "t1_mean_right",
                        "t1_gradient_left", "t1_gradient_right"]:
                val = m.get(key)
                if val is not None:
                    try:
                        fv = float(val)
                        if not np.isnan(fv) and not np.isinf(fv):
                            label_data[label][key].append(fv)
                    except (TypeError, ValueError):
                        pass

    consistency = {}
    for label, metrics in label_data.items():
        cvs = []
        for key, vals in metrics.items():
            if len(vals) < 2:
                continue
            arr     = np.array(vals)
            mean_ab = abs(arr.mean()) + 1e-6
            cv      = arr.std() / mean_ab
            cvs.append(min(cv, 1.0))

        if cvs:
            # CV 0 → skor 1.0 (mukemmel tutarlilik)
            # CV 0.5 → skor 0.5
            # CV 1.0 → skor 0.0
            consistency[label] = round(max(1.0 - float(np.mean(cvs)), 0.0), 3)
        else:
            consistency[label] = 0.5   # yeterli veri yok — orta

    return consistency


# ── Atlas label meta verisi ───────────────────────────────────────────────────

def load_atlas_meta():
    """Tum atlas label'larini yukle, hacimlerini hesapla."""
    meta = {}
    for f in sorted(os.listdir(LEFT_LABELS)):
        if not f.endswith(".nii.gz"):
            continue
        name = f.replace(".nii.gz", "")
        img  = nib.load(os.path.join(LEFT_LABELS, f))
        vol  = float(img.get_fdata().sum() * np.prod(img.header.get_zooms()))
        meta[name] = {"atlas_vol_mm3": round(vol, 1), "file": f}
    return meta


def classify_label(name, atlas_vol, prop_vol_l, prop_vol_r, frag_l, frag_r,
                   morph=None, loo_dsc=None, consistency=None):
    """
    Cok kriterli TIER siniflandirmasi — TUM morphometrics entegre:
      1. Atlas hacmi & hacim korunumu (temel kriter)
      2. LOO DSC skoru (pseudo ground-truth, mevcutsa)
      3. Kapsamli morphometrics skoru (8 alt kriter):
         a. L/R hacim simetrisi
         b. T1 arka plan kirliligi (dusuk yogunluk)
         c. Centroid simetri hatasi
         d. Kompaktlik (sekil duzenliligi)
         e. T1 outlier orani (doku tutarsizligi)
         f. Gradient sinir netligi
         g. BBox doluluk orani
         h. GLCM homojenlik (buyuk labellar icin)
      4. Cross-subject tutarlilik skoru (5 hasta CV analizi)
      5. Fragmentation cezasi

    Agirliklar (LOO DSC yoksa): base 0.50 + morph 0.30 + consist 0.20
    Agirliklar (LOO DSC varsa): base 0.35 + dsc 0.30 + morph 0.20 + consist 0.15

    Donus: (tier_str, quality_score_0_1, flags_list)
    """
    retain_l = prop_vol_l / (atlas_vol + 1e-6) * 100
    retain_r = prop_vol_r / (atlas_vol + 1e-6) * 100
    retain   = (retain_l + retain_r) / 2

    flags = []

    # ── 1. Temel anatomik skor (0-1) ──────────────────────────────────────────
    atlas_score  = min(atlas_vol / 200.0, 1.0)
    retain_score = min(retain / 65.0, 1.0)
    base_score   = 0.5 * atlas_score + 0.5 * retain_score

    # ── 2. LOO DSC skoru ──────────────────────────────────────────────────────
    dsc_val   = None
    dsc_score = base_score
    if loo_dsc is not None:
        dsc_val = loo_dsc.get(name, None)
        if dsc_val is not None:
            dsc_score = float(dsc_val)

    # ── 3. Kapsamli morphometrics skoru (penalti bazli, 0-1) ─────────────────
    morph_score = 1.0
    morph_m     = morph.get(name, {}) if morph else {}

    if morph_m:
        # 3a. L/R hacim simetrisi
        lr  = morph_m.get("lr_volume_ratio", 1.0) or 1.0
        lr_n = max(lr, 1.0 / max(lr, 1e-6))
        if lr_n > LR_RATIO_MAX:
            morph_score -= 0.30
            flags.append(f"LR asimetri {lr:.1f}x")
        elif lr_n > 2.0:
            morph_score -= 0.15
            flags.append(f"LR orta asimetri {lr:.1f}x")

        # 3b. T1 arka plan kirliligi
        t1l = morph_m.get("t1_mean_left",  0.0) or 0.0
        t1r = morph_m.get("t1_mean_right", 0.0) or 0.0
        if prop_vol_l > 0 and t1l < T1_MEAN_MIN:
            morph_score -= 0.20
            flags.append(f"Sol T1 yogunluk dusuk ({t1l:.2f})")
        if prop_vol_r > 0 and t1r < T1_MEAN_MIN:
            morph_score -= 0.20
            flags.append(f"Sag T1 yogunluk dusuk ({t1r:.2f})")

        # 3c. Centroid simetri hatasi
        mirror = morph_m.get("centroid_mirror_err", 0.0) or 0.0
        if mirror > MIRROR_ERR_WARN:
            morph_score -= 0.15
            flags.append(f"Sentroid hatasi {mirror:.1f}mm")

        # 3d. Kompaktlik (sekil duzenliligi)
        comp_l = min(morph_m.get("compactness_left",  0.5) or 0.0, 1.0)
        comp_r = min(morph_m.get("compactness_right", 0.5) or 0.0, 1.0)
        min_comp = min(comp_l, comp_r) if prop_vol_r > 0 else comp_l
        if min_comp < COMPACTNESS_MIN:
            morph_score -= 0.20
            flags.append(f"Dusuk kompaktlik ({min_comp:.2f})")
        elif min_comp < 0.20:
            morph_score -= 0.10
            flags.append(f"Orta kompaktlik ({min_comp:.2f})")

        # 3e. T1 outlier orani (doku tutarsizligi / sizinti tespiti)
        outr_l = morph_m.get("t1_outlier_left",  0.0) or 0.0
        outr_r = morph_m.get("t1_outlier_right", 0.0) or 0.0
        max_outr = max(outr_l, outr_r)
        if max_outr > T1_OUTLIER_WARN:
            morph_score -= 0.15
            flags.append(f"T1 outlier yuksek ({max_outr:.2f})")
        elif max_outr > 0.10:
            morph_score -= 0.08
            flags.append(f"T1 outlier orta ({max_outr:.2f})")

        # 3f. Gradient sinir netligi (sinir keskinligi)
        grad_l = morph_m.get("t1_gradient_left",  0.0) or 0.0
        grad_r = morph_m.get("t1_gradient_right", 0.0) or 0.0
        if prop_vol_l > 100 and grad_l < T1_GRADIENT_MIN:
            morph_score -= 0.10
            flags.append(f"Sol sinir belirsiz (grad={grad_l:.3f})")
        if prop_vol_r > 100 and grad_r < T1_GRADIENT_MIN:
            morph_score -= 0.10
            flags.append(f"Sag sinir belirsiz (grad={grad_r:.3f})")

        # 3g. BBox doluluk orani (cok dusuk = label bbox'ini doldurmuyorsa parcali)
        fill_l = morph_m.get("bbox_fill_left",  0.5) or 0.5
        fill_r = morph_m.get("bbox_fill_right", 0.5) or 0.5
        min_fill = min(fill_l, fill_r) if prop_vol_r > 0 else fill_l
        if min_fill < BBOX_FILL_MIN:
            morph_score -= 0.10
            flags.append(f"BBox doluluk dusuk ({min_fill:.2f})")

        # 3h. GLCM homojenlik (sadece yeterince buyuk labellar icin mevcutsa)
        homo_l = morph_m.get("glcm_homo_left")
        homo_r = morph_m.get("glcm_homo_right")
        homo_vals = [v for v in [homo_l, homo_r] if v is not None]
        if homo_vals:
            min_homo = min(homo_vals)
            if min_homo < GLCM_HOMO_MIN:
                morph_score -= 0.10
                flags.append(f"GLCM homojenlik dusuk ({min_homo:.2f})")

        morph_score = max(morph_score, 0.0)

    # ── 4. Cross-subject tutarlilik skoru ─────────────────────────────────────
    consist_score = 0.5   # yeterli veri yoksa orta deger
    if consistency is not None:
        consist_score = consistency.get(name, 0.5)
    if consist_score < 0.4:
        flags.append(f"Hastalar arasi tutarsiz (CV skoru={consist_score:.2f})")

    # ── 5. Fragmentation cezasi ───────────────────────────────────────────────
    frag_penalty = 0.0
    if frag_l and frag_r:
        frag_penalty = 0.25
        flags.append("Her iki hemisfer parcali")
    elif frag_l or frag_r:
        frag_penalty = 0.10
        flags.append("Tek hemisfer parcali")

    # ── Birlesik kalite puani ─────────────────────────────────────────────────
    # LOO DSC yoksa: base 0.50 + morph 0.30 + consist 0.20
    # LOO DSC varsa: base 0.35 + dsc 0.30 + morph 0.20 + consist 0.15
    if dsc_val is not None:
        quality = (0.35 * base_score + 0.30 * dsc_score +
                   0.20 * morph_score + 0.15 * consist_score)
    else:
        quality = 0.50 * base_score + 0.30 * morph_score + 0.20 * consist_score

    quality = max(quality - frag_penalty, 0.0)

    # ── TIER karari ───────────────────────────────────────────────────────────
    if atlas_vol < TIER2_MIN_ATLAS or retain < 30:
        return "TIER-3", round(quality, 3), flags

    if quality >= 0.60:
        return "TIER-1", round(quality, 3), flags
    if quality >= 0.40:
        return "TIER-2", round(quality, 3), flags
    return "TIER-3", round(quality, 3), flags


# ── Ana islem: tek hasta ──────────────────────────────────────────────────────

def export_subject(subject_id: str, warp: str, atlas_meta: dict,
                   loo_dsc: dict = None, morph_all: dict = None,
                   consistency: dict = None):
    print(f"\n{'='*60}")
    print(f"  {subject_id}  |  Warp: {warp}")
    print(f"{'='*60}")

    subj_out  = os.path.join(EXPORT_DIR, subject_id)
    os.makedirs(subj_out, exist_ok=True)

    prop_dir  = os.path.join(OUTPUT_DIR, subject_id, "propagate", warp)
    left_dir  = os.path.join(prop_dir, "left_labels")
    right_dir = os.path.join(prop_dir, "right_labels")
    t1_path   = os.path.join(OUTPUT_DIR, subject_id, "preproc", "T1_preproc.nii.gz")

    if not os.path.exists(left_dir):
        print(f"  [ATLA] Propagate ciktisi yok: {left_dir}")
        return None

    t1_nib  = nib.load(t1_path)
    t1_data = t1_nib.get_fdata()
    affine  = t1_nib.affine
    shape   = t1_data.shape
    vvol    = float(np.prod(t1_nib.header.get_zooms()))

    # Hasta morphometrics yukle (yoksa genel sozluk kullan)
    morph = morph_all if morph_all is not None else load_morphometrics(subject_id, warp)

    # Fragmented listesi
    frag_set = set()
    rp = os.path.join(prop_dir, f"report_IP3_Propagate_{warp}.json")
    if os.path.exists(rp):
        with open(rp, encoding="utf-8") as f:
            d = json.load(f)
        frag_set = set(d.get("metrics", {}).get("fragmented_labels", []))

    # ── 1. Label meta + siniflandirma ─────────────────────────────────────────
    label_list = sorted(atlas_meta.keys())
    records    = []

    for idx, name in enumerate(label_list, start=1):
        atlas_vol = atlas_meta[name]["atlas_vol_mm3"]
        lp   = os.path.join(left_dir,  f"{name}.nii.gz")
        rp_f = os.path.join(right_dir, f"{name}.nii.gz")

        l_arr = nib.load(lp).get_fdata()   if os.path.exists(lp)   else np.zeros(shape)
        r_arr = nib.load(rp_f).get_fdata() if os.path.exists(rp_f) else np.zeros(shape)

        l_vol  = float(l_arr.sum() * vvol)
        r_vol  = float(r_arr.sum() * vvol)
        l_frag = f"{name}_left"  in frag_set
        r_frag = f"{name}_right" in frag_set
        retain = ((l_vol + r_vol) / (2 * atlas_vol + 1e-6)) * 100

        # Cok kriterli siniflandirma
        tier, quality, flags = classify_label(
            name, atlas_vol, l_vol, r_vol, l_frag, r_frag,
            morph=morph, loo_dsc=loo_dsc, consistency=consistency,
        )

        # Morphometrics ozet (CSV icin)
        m = morph.get(name, {})
        dsc_val = loo_dsc.get(name, None) if loo_dsc else None

        def _fmt(v, nd=3):
            try:
                return round(float(v), nd) if v is not None else "—"
            except (TypeError, ValueError):
                return "—"

        consist_val = consistency.get(name, None) if consistency else None

        records.append({
            "id":               idx,
            "label":            name,
            "atlas_mm3":        round(atlas_vol, 1),
            "left_mm3":         round(l_vol, 1),
            "right_mm3":        round(r_vol, 1),
            "retention_%":      round(retain, 1),
            "left_fragmented":  l_frag,
            "right_fragmented": r_frag,
            # Temel morphometrics
            "lr_volume_ratio":  _fmt(m.get("lr_volume_ratio"), 2),
            "t1_mean_left":     _fmt(m.get("t1_mean_left"),    3),
            "t1_mean_right":    _fmt(m.get("t1_mean_right"),   3),
            "mirror_err_mm":    _fmt(m.get("centroid_mirror_err"), 1),
            # Yeni morphometrics kolonlari
            "compactness_l":    _fmt(m.get("compactness_left"),  3),
            "compactness_r":    _fmt(m.get("compactness_right"), 3),
            "t1_outlier_l":     _fmt(m.get("t1_outlier_left"),   3),
            "t1_outlier_r":     _fmt(m.get("t1_outlier_right"),  3),
            "gradient_l":       _fmt(m.get("t1_gradient_left"),  3),
            "gradient_r":       _fmt(m.get("t1_gradient_right"), 3),
            "bbox_fill_l":      _fmt(m.get("bbox_fill_left"),    3),
            "bbox_fill_r":      _fmt(m.get("bbox_fill_right"),   3),
            "glcm_homo_l":      _fmt(m.get("glcm_homo_left"),    3),
            "glcm_homo_r":      _fmt(m.get("glcm_homo_right"),   3),
            "consist_score":    _fmt(consist_val, 3),
            # Skor ve siniflandirma
            "loo_dsc":          round(dsc_val, 3) if dsc_val is not None else "—",
            "quality_score":    quality,
            "morph_flags":      "; ".join(flags) if flags else "—",
            "tier":             tier,
            "tier_note":   {
                "TIER-1": "Guvenilir — uzman dogrulamasi onerilir",
                "TIER-2": "Belirsiz  — dikkatli inceleme gerekli",
                "TIER-3": "Guvenilmez — kucuk veya kotu propagate",
            }[tier],
            "_l_arr": l_arr,
            "_r_arr": r_arr,
        })

    # ── 2. Birlesik parcellation NIfTI ────────────────────────────────────────
    # Buyuk label'lar once yazilir, kucukler (alt-cekirdekler) uzerlerine yazar
    # global / MAX_VOLUME sadece zemin referansi — sub-cekirdeklerin altina gom
    ENVELOPE_LABELS = {"global", "MAX_VOLUME"}
    parcel = np.zeros(shape, dtype=np.int16)
    sorted_recs = sorted(
        [r for r in records if r["tier"] != "TIER-3"],
        key=lambda r: -r["atlas_mm3"],   # en buyuk once
    )
    for rec in sorted_recs:
        parcel[rec["_l_arr"] > 0] = rec["id"]
        parcel[rec["_r_arr"] > 0] = rec["id"]

    parcel_path = os.path.join(subj_out, "parcellation.nii.gz")
    nib.save(nib.Nifti1Image(parcel, affine), parcel_path)
    print(f"  [OK] parcellation.nii.gz -> {parcel_path}")

    # ── 3. 3D Slicer renk tablosu (.ctbl) ────────────────────────────────────
    ctbl_path = os.path.join(subj_out, "parcellation.ctbl")
    with open(ctbl_path, "w", encoding="utf-8") as f:
        f.write("# 3D Slicer Color Table\n")
        for rec in records:
            if rec["tier"] == "TIER-3":
                continue
            col = LABEL_COLORS[(rec["id"] - 1) % len(LABEL_COLORS)]
            alpha = 255 if rec["tier"] == "TIER-1" else 200
            f.write(f"{rec['id']} {rec['label']} {col[0]} {col[1]} {col[2]} {alpha}\n")
    print(f"  [OK] parcellation.ctbl")

    # ── 4. CSV guvenilirlik raporu ────────────────────────────────────────────
    csv_path = os.path.join(subj_out, "label_reliability.csv")
    csv_cols = [
        "id","label","atlas_mm3","left_mm3","right_mm3","retention_%",
        "left_fragmented","right_fragmented",
        # Temel morphometrics
        "lr_volume_ratio","t1_mean_left","t1_mean_right","mirror_err_mm",
        # Genisletilmis morphometrics
        "compactness_l","compactness_r",
        "t1_outlier_l","t1_outlier_r",
        "gradient_l","gradient_r",
        "bbox_fill_l","bbox_fill_r",
        "glcm_homo_l","glcm_homo_r",
        "consist_score",
        # Skor ve siniflandirma
        "loo_dsc","quality_score","morph_flags",
        "tier","tier_note",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for rec in records:
            w.writerow({k: rec[k] for k in csv_cols})
    print(f"  [OK] label_reliability.csv (morphometrics + LOO DSC dahil)")

    # ── 5. Cok dilimli QC gorsel ──────────────────────────────────────────────
    _qc_multiview(t1_data, parcel, records, subject_id, warp, subj_out)

    # ── 6. Tier ozeti ─────────────────────────────────────────────────────────
    t1_cnt = sum(1 for r in records if r["tier"] == "TIER-1")
    t2_cnt = sum(1 for r in records if r["tier"] == "TIER-2")
    t3_cnt = sum(1 for r in records if r["tier"] == "TIER-3")
    print(f"\n  Siniflandirma: TIER-1={t1_cnt}  TIER-2={t2_cnt}  TIER-3={t3_cnt}  (toplam {len(records)})")

    return {
        "subject_id": subject_id,
        "warp":       warp,
        "tier1_count": t1_cnt,
        "tier2_count": t2_cnt,
        "tier3_count": t3_cnt,
        "records":    records,
    }


# ── QC gorsel ─────────────────────────────────────────────────────────────────

def _qc_multiview(t1, parcel, records, subject_id, warp, out_dir):
    """
    Uzman icin QC gorseli:
    - Satir 1: Axial — parcellation'in bulundugu talamusik bolge
    - Satir 2: Coronal — bilateral simetri kontrolu
    - Satir 3: Sagittal — yapilarin on-arka konumu
    - Satir 4: Zoom — talamus bolgesi buyutulmush (3 kesit)
    """
    # Renk haritasi (label_id -> RGB)
    color_map = {}
    for rec in records:
        if rec["tier"] == "TIER-3":
            continue
        col = LABEL_COLORS[(rec["id"] - 1) % len(LABEL_COLORS)]
        color_map[rec["id"]] = tuple(c / 255.0 for c in col)

    sx, sy, sz = t1.shape
    t1_p2  = np.percentile(t1[t1 > 0], 2)  if (t1 > 0).any() else 0
    t1_p99 = np.percentile(t1[t1 > 0], 99) if (t1 > 0).any() else 1

    # Parcellation'in oldugu z araligini bul (talamus seviyesi)
    z_has_label = np.where(parcel.sum(axis=(0, 1)) > 0)[0]
    y_has_label = np.where(parcel.sum(axis=(0, 2)) > 0)[0]
    x_has_label = np.where(parcel.sum(axis=(1, 2)) > 0)[0]

    if len(z_has_label) == 0:
        z_has_label = np.arange(sz // 4, 3 * sz // 4)
        y_has_label = np.arange(sy // 4, 3 * sy // 4)
        x_has_label = np.arange(sx // 4, 3 * sx // 4)

    z_lo, z_hi = z_has_label.min(), z_has_label.max()
    y_lo, y_hi = y_has_label.min(), y_has_label.max()
    x_lo, x_hi = x_has_label.min(), x_has_label.max()

    n = 5  # her satirda kesit sayisi
    z_idxs = np.linspace(z_lo, z_hi, n, dtype=int)
    y_idxs = np.linspace(y_lo, y_hi, n, dtype=int)
    x_idxs = np.linspace(x_lo, x_hi, n, dtype=int)

    # Zoom icin orta 3 axial kesit, talamus bolgesine odakli
    z_mid = (z_lo + z_hi) // 2
    z_zoom = [z_mid - 6, z_mid, z_mid + 6]

    fig = plt.figure(figsize=(n * 4, 17), facecolor="#111111")
    fig.suptitle(
        f"BrainSeg Parcellation QC\n{subject_id}  |  Warp: {warp}  |  "
        f"TIER-1 (guvenlir): opak  |  TIER-2 (belirsiz): saydam",
        fontsize=11, color="white", y=0.995,
    )

    gs_main = fig.add_gridspec(4, n, hspace=0.08, wspace=0.04,
                                top=0.97, bottom=0.08)

    def _make_rgba(seg_sl):
        """2D slice -> RGBA array."""
        h, w = seg_sl.shape
        rgba = np.zeros((h, w, 4), dtype=float)
        for lid, rgb in color_map.items():
            m = seg_sl == lid
            if not m.any():
                continue
            # TIER-1 vs TIER-2 alpha
            rec = next((r for r in records if r["id"] == lid), None)
            alpha = 0.82 if (rec and rec["tier"] == "TIER-1") else 0.50
            rgba[m, :3] = rgb
            rgba[m,  3] = alpha
        return rgba

    def _draw(ax, bg_sl, seg_sl, title, zoom_box=None):
        bg_T = bg_sl.T
        ax.imshow(bg_T, cmap="gray", origin="lower",
                  vmin=t1_p2, vmax=t1_p99, interpolation="bilinear")
        rgba = _make_rgba(seg_sl)
        ax.imshow(rgba.transpose(1, 0, 2), origin="lower", interpolation="none")
        if zoom_box:
            from matplotlib.patches import Rectangle
            r = Rectangle((zoom_box[0], zoom_box[2]),
                           zoom_box[1]-zoom_box[0], zoom_box[3]-zoom_box[2],
                           linewidth=1.5, edgecolor="yellow", facecolor="none")
            ax.add_patch(r)
        ax.set_title(title, fontsize=7.5, color="#cccccc", pad=2)
        ax.axis("off")

    # Satir 0: Axial
    for i, z in enumerate(z_idxs):
        ax = fig.add_subplot(gs_main[0, i])
        _draw(ax, t1[:, :, z], parcel[:, :, z], f"Axial  z={z}")

    # Satir 1: Coronal
    for i, y in enumerate(y_idxs):
        ax = fig.add_subplot(gs_main[1, i])
        _draw(ax, t1[:, y, :], parcel[:, y, :], f"Coronal y={y}")

    # Satir 2: Sagittal
    for i, x in enumerate(x_idxs):
        ax = fig.add_subplot(gs_main[2, i])
        _draw(ax, t1[x, :, :], parcel[x, :, :], f"Sagittal x={x}")

    # Satir 3: Zoom (talamus merkezi)
    # 3 kesit merkez, 2 bos panel legenda ayrildi
    for i, z in enumerate(z_zoom):
        ax = fig.add_subplot(gs_main[3, i])
        bg = t1[:, :, z]
        sg = parcel[:, :, z]
        # Talamus bolgesi crop
        pad = 15
        xc = np.clip([x_lo - pad, x_hi + pad], 0, sx)
        yc = np.clip([y_lo - pad, y_hi + pad], 0, sy)
        bg_crop = bg[xc[0]:xc[1], yc[0]:yc[1]]
        sg_crop = sg[xc[0]:xc[1], yc[0]:yc[1]]
        _draw(ax, bg_crop, sg_crop, f"ZOOM Axial z={z}")

    # Legend paneli (son 2 kolum, son satir)
    ax_leg = fig.add_subplot(gs_main[3, 3:])
    ax_leg.set_facecolor("#111111")
    ax_leg.axis("off")

    t1_labels = [r for r in records if r["tier"] == "TIER-1"]
    t2_labels = [r for r in records if r["tier"] == "TIER-2"]

    leg_lines = ["TIER-1 — Guvenilir Labellar:"]
    for r in t1_labels:
        col = LABEL_COLORS[(r["id"]-1) % len(LABEL_COLORS)]
        hex_c = "#{:02x}{:02x}{:02x}".format(*col)
        leg_lines.append(f"  [#{r['id']:02d}] {r['label']}  "
                         f"{r['left_mm3']:.0f}|{r['right_mm3']:.0f} mm3")
    leg_lines += ["", "TIER-2 — Belirsiz (dikkatli incele):"]
    for r in t2_labels[:12]:
        leg_lines.append(f"  [#{r['id']:02d}] {r['label']}  "
                         f"{r['left_mm3']:.0f}|{r['right_mm3']:.0f} mm3")
    if len(t2_labels) > 12:
        leg_lines.append(f"  ... +{len(t2_labels)-12} label daha")

    ax_leg.text(0.02, 0.98, "\n".join(leg_lines),
                transform=ax_leg.transAxes,
                va="top", ha="left", fontsize=6.5,
                color="#dddddd", fontfamily="monospace")

    qc_path = os.path.join(out_dir, "QC_multiview.png")
    plt.savefig(qc_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] QC_multiview.png")


# ── Ozet HTML raporu ──────────────────────────────────────────────────────────

def write_summary_html(all_results, warp, out_dir):
    html_path = os.path.join(out_dir, "expert_summary.html")

    tier_colors = {"TIER-1": "#2ecc71", "TIER-2": "#f39c12", "TIER-3": "#e74c3c"}

    html = ["""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<title>BrainSeg Uzman İnceleme Raporu</title>
<style>
  body { font-family: Arial, sans-serif; background: #1a1a2e; color: #e0e0e0; margin: 20px; }
  h1   { color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 8px; }
  h2   { color: #a0c4ff; margin-top: 30px; }
  h3   { color: #ccc; }
  .warning { background: #3a2a00; border-left: 4px solid #f39c12;
             padding: 12px; margin: 10px 0; border-radius: 4px; }
  .info    { background: #0a2a3a; border-left: 4px solid #00d4ff;
             padding: 12px; margin: 10px 0; border-radius: 4px; }
  table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }
  th    { background: #16213e; color: #00d4ff; padding: 8px; text-align: left; }
  td    { padding: 6px 8px; border-bottom: 1px solid #2a2a4a; }
  tr:hover td { background: #1e1e3e; }
  .t1 { color: #2ecc71; font-weight: bold; }
  .t2 { color: #f39c12; }
  .t3 { color: #e74c3c; }
  .subject-box { background: #16213e; border-radius: 8px; padding: 15px; margin: 20px 0; }
  img { max-width: 100%; border-radius: 6px; margin: 10px 0; }
  .method { background: #0d1b2a; padding: 10px; border-radius: 4px;
            font-family: monospace; font-size: 12px; }
</style>
</head>
<body>
"""]

    html.append(f"""
<h1>BrainSeg — Uzman Inceleme Raporu</h1>
<div class="info">
  <strong>Yontem:</strong> Atlas bazli talamusik cekirdek segmentasyonu (Multi-label propagation)<br>
  <strong>Atlas:</strong> IXI (MNI152 uzayi, talamik sub-cekirdekler)<br>
  <strong>Registration:</strong> ANTsPy SyN — Warp: <b>{warp}</b> (Affine+SyN, Mattes MI)<br>
  <strong>Hastalar:</strong> {len(all_results)} (IXI veri seti)<br>
  <strong>Tarih:</strong> 2026-04-08
</div>
<div class="warning">
  <strong>UYARI:</strong> Bu ciktilar uzman incelemesi icin hazirlanmistir.
  Klinik karar vermek icin kullanilmamalidir. Ground-truth validasyonu yapilmamistir.
  Lütfen her hastanin QC görselini ve guvenilirlik tablosunu dikkatlice inceleyin.
</div>

<h2>Guvenilirlik Siniflandirmasi (Cok Kriterli)</h2>
<table>
<tr><th>Tier</th><th>Kalite Puani</th><th>Kriterler</th><th>Anlami</th></tr>
<tr><td class="t1">TIER-1</td>
    <td class="t1">&ge;0.60</td>
    <td>Atlas &ge;200mm&sup3; + Ret &ge;65% + LOO DSC &ge;0.6 + L/R simetri + T1 yogunluk OK</td>
    <td>Anatomik olarak guvenilir; uzman dogrulamasi onerilir</td></tr>
<tr><td class="t2">TIER-2</td>
    <td class="t2">0.40-0.60</td>
    <td>Atlas &ge;50mm&sup3; + orta LOO DSC veya morphometri uyarilari</td>
    <td>Dikkatli inceleme gerekli; sonuclar yoruma acik</td></tr>
<tr><td class="t3">TIER-3</td>
    <td class="t3">&lt;0.40</td>
    <td>Kucuk atlas (&lt;50mm&sup3;) veya dusuk LOO DSC veya zayif morphometri</td>
    <td>Guvenilmez — parcellation haritasina dahil edilmedi</td></tr>
</table>
<div class="info">
<strong>Kalite puani bilesenleri:</strong>
LOO DSC (agirlik 0.35) + Temel anatomik skor (0.40) + Morphometri skoru (0.25)<br>
<strong>Morphometri kontrolleri:</strong> L/R hacim simetri (&lt;3x), T1 yogunluk (&gt;-0.4),
Centroid simetri hatasi (&lt;8mm), Fragmentation cezasi
</div>
""")

    for res in all_results:
        sid   = res["subject_id"]
        recs  = res["records"]
        t1c   = res["tier1_count"]
        t2c   = res["tier2_count"]
        t3c   = res["tier3_count"]
        qc_rel = f"{sid}/QC_multiview.png"

        html.append(f"""
<div class="subject-box">
<h2>Hasta: {sid}</h2>
<p>TIER-1: <span class="t1">{t1c} label</span> &nbsp;|&nbsp;
   TIER-2: <span class="t2">{t2c} label</span> &nbsp;|&nbsp;
   TIER-3: <span class="t3">{t3c} label (parcellation disinda)</span></p>

<h3>Parcellation Gorsel Kontrol</h3>
<img src="{qc_rel}" alt="QC multiview {sid}">

<h3>Label Guvenilirlik Tablosu (Morphometrics + LOO DSC entegre)</h3>
<table>
<tr>
  <th>Label</th><th>Atlas mm&sup3;</th>
  <th>Sol mm&sup3;</th><th>Sag mm&sup3;</th><th>Korunum%</th>
  <th>L/R Oran</th><th>T1 Sol</th><th>T1 Sag</th>
  <th>LOO DSC</th><th>Kalite</th><th>Parca</th>
  <th>Uyarilar</th><th>Tier</th>
</tr>
""")
        for rec in sorted(recs, key=lambda r: (r["tier"], -r["quality_score"])):
            t_cls  = rec["tier"].lower().replace("-", "")
            frag_s = ""
            if rec["left_fragmented"] and rec["right_fragmented"]:
                frag_s = '<span style="color:#e74c3c">L+R</span>'
            elif rec["left_fragmented"]:
                frag_s = '<span style="color:#f39c12">L</span>'
            elif rec["right_fragmented"]:
                frag_s = '<span style="color:#f39c12">R</span>'
            else:
                frag_s = '<span style="color:#888">—</span>'

            dsc_val = rec["loo_dsc"]
            if dsc_val != "—":
                dsc_color = "#2ecc71" if dsc_val >= 0.6 else ("#f39c12" if dsc_val >= 0.4 else "#e74c3c")
                dsc_str = f'<span style="color:{dsc_color}">{dsc_val:.3f}</span>'
            else:
                dsc_str = '<span style="color:#666">—</span>'

            lr = rec["lr_volume_ratio"]
            lr_color = "#e74c3c" if (isinstance(lr, float) and (lr > LR_RATIO_MAX or lr < LR_RATIO_MIN)) else "#ccc"

            t1l = rec["t1_mean_left"]
            t1r = rec["t1_mean_right"]
            t1l_color = "#e74c3c" if (isinstance(t1l, float) and t1l < T1_MEAN_MIN) else "#ccc"
            t1r_color = "#e74c3c" if (isinstance(t1r, float) and t1r < T1_MEAN_MIN) else "#ccc"

            q = rec["quality_score"]
            q_color = "#2ecc71" if q >= 0.6 else ("#f39c12" if q >= 0.4 else "#e74c3c")

            flags_html = rec["morph_flags"] if rec["morph_flags"] != "—" else '<span style="color:#666">—</span>'

            html.append(
                f'<tr>'
                f'<td><b>{rec["label"]}</b></td>'
                f'<td>{rec["atlas_mm3"]}</td>'
                f'<td>{rec["left_mm3"]}</td>'
                f'<td>{rec["right_mm3"]}</td>'
                f'<td>{rec["retention_%"]:.0f}%</td>'
                f'<td style="color:{lr_color}">{lr if isinstance(lr, float) else lr}</td>'
                f'<td style="color:{t1l_color}">{t1l if isinstance(t1l, float) else t1l}</td>'
                f'<td style="color:{t1r_color}">{t1r if isinstance(t1r, float) else t1r}</td>'
                f'<td>{dsc_str}</td>'
                f'<td style="color:{q_color}">{q:.2f}</td>'
                f'<td>{frag_s}</td>'
                f'<td style="font-size:11px;color:#f39c12">{flags_html}</td>'
                f'<td class="{t_cls}">{rec["tier"]}</td>'
                f'</tr>\n'
            )
        html.append("</table></div>\n")

    html.append("""
<h2>3D Slicer'da Nasil Acilir?</h2>
<div class="method">
1. 3D Slicer'i acin<br>
2. File > Add Data > parcellation.nii.gz dosyasini yukleyin (Labelmap olarak)<br>
3. Ayni sekilde T1_preproc.nii.gz dosyasini yukleyin (Volume olarak)<br>
4. Modules > Colors > Load Color Table > parcellation.ctbl secin<br>
5. Slice view'da T1'i zemin, parcellation'i overlay olarak ayarlayin<br>
6. Her label ID'si label_reliability.csv dosyasindaki isimle eslesiyor
</div>

<h2>Metodoloji Notlari</h2>
<div class="info">
<ul>
<li>Registration: ANTsPy SyN (Symmetric Diffeomorphic Normalization), Mattes MI metrik</li>
<li>Propagation: Nearest-neighbour interpolation ile forward transform uygulamasi</li>
<li>Refinement uygulanmamistir — ham propagation ciktilari kullanilmaktadir</li>
<li>Hacim korunumu = (propagated vol / atlas vol) x 100</li>
<li>Parcalanma: bir label birden fazla ayrik bolgeye bolunen yapilar icin isaretlenmistir</li>
<li>Kucuk cekirdekler (< 50 mm3 atlas boyutu) cok dusuk cozunurluklu verilerde guvenilir degildir</li>
</ul>
</div>

</body></html>
""")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"\n  [OK] expert_summary.html -> {html_path}")
    return html_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warp",    default="W2",
                        choices=["W1", "W2", "W3"])
    parser.add_argument("--subject", default=None)
    args = parser.parse_args()

    subjects   = [args.subject] if args.subject else SUBJECTS
    atlas_meta = load_atlas_meta()
    loo_dsc    = load_loo_dsc()

    os.makedirs(EXPORT_DIR, exist_ok=True)

    if loo_dsc:
        print(f"  LOO DSC yuklendi: {len(loo_dsc)} label")
    else:
        print("  [UYARI] LOO DSC bulunamadi — sadece anatomik kriterler kullanilacak")

    print(f"\n{'='*60}")
    print(f"  Uzman Inceleme Paketi — {len(subjects)} hasta, Warp={args.warp}")
    print(f"  Morphometrics + LOO DSC entegre TIER siniflandirmasi")
    print(f"{'='*60}")

    # Cross-subject tutarlilik: tum hastalar uzerinden CV analizi
    print(f"\n  Cross-subject tutarlilik hesaplaniyor ({len(subjects)} hasta)...")
    consistency = compute_cross_subject_consistency(subjects, args.warp)
    print(f"  Tutarlilik skoru hesaplandi: {len(consistency)} label")
    if consistency:
        scores = list(consistency.values())
        print(f"  Ort. tutarlilik: {np.mean(scores):.3f}  "
              f"Min: {min(scores):.3f}  Max: {max(scores):.3f}")

    all_results = []
    for sid in subjects:
        morph = load_morphometrics(sid, args.warp)
        print(f"  Morphometrics yuklendi: {len(morph)} label ({sid})")
        res = export_subject(sid, args.warp, atlas_meta,
                             loo_dsc=loo_dsc, morph_all=morph,
                             consistency=consistency)
        if res:
            all_results.append(res)

    if all_results:
        html_path = write_summary_html(all_results, args.warp, EXPORT_DIR)
        print(f"\n{'='*60}")
        print(f"  TAMAMLANDI")
        print(f"  Cikti dizini: {EXPORT_DIR}")
        print(f"  Uzman raporu: {html_path}")
        print(f"{'='*60}\n")
        print("  Her hasta icin:")
        print("    parcellation.nii.gz  — 3D Slicer'a yukleyin (Labelmap)")
        print("    parcellation.ctbl    — Renk tablosu")
        print("    label_reliability.csv — Guvenilirlik tablosu")
        print("    QC_multiview.png     — Gorsel kontrol")
        print()


if __name__ == "__main__":
    main()
