"""
İP-3: Propagate ve Label Kalite Analizi
========================================
Atlas label'larını forward transform ile subject space'e aktarır (NN interpolation).
Her label için kapsamlı morphometrics hesaplar (Bölüm 6.2).

Her hasta × warp adayı için:
- Sol/sağ tüm label'lar → propagate
- Geometrik, Konumsal, Görünüm metrikleri
- Parçalanma tespiti (CC > 1)
- Hemisfer ihlali kontrolü
- L/R simetri analizi
- label_quality_report.json

Kullanım:
    python scripts/ip3_propagate.py
    python scripts/ip3_propagate.py --subject IXI002-Guys-0828 --candidate W1
    python scripts/ip3_propagate.py --labels-only STh,RN,Hb
"""

import argparse
import json
import os
import sys

# Windows terminal encoding fix
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(_sys.stderr, "reconfigure"):
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import ants

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import (
    SUBJECTS, OUTPUT_DIR, LEFT_LABELS, RIGHT_LABELS,
    WARP_CANDIDATES, THRESHOLDS, QC_DPI
)
from scripts.utils.reporter import StepReporter
from scripts.utils.nifti_utils import load_nifti, save_nifti, get_voxel_volume
from scripts.utils.metrics import (
    compute_geometric_morphometrics,
    compute_positional_morphometrics,
    compute_intensity_morphometrics,
    compute_label_quality_report,
)


# ──────────────────────────────────────────────────────────────────────────────

def propagate_subject_candidate(
    subject_id: str,
    candidate_name: str,
    label_filter: list = None,
) -> dict:
    """
    Tek hasta + tek warp adayı için tüm label'ları propagate et.
    label_filter: ['STh', 'RN'] gibi liste; None → tümü
    """
    preproc_dir = os.path.join(OUTPUT_DIR, subject_id, "preproc")
    warp_dir    = os.path.join(OUTPUT_DIR, subject_id, "warp", candidate_name)
    out_dir     = os.path.join(OUTPUT_DIR, subject_id, "propagate", candidate_name)
    left_out    = os.path.join(out_dir, "left_labels")
    right_out   = os.path.join(out_dir, "right_labels")
    os.makedirs(left_out,  exist_ok=True)
    os.makedirs(right_out, exist_ok=True)

    rep = StepReporter(subject_id, f"IP3_Propagate_{candidate_name}", out_dir)

    try:
        # ── Jacobian fail kontrolü ─────────────────────────────────────────
        sim_json_path = os.path.join(warp_dir, "similarity_metric.json")
        if os.path.exists(sim_json_path):
            with open(sim_json_path) as f:
                sim = json.load(f)
            if sim.get("jacobian_fail", False):
                msg = (f"{candidate_name} Jacobian testi başarısız "
                       f"(neg_ratio={sim.get('jacobian_neg_ratio')}) — propagate atlandı")
                print(f"  ⊘  {msg}")
                rep.set_flag("jacobian_fail_skip", True, msg)
                return rep.finish("SKIPPED")

        # ── Girdi yolları ──────────────────────────────────────────────────
        t1_path   = os.path.join(preproc_dir, "T1_preproc.nii.gz")
        t2_path   = os.path.join(preproc_dir, "T2_to_T1_preproc.nii.gz")
        mask_path = os.path.join(preproc_dir, "brain_mask.nii.gz")

        for p in [t1_path, mask_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Girdi yok: {p}")

        fixed_ants = ants.image_read(t1_path)
        t1_nib     = load_nifti(t1_path)
        t2_nib     = load_nifti(t2_path) if os.path.exists(t2_path) else None
        mask_nib   = load_nifti(mask_path)
        t1_data    = t1_nib.get_fdata()
        t2_data    = t2_nib.get_fdata() if t2_nib else None
        mask_data  = mask_nib.get_fdata()
        spacing    = t1_nib.header.get_zooms()[:3]
        voxel_vol  = float(np.prod(spacing))

        # ── Transform listesi ──────────────────────────────────────────────
        fwd_transforms = _find_forward_transforms(warp_dir)
        rep.log("fwd_transforms_found", len(fwd_transforms))

        # ── Label dosyalarını bul ──────────────────────────────────────────
        label_names = _get_label_names(LEFT_LABELS, label_filter)
        rep.log("n_labels", len(label_names))

        all_quality   = {}
        all_morpho    = {}
        problem_labels = []
        fragmented     = []
        hemisphere_viol = []

        # ── Propagate döngüsü ──────────────────────────────────────────────
        for label_name in label_names:
            lpath = os.path.join(LEFT_LABELS,  f"{label_name}.nii.gz")
            rpath = os.path.join(RIGHT_LABELS, f"{label_name}.nii.gz")

            if not os.path.exists(lpath):
                print(f"  ⚠  Sol label yok, atlandı: {label_name}")
                continue

            # ── Sol label propagate ────────────────────────────────────────
            l_prop  = _propagate_label(fixed_ants, lpath, fwd_transforms)
            l_arr   = _defrag(l_prop.numpy())          # fragmentation duzelt
            l_out   = os.path.join(left_out,  f"{label_name}.nii.gz")
            save_nifti(l_arr.astype(np.uint8), t1_nib.affine, l_out)

            # ── Sağ label propagate ────────────────────────────────────────
            if os.path.exists(rpath):
                r_prop = _propagate_label(fixed_ants, rpath, fwd_transforms)
                r_arr  = _defrag(r_prop.numpy())       # fragmentation duzelt
                r_out  = os.path.join(right_out, f"{label_name}.nii.gz")
                save_nifti(r_arr.astype(np.uint8), t1_nib.affine, r_out)
            else:
                r_arr = np.zeros_like(l_arr)

            # ── Kalite raporu ──────────────────────────────────────────────
            qr_l = compute_label_quality_report(
                l_arr, spacing, label_name, "left", mask_data
            )
            qr_r = compute_label_quality_report(
                r_arr, spacing, label_name, "right", mask_data
            )

            # L/R hacim oranı — sorun tespiti
            vol_l = qr_l["volume_mm3"]
            vol_r = qr_r["volume_mm3"]
            lr_ratio = vol_l / (vol_r + 1e-10)
            qr_l["LR_volume_ratio"] = round(lr_ratio, 4)

            # Uyarı koşulları
            if qr_l["connected_components"] > THRESHOLDS["max_connected_components"]:
                fragmented.append(f"{label_name}_left")
                problem_labels.append(f"{label_name}_left")
            if qr_r["connected_components"] > THRESHOLDS["max_connected_components"]:
                fragmented.append(f"{label_name}_right")
                problem_labels.append(f"{label_name}_right")

            if not (THRESHOLDS["lr_volume_ratio_min"] < lr_ratio
                    < THRESHOLDS["lr_volume_ratio_max"]):
                problem_labels.append(f"{label_name}_LR_asymmetry")

            all_quality[label_name] = {"left": qr_l, "right": qr_r}

            # ── Kapsamlı Morphometrics ─────────────────────────────────────
            geom_l  = compute_geometric_morphometrics(l_arr, spacing)
            geom_r  = compute_geometric_morphometrics(r_arr, spacing)
            posit   = compute_positional_morphometrics(
                l_arr, r_arr, mask_data, spacing, label_name
            )
            intens_l = compute_intensity_morphometrics(l_arr, t1_data, t2_data)
            intens_r = compute_intensity_morphometrics(r_arr, t1_data, t2_data)

            all_morpho[label_name] = {
                "geometric_left":   geom_l,
                "geometric_right":  geom_r,
                "positional":       posit,
                "intensity_left":   intens_l,
                "intensity_right":  intens_r,
            }

            # ── QC zoom görseli (küçük çekirdekler) ───────────────────────
            if vol_l < 2000 or label_name in ["STh", "RN", "Hb", "LGNmc", "LGNpc"]:
                _qc_label_zoom(
                    t1_data, l_arr, label_name, spacing,
                    os.path.join(out_dir, f"label_zoom_{label_name}.png"),
                )

        # ── Özet raporlama ─────────────────────────────────────────────────
        rep.log("problem_labels_count", len(set(problem_labels)))
        rep.log("fragmented_labels",    fragmented)
        rep.add_metric_group("label_quality",   all_quality)
        rep.add_metric_group("morphometrics",   all_morpho)

        # label_quality_report.json — ayrıca yaz
        lq_path = os.path.join(out_dir, "label_quality_report.json")
        with open(lq_path, "w", encoding="utf-8") as f:
            json.dump({
                "subject_id":    subject_id,
                "candidate":     candidate_name,
                "problem_labels": list(set(problem_labels)),
                "fragmented":    fragmented,
                "hemisphere_violations": hemisphere_viol,
                "labels":        all_quality,
            }, f, indent=2, ensure_ascii=False)

        rep.add_file("label_quality_report", lq_path)
        rep.add_file("left_labels_dir",  left_out)
        rep.add_file("right_labels_dir", right_out)

        # ── Genel T1/label overlay ─────────────────────────────────────────
        # thalamus_body veya global ile overlay
        ref_label = "thalamus_body" if "thalamus_body" in label_names else label_names[0]
        ref_arr_l = np.zeros_like(t1_data)
        ref_path  = os.path.join(left_out, f"{ref_label}.nii.gz")
        if os.path.exists(ref_path):
            ref_arr_l = load_nifti(ref_path).get_fdata()
        _qc_label_overlay(
            t1_data, ref_arr_l,
            title=f"{subject_id} — {candidate_name} | {ref_label}",
            path=os.path.join(out_dir, "overlay_T1_labels_axial.png"),
        )
        rep.add_file("QC_label_overlay",
                     os.path.join(out_dir, "overlay_T1_labels_axial.png"))

    except Exception as exc:
        rep.record_exception(exc)
        return rep.finish("FAILED")

    return rep.finish("SUCCESS")


# ──────────────────────────────────────────────────────────────────────────────
# Yardımcı
# ──────────────────────────────────────────────────────────────────────────────

def _propagate_label(
    fixed_ants, label_path: str, fwd_transforms: list
):
    """Label'ı NN interpolation ile warp et."""
    label_ants = ants.image_read(label_path)
    propagated = ants.apply_transforms(
        fixed        = fixed_ants,
        moving       = label_ants,
        transformlist= fwd_transforms,
        interpolator = "nearestNeighbor",
    )
    return propagated


def _defrag(arr: np.ndarray, min_ratio: float = 0.10) -> np.ndarray:
    """
    Fragmentation otomatik duzeltme.

    Propagasyon sonrasi label birden fazla ayrik parcaya bolunebilir.
    Bu fonksiyon en buyuk connected component'i korur, geri kalanlardan
    sadece ana hacmin %10'undan buyuk olanlari tutar — kucuk kopuk parcalari siler.

    min_ratio: kucuk parcalarin tutulmasi icin ana parca hacmine orani
    """
    from scipy import ndimage as _ndi
    binary = (arr > 0)
    if not binary.any():
        return arr.astype(np.uint8)

    labeled, n_cc = _ndi.label(binary)
    if n_cc <= 1:
        return arr.astype(np.uint8)   # zaten tek parca

    # Her component'in hacmini hesapla
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # arka plan
    main_size = sizes.max()

    # Ana + min_ratio'yu gecen komponentleri tut
    keep = np.zeros_like(labeled, dtype=bool)
    for comp_id in range(1, n_cc + 1):
        if sizes[comp_id] >= main_size * min_ratio:
            keep |= (labeled == comp_id)

    result = np.where(keep, arr, 0).astype(np.uint8)
    return result


def _find_forward_transforms(warp_dir: str) -> list:
    """
    warp dizininde ANTs forward transform dosyalarını döndür.
    ANTsPy outprefix='fwd_transform_' ile üretilen dosya adları:
      fwd_transform_1Warp.nii.gz      (displacement field — önce uygulanır)
      fwd_transform_0GenericAffine.mat (affine — sonra uygulanır)
    apply_transforms için sıra: [Warp, Affine]
    """
    warp_file   = None
    affine_file = None

    for fname in os.listdir(warp_dir):
        fpath = os.path.join(warp_dir, fname)
        if fname.endswith("1Warp.nii.gz"):
            warp_file = fpath
        elif fname.endswith("0GenericAffine.mat"):
            affine_file = fpath

    transforms = []
    if warp_file:
        transforms.append(warp_file)
    if affine_file:
        transforms.append(affine_file)

    if not transforms:
        # Fallback: tüm fwd_transform* dosyaları alfabetik sırayla
        for fname in sorted(os.listdir(warp_dir)):
            if fname.startswith("fwd_transform") and (
                fname.endswith(".mat") or fname.endswith(".nii.gz")
            ):
                transforms.append(os.path.join(warp_dir, fname))

    if not transforms:
        raise FileNotFoundError(f"Transform dosyası bulunamadı: {warp_dir}")

    return transforms


def _get_label_names(label_dir: str, label_filter: list = None) -> list:
    names = [
        os.path.splitext(os.path.splitext(f)[0])[0]
        for f in os.listdir(label_dir)
        if f.endswith(".nii.gz")
    ]
    if label_filter:
        names = [n for n in names if n in label_filter]
    return sorted(names)


# ──────────────────────────────────────────────────────────────────────────────
# QC
# ──────────────────────────────────────────────────────────────────────────────

def _qc_label_overlay(
    t1: np.ndarray, label: np.ndarray, title: str, path: str, n: int = 3
):
    if label.sum() > 0:
        z_center = int(np.where(label.sum(axis=(0, 1)) > 0)[0].mean())
    else:
        z_center = t1.shape[2] // 2
    z_indices = np.clip(
        np.linspace(z_center - 10, z_center + 10, n, dtype=int),
        0, t1.shape[2] - 1
    )
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.suptitle(title, fontsize=10)
    for i, z in enumerate(z_indices):
        sl_t1  = t1[:, :, z]
        sl_lab = label[:, :, z]
        axes[i].imshow(sl_t1.T, cmap="gray", origin="lower",
                       vmin=np.percentile(sl_t1, 1), vmax=np.percentile(sl_t1, 99))
        if sl_lab.sum() > 0:
            axes[i].contour(sl_lab.T, levels=[0.5], colors=["red"], linewidths=1)
        axes[i].set_title(f"z={z}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=QC_DPI, bbox_inches="tight")
    plt.close(fig)


def _qc_label_zoom(
    t1: np.ndarray, label: np.ndarray, label_name: str,
    spacing: tuple, path: str, pad: int = 10
):
    coords = np.where(label > 0)
    if len(coords[0]) == 0:
        return
    z_min, z_max = int(coords[2].min()), int(coords[2].max())
    z_mid = (z_min + z_max) // 2
    x_min = max(0, int(coords[0].min()) - pad)
    x_max = min(t1.shape[0] - 1, int(coords[0].max()) + pad)
    y_min = max(0, int(coords[1].min()) - pad)
    y_max = min(t1.shape[1] - 1, int(coords[1].max()) + pad)

    sl_t1  = t1[x_min:x_max, y_min:y_max, z_mid]
    sl_lab = label[x_min:x_max, y_min:y_max, z_mid]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(sl_t1.T, cmap="gray", origin="lower",
              vmin=np.percentile(sl_t1, 1), vmax=np.percentile(sl_t1, 99))
    if sl_lab.sum() > 0:
        ax.contour(sl_lab.T, levels=[0.5], colors=["red"], linewidths=1.5)
    ax.set_title(f"{label_name} (sol, z={z_mid})")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=QC_DPI, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="İP-3 Propagate ve Morphometrics")
    parser.add_argument("--subject",     type=str, default=None)
    parser.add_argument("--candidate",   type=str, default=None,
                        choices=list(WARP_CANDIDATES.keys()))
    parser.add_argument("--labels-only", type=str, default=None,
                        help="Virgülle ayrılmış label isimleri: STh,RN,Hb")
    args = parser.parse_args()

    subjects   = [args.subject]   if args.subject   else SUBJECTS
    candidates = [args.candidate] if args.candidate else list(WARP_CANDIDATES.keys())
    label_filter = args.labels_only.split(",") if args.labels_only else None

    print(f"\n{'='*60}")
    print(f"  İP-3 Propagate — {len(subjects)} hasta × {len(candidates)} aday")
    print(f"{'='*60}")

    results = []
    for subj in subjects:
        for cand in candidates:
            r = propagate_subject_candidate(subj, cand, label_filter)
            results.append(r)

    ok      = sum(1 for r in results if r["status"] == "SUCCESS")
    skip    = sum(1 for r in results if r["status"] == "SKIPPED")
    fail    = len(results) - ok - skip
    print(f"\n{'='*60}")
    print(f"  İP-3 TAMAMLANDI — ✓ {ok} başarılı, ⊘ {skip} atlandı, ✗ {fail} hata")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
